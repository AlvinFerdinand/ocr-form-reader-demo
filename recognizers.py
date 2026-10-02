# -*- coding: utf-8 -*-
"""Stage 2 of the reader: recognise a single cell - and, crucially,
decide when NOT to.

The design decision this module exists to demonstrate: the engine is
tuned so that a reading it reports is correct, rather than so that it
reports as many readings as possible. Cells it isn't sure about come
back as None and go to a human.

In an operational system that feeds a database, a silently wrong number
is far more expensive than a blank one: the blank gets noticed and
filled in, the wrong one gets trusted, propagated, and found three
months later during a reconciliation nobody budgeted for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glyphs import GLYPH_HEIGHT, GLYPH_WIDTH, all_glyphs

_TEMPLATES = all_glyphs()


@dataclass(frozen=True)
class Variant:
    """One preprocessing configuration. Running several slightly
    different variants over the same cell and comparing their answers is
    what makes disagreement - and therefore uncertainty - visible."""
    binarize_at: float
    erode: bool = False

    @property
    def name(self) -> str:
        return f"thr={self.binarize_at:g}{',erode' if self.erode else ''}"


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(binarize_at=0.35),
    Variant(binarize_at=0.45),
    Variant(binarize_at=0.55),
    Variant(binarize_at=0.65),
    Variant(binarize_at=0.45, erode=True),
    Variant(binarize_at=0.55, erode=True),
)


@dataclass
class CellReading:
    digit: int | None          # None == abstained; a human reads this cell
    votes: dict[int, int]      # how many variants voted for each digit
    agreement: int             # votes for the winning digit
    margin: float              # winner's template distance vs runner-up's

    @property
    def abstained(self) -> bool:
        return self.digit is None


def _erode(binary: np.ndarray) -> np.ndarray:
    """Shrink ink by one pixel - undoes the thickening that blur and
    over-eager binarization cause on a photographed page."""
    padded = np.pad(binary, 1, mode="constant", constant_values=0.0)
    out = np.ones_like(binary)
    for dy in range(3):
        for dx in range(3):
            out = np.minimum(out, padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]])
    return out


def _crop_to_ink(binary: np.ndarray) -> np.ndarray | None:
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return binary[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _resize_to_glyph(patch: np.ndarray) -> np.ndarray:
    """Area-average down to the template size (7x5)."""
    h, w = patch.shape
    out = np.zeros((GLYPH_HEIGHT, GLYPH_WIDTH), dtype=np.float64)
    y_edges = np.linspace(0, h, GLYPH_HEIGHT + 1).round().astype(int)
    x_edges = np.linspace(0, w, GLYPH_WIDTH + 1).round().astype(int)
    for r in range(GLYPH_HEIGHT):
        for c in range(GLYPH_WIDTH):
            # Clamp so a patch smaller than the template grid still
            # yields a non-empty slice for every output pixel instead of
            # averaging an empty array (which silently produces NaN and
            # poisons every distance comparison downstream).
            y0 = min(y_edges[r], h - 1)
            y1 = max(y_edges[r + 1], y0 + 1)
            x0 = min(x_edges[c], w - 1)
            x1 = max(x_edges[c + 1], x0 + 1)
            out[r, c] = patch[y0:y1, x0:x1].mean()
    return out


def recognize_with_variant(cell: np.ndarray, variant: Variant) -> tuple[int, float, float] | None:
    """Return (digit, best_distance, runner_up_distance) or None if the
    cell holds no usable ink at all."""
    binary = (cell >= variant.binarize_at).astype(np.float64)
    if variant.erode:
        binary = _erode(binary)

    patch = _crop_to_ink(binary)
    if patch is None or patch.size < 4:
        return None

    normalized = _resize_to_glyph(patch)
    distances = {
        digit: float(np.linalg.norm(normalized - template))
        for digit, template in _TEMPLATES.items()
    }
    ranked = sorted(distances.items(), key=lambda kv: kv[1])
    best_digit, best_distance = ranked[0]
    runner_up_distance = ranked[1][1]
    return best_digit, best_distance, runner_up_distance


def read_cell(
    cell: np.ndarray,
    variants: tuple[Variant, ...] = DEFAULT_VARIANTS,
    min_agreement: int = 5,
    min_margin: float = 0.25,
) -> CellReading:
    """Run every variant, then accept the answer only if the variants
    agree strongly enough AND the winning template beats the runner-up by
    a clear margin.

    `min_agreement` is the knob the whole design turns on. Raising it
    trades coverage for correctness; the demo's tests measure exactly
    what that trade costs at each setting instead of guessing.
    """
    votes: dict[int, int] = {}
    margins: dict[int, list[float]] = {}

    for variant in variants:
        result = recognize_with_variant(cell, variant)
        if result is None:
            continue
        digit, best, runner_up = result
        votes[digit] = votes.get(digit, 0) + 1
        margins.setdefault(digit, []).append(runner_up - best)

    if not votes:
        return CellReading(digit=None, votes={}, agreement=0, margin=0.0)

    winner, agreement = max(votes.items(), key=lambda kv: kv[1])
    winner_margin = float(np.mean(margins[winner]))

    if agreement < min_agreement or winner_margin < min_margin:
        return CellReading(digit=None, votes=votes, agreement=agreement, margin=winner_margin)

    return CellReading(digit=winner, votes=votes, agreement=agreement, margin=winner_margin)


@dataclass
class SheetReading:
    cells: list[list[CellReading]]

    @property
    def filled(self) -> int:
        return sum(1 for row in self.cells for c in row if not c.abstained)

    @property
    def total(self) -> int:
        return sum(len(row) for row in self.cells)

    @property
    def coverage(self) -> float:
        return self.filled / self.total if self.total else 0.0

    def score_against(self, truth: list[list[int]]) -> tuple[int, int, int]:
        """Return (correct, wrong, abstained) - the only three outcomes
        that matter, and the reason `wrong` is tracked separately from
        `abstained` instead of lumped into one accuracy percentage."""
        correct = wrong = abstained = 0
        for r, row in enumerate(self.cells):
            for c, reading in enumerate(row):
                if reading.abstained:
                    abstained += 1
                elif reading.digit == truth[r][c]:
                    correct += 1
                else:
                    wrong += 1
        return correct, wrong, abstained


def read_sheet(
    cells: list[list[np.ndarray]],
    variants: tuple[Variant, ...] = DEFAULT_VARIANTS,
    min_agreement: int = 5,
    min_margin: float = 0.25,
) -> SheetReading:
    return SheetReading(cells=[
        [read_cell(cell, variants, min_agreement, min_margin) for cell in row]
        for row in cells
    ])
