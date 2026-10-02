# -*- coding: utf-8 -*-
"""The whole reader, end to end: photo in, structured numbers out - or an
explicit refusal.

photo -> normalize contrast -> deskew -> detect printed grid
      -> cut into cells -> read each cell with a voting ensemble
      -> accept only confident cells, abstain on the rest
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline import deskew, normalize_contrast, segment_cells
from recognizers import DEFAULT_VARIANTS, SheetReading, Variant, read_sheet


class GridNotFoundError(Exception):
    """The printed grid could not be located, so there is nothing to read.

    This is a deliberate, loud failure. The alternative - carving the
    image into a guessed number of equal rectangles and reading whatever
    lands in them - produces a full sheet of confident, meaningless
    numbers, which is the single worst outcome for a system that feeds a
    database.
    """

    def __init__(self, rows: int, cols: int, expected: tuple[int, int] | None):
        expectation = f", expected {expected[0]}x{expected[1]}" if expected else ""
        super().__init__(
            f"could not locate the form grid (found {rows}x{cols} cells{expectation})"
        )
        self.rows = rows
        self.cols = cols


@dataclass
class FormReading:
    sheet: SheetReading
    rows: int
    cols: int

    @property
    def coverage(self) -> float:
        return self.sheet.coverage

    def as_grid(self) -> list[list[int | None]]:
        """The readable payload: digits where the reader is confident,
        None where a human needs to look."""
        return [[cell.digit for cell in row] for row in self.sheet.cells]


def read_form(
    image: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
    variants: tuple[Variant, ...] = DEFAULT_VARIANTS,
    min_agreement: int = 5,
    min_margin: float = 0.25,
    correct_skew: bool = True,
) -> FormReading:
    """Read a photographed form.

    Raises GridNotFoundError when the page is too degraded to locate the
    grid, or when the grid found doesn't match `expected_shape` - a form
    whose row/column count doesn't match the template it's supposed to be
    is a form that was mis-detected, and reading it anyway would silently
    shift every value into the wrong field.
    """
    working = normalize_contrast(image)
    if correct_skew:
        working = deskew(working)

    cells, (rows, cols) = segment_cells(working)

    if rows == 0 or cols == 0:
        raise GridNotFoundError(rows, cols, expected_shape)
    if expected_shape is not None and (rows, cols) != expected_shape:
        raise GridNotFoundError(rows, cols, expected_shape)

    sheet = read_sheet(cells, variants, min_agreement, min_margin)
    return FormReading(sheet=sheet, rows=rows, cols=cols)
