# -*- coding: utf-8 -*-
"""Stage 1 of the reader: find the printed grid on the page and cut it
into per-cell images.

The ruled lines a form is printed with are usually the most reliable
structure in a bad photo - far more reliable than trying to locate the
handwriting directly. Finding them first turns "read this page" into
"read these 40 small, independent images", which is a much easier
problem and one where a per-cell confidence score actually means
something.
"""
from __future__ import annotations

import numpy as np


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    k = 2 * radius + 1
    padded = np.pad(img, radius, mode="edge")
    out = np.zeros_like(img)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    return out / (k * k)


def normalize_contrast(img: np.ndarray) -> np.ndarray:
    """Stretch intensities back to the full 0..1 range.

    Optical blur spreads a thin printed rule over several pixels and
    drops its peak well below any fixed threshold; rescaling puts it back
    within reach. This is deliberately a GLOBAL stretch: an earlier
    attempt here subtracted a locally-blurred background estimate to also
    cancel uneven lighting, and it measurably made things worse - the
    local window around a full-width rule is itself full of that rule, so
    the subtraction erased the very lines it was meant to rescue.

    Severe lighting gradients are therefore NOT handled here, and the
    pipeline is honest about failing on them (see `read_form`) rather
    than returning a confident wrong answer.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo <= 1e-9:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _line_positions(profile: np.ndarray, min_fill: float) -> list[int]:
    """Given a 1-D projection profile of a BINARIZED image (each value =
    the fraction of that row/column that is ink), return the center of
    each run of consecutive line-like values.

    Thresholding on "what fraction of this row is ink" rather than on
    "how dark is this row relative to the darkest row" is what makes
    this survive a noisy photo: a printed rule spans essentially the
    whole width, and no amount of handwriting in a row of cells ever
    does. Scaling against the max instead (the obvious first attempt)
    breaks as soon as sensor noise lifts the floor.
    """
    if profile.size == 0:
        return []
    is_line = profile >= min_fill

    positions: list[int] = []
    start: int | None = None
    for i, flag in enumerate(is_line):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            positions.append((start + i - 1) // 2)
            start = None
    if start is not None:
        positions.append((start + len(is_line) - 1) // 2)
    return positions


def detect_grid(
    img: np.ndarray,
    binarize_at: float = 0.5,
    min_fill: float = 0.7,
) -> tuple[list[int], list[int]]:
    """Return (horizontal_line_rows, vertical_line_cols)."""
    binary = (img >= binarize_at).astype(np.float64)
    return (
        _line_positions(binary.mean(axis=1), min_fill),
        _line_positions(binary.mean(axis=0), min_fill),
    )


def segment_cells(
    img: np.ndarray,
    margin: int = 2,
    binarize_at: float = 0.5,
    min_fill: float = 0.7,
) -> tuple[list[list[np.ndarray]], tuple[int, int]]:
    """Cut the image into cells using the detected grid.

    `margin` trims a few pixels inside each cell so the printed rules
    themselves don't bleed into the cell image and get mistaken for ink.
    Returns (cells, (rows, cols)).
    """
    h_lines, v_lines = detect_grid(img, binarize_at, min_fill)
    if len(h_lines) < 2 or len(v_lines) < 2:
        return [], (0, 0)

    cells: list[list[np.ndarray]] = []
    for top, bottom in zip(h_lines, h_lines[1:]):
        row_cells: list[np.ndarray] = []
        for left, right in zip(v_lines, v_lines[1:]):
            y0, y1 = top + margin, bottom - margin
            x0, x1 = left + margin, right - margin
            if y1 <= y0 or x1 <= x0:
                row_cells.append(np.zeros((1, 1)))
            else:
                row_cells.append(img[y0:y1, x0:x1])
        cells.append(row_cells)

    return cells, (len(cells), len(cells[0]) if cells else 0)


def estimate_skew(img: np.ndarray, max_degrees: float = 4.0, step: float = 0.5) -> float:
    """Estimate page rotation by finding the angle whose horizontal
    projection profile has the highest variance - when the ruled lines
    are level, their ink concentrates into a few rows and the variance
    spikes. Returns degrees (positive = image is rotated clockwise)."""
    best_angle, best_score = 0.0, -1.0
    angle = -max_degrees
    while angle <= max_degrees + 1e-9:
        rotated = rotate(img, angle)
        score = float(rotated.mean(axis=1).var())
        if score > best_score:
            best_angle, best_score = angle, score
        angle += step
    return -best_angle


def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the center with nearest-neighbour sampling.

    Hand-rolled rather than pulled from an imaging library so the whole
    pipeline stays dependency-light and every step is inspectable.
    """
    if degrees == 0:
        return img
    theta = np.deg2rad(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y_rel, x_rel = yy - cy, xx - cx
    src_y = (cos_t * y_rel + sin_t * x_rel + cy).round().astype(int)
    src_x = (-sin_t * y_rel + cos_t * x_rel + cx).round().astype(int)

    inside = (src_y >= 0) & (src_y < h) & (src_x >= 0) & (src_x < w)
    out = np.zeros_like(img)
    out[inside] = img[src_y[inside], src_x[inside]]
    return out


def deskew(img: np.ndarray, max_degrees: float = 4.0, step: float = 0.5) -> np.ndarray:
    """Straighten the page before segmenting it. A 2-degree tilt is
    invisible to a human and completely breaks row/column segmentation.

    Note the negation: `estimate_skew` reports how far the page IS
    tilted, so correcting it means rotating by the opposite amount. An
    earlier version of this function applied the angle directly and
    cheerfully tilted every page twice as far - which the segmentation
    test caught immediately, because the cell count never recovered.
    """
    return rotate(img, -estimate_skew(img, max_degrees, step))
