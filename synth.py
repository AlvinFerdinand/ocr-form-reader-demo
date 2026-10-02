# -*- coding: utf-8 -*-
"""Synthetic form generator: builds a ruled grid with a digit written in
each cell, then degrades it the way a real phone photo of a paper form
degrades - blur, sensor noise, uneven lighting, slight rotation.

This exists so the pipeline can be measured against known ground truth.
A reader you can't score is a reader you can't tune.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glyphs import GLYPH_HEIGHT, GLYPH_WIDTH, glyph


@dataclass
class SyntheticForm:
    image: np.ndarray          # 2-D float array, 0.0 = white paper, 1.0 = black ink
    truth: list[list[int]]     # the digit actually written in each cell
    rows: int
    cols: int


def render_form(
    truth: list[list[int]],
    cell_h: int = 28,
    cell_w: int = 20,
    scale: int = 3,
    rule_width: int = 2,
    rng: np.random.Generator | None = None,
) -> SyntheticForm:
    """Draw a clean ruled form with the given digits, one per cell.

    `rule_width` matters more than it looks: a 1-pixel rule survives in a
    clean render but disappears entirely once the image is blurred, and
    then grid detection has nothing to lock onto. Printed rules in a real
    photographed form are several pixels wide, so the synthetic form
    models that rather than the easier, unrealistic case.
    """
    rng = rng or np.random.default_rng(0)
    rows, cols = len(truth), len(truth[0])
    h, w = rows * cell_h + rule_width, cols * cell_w + rule_width
    img = np.zeros((h, w), dtype=np.float64)

    # Table rules (the printed grid lines on the paper form)
    for r in range(rows + 1):
        y = min(r * cell_h, h - rule_width)
        img[y:y + rule_width, :] = 1.0
    for c in range(cols + 1):
        x = min(c * cell_w, w - rule_width)
        img[:, x:x + rule_width] = 1.0

    gh, gw = GLYPH_HEIGHT * scale, GLYPH_WIDTH * scale
    for r in range(rows):
        for c in range(cols):
            g = np.kron(glyph(truth[r][c]), np.ones((scale, scale)))
            top = r * cell_h + (cell_h - gh) // 2
            left = c * cell_w + (cell_w - gw) // 2
            img[top:top + gh, left:left + gw] = np.maximum(
                img[top:top + gh, left:left + gw], g
            )

    return SyntheticForm(image=img, truth=truth, rows=rows, cols=cols)


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


def degrade(
    img: np.ndarray,
    blur_radius: int = 1,
    noise_sigma: float = 0.05,
    lighting_gradient: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate photographing the form: optics blur, sensor noise, and a
    lighting gradient from holding the phone at an angle."""
    rng = rng or np.random.default_rng(0)
    out = _box_blur(img, blur_radius)

    if lighting_gradient:
        h, w = out.shape
        ramp = np.linspace(0.0, lighting_gradient, w)[None, :] * np.ones((h, 1))
        out = np.clip(out - ramp, 0.0, 1.0)

    if noise_sigma:
        out = out + rng.normal(0.0, noise_sigma, size=out.shape)

    return np.clip(out, 0.0, 1.0)
