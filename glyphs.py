# -*- coding: utf-8 -*-
"""A tiny hardcoded 5x7 bitmap font for digits 0-9.

Deliberately NOT using a system font: a demo whose results depend on
which fonts happen to be installed produces different numbers on a
laptop than in CI, which makes every accuracy claim in the README
unverifiable. These glyphs are part of the repo, so every run - here, in
CI, on your machine - sees exactly the same pixels.
"""
from __future__ import annotations

import numpy as np

GLYPH_HEIGHT = 7
GLYPH_WIDTH = 5

_PATTERNS: dict[int, list[str]] = {
    0: [".###.",
        "#...#",
        "#..##",
        "#.#.#",
        "##..#",
        "#...#",
        ".###."],
    1: ["..#..",
        ".##..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        ".###."],
    2: [".###.",
        "#...#",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        "#####"],
    3: ["#####",
        "...#.",
        "..#..",
        "...#.",
        "....#",
        "#...#",
        ".###."],
    4: ["...#.",
        "..##.",
        ".#.#.",
        "#..#.",
        "#####",
        "...#.",
        "...#."],
    5: ["#####",
        "#....",
        "####.",
        "....#",
        "....#",
        "#...#",
        ".###."],
    6: ["..##.",
        ".#...",
        "#....",
        "####.",
        "#...#",
        "#...#",
        ".###."],
    7: ["#####",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        ".#...",
        ".#..."],
    8: [".###.",
        "#...#",
        "#...#",
        ".###.",
        "#...#",
        "#...#",
        ".###."],
    9: [".###.",
        "#...#",
        "#...#",
        ".####",
        "....#",
        "...#.",
        ".##.."],
}


def glyph(digit: int) -> np.ndarray:
    """Return a (7, 5) float array: 1.0 = ink, 0.0 = background."""
    rows = _PATTERNS[digit]
    return np.array([[1.0 if ch == "#" else 0.0 for ch in row] for row in rows],
                    dtype=np.float64)


def all_glyphs() -> dict[int, np.ndarray]:
    return {d: glyph(d) for d in _PATTERNS}
