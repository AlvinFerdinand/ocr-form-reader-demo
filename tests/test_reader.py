# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from pipeline import deskew, detect_grid, estimate_skew, normalize_contrast, rotate, segment_cells
from reader import GridNotFoundError, read_form
from recognizers import read_cell
from synth import degrade, render_form

ROWS, COLS = 5, 8


def make_truth(seed: int = 7) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [[int(rng.integers(0, 10)) for _ in range(COLS)] for _ in range(ROWS)]


# --------------------------------------------------------------- grid ----

def test_detect_grid_finds_every_printed_rule():
    form = render_form(make_truth())
    h_lines, v_lines = detect_grid(form.image)
    assert len(h_lines) == ROWS + 1
    assert len(v_lines) == COLS + 1


def test_segment_cells_returns_one_cell_per_form_box():
    form = render_form(make_truth())
    cells, shape = segment_cells(form.image)
    assert shape == (ROWS, COLS)
    assert all(len(row) == COLS for row in cells)


def test_handwriting_rows_are_not_mistaken_for_printed_rules():
    # A row of digits is dense, but never spans the full width the way a
    # printed rule does - which is exactly why the detector thresholds on
    # "fraction of this row that is ink" rather than on relative darkness.
    form = render_form([[8] * COLS for _ in range(ROWS)])  # 8 is the densest glyph
    _, shape = segment_cells(form.image)
    assert shape == (ROWS, COLS)


# -------------------------------------------------------------- skew ----

def test_estimate_skew_reports_how_far_the_page_is_tilted():
    # Convention: estimate_skew reports the tilt itself, so correcting it
    # means rotating by the NEGATIVE of this value (see deskew).
    form = render_form(make_truth())
    tilted = rotate(form.image, 2.0)
    assert estimate_skew(tilted) == pytest.approx(2.0, abs=0.5)


def test_deskew_restores_segmentation_that_tilt_had_broken():
    form = render_form(make_truth())
    tilted = rotate(form.image, 2.0)

    _, broken_shape = segment_cells(tilted)
    assert broken_shape != (ROWS, COLS)  # a 2-degree tilt is enough to break it

    _, fixed_shape = segment_cells(deskew(tilted))
    assert fixed_shape == (ROWS, COLS)


# ---------------------------------------------------------- contrast ----

def test_normalize_contrast_stretches_to_full_range():
    faint = render_form(make_truth()).image * 0.4
    out = normalize_contrast(faint)
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(0.0)


def test_normalize_contrast_handles_a_blank_image_without_dividing_by_zero():
    assert normalize_contrast(np.zeros((10, 10))).max() == 0.0


# ------------------------------------------------------------ reading ---

def test_clean_form_is_read_completely_and_correctly():
    truth = make_truth()
    reading = read_form(render_form(truth).image, expected_shape=(ROWS, COLS))
    correct, wrong, abstained = reading.sheet.score_against(truth)
    assert (correct, wrong, abstained) == (ROWS * COLS, 0, 0)


def test_degraded_form_loses_coverage_but_never_returns_a_wrong_digit():
    # The central claim of the whole design: degradation is allowed to
    # cost coverage, never correctness.
    truth = make_truth()
    noisy = degrade(render_form(truth).image, blur_radius=1, noise_sigma=0.12,
                    rng=np.random.default_rng(4))

    reading = read_form(noisy, expected_shape=(ROWS, COLS))
    correct, wrong, abstained = reading.sheet.score_against(truth)

    assert wrong == 0
    assert abstained > 0          # it did give some cells up
    assert correct > 0            # but it is not simply abstaining on everything


def test_strict_threshold_prevents_the_wrong_answers_a_loose_one_produces():
    # Measured, not assumed: at this noise level a permissive ensemble
    # confidently reports digits that are wrong, and the strict default
    # does not. This is the entire justification for the default.
    truth = make_truth()
    noisy = degrade(render_form(truth).image, blur_radius=1, noise_sigma=0.30,
                    rng=np.random.default_rng(11))

    loose = read_form(noisy, expected_shape=(ROWS, COLS), min_agreement=1, min_margin=0.0)
    strict = read_form(noisy, expected_shape=(ROWS, COLS), min_agreement=5, min_margin=0.25)

    _, loose_wrong, _ = loose.sheet.score_against(truth)
    _, strict_wrong, _ = strict.sheet.score_against(truth)

    assert loose_wrong > 0        # the permissive setting really does err here
    assert strict_wrong == 0      # the shipped default does not
    assert strict.coverage < loose.coverage  # and that costs coverage - honestly


def test_unreadable_page_is_refused_instead_of_guessed():
    truth = make_truth()
    ruined = degrade(render_form(truth).image, blur_radius=2, noise_sigma=0.18,
                     lighting_gradient=0.3, rng=np.random.default_rng(5))

    with pytest.raises(GridNotFoundError):
        read_form(ruined, expected_shape=(ROWS, COLS))


def test_grid_whose_shape_disagrees_with_the_template_is_refused():
    # Reading a mis-detected grid anyway would shift every value into the
    # wrong field - worse than reading nothing.
    form = render_form(make_truth())
    with pytest.raises(GridNotFoundError):
        read_form(form.image, expected_shape=(ROWS + 1, COLS))


def test_empty_cell_abstains_rather_than_inventing_a_digit():
    blank = np.zeros((24, 16))
    assert read_cell(blank).abstained


def test_as_grid_reports_none_exactly_where_it_abstained():
    truth = make_truth()
    noisy = degrade(render_form(truth).image, blur_radius=1, noise_sigma=0.12,
                    rng=np.random.default_rng(4))
    reading = read_form(noisy, expected_shape=(ROWS, COLS))

    grid = reading.as_grid()
    none_count = sum(1 for row in grid for value in row if value is None)
    _, _, abstained = reading.sheet.score_against(truth)
    assert none_count == abstained
