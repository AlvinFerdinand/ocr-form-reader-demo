# ocr-form-reader-demo

> **Demo reconstruction of a real production system.** The original reads
> handwritten field forms photographed on phones and feeds the numbers
> into an ERP; it was built during my IT internship at **GSI Group**
> (Semarang, Indonesia). That system's code, data, forms and database
> schema are the company's and are **not** published here. This
> repository is a clean-room rebuild of the *approach* on synthetic data,
> written from scratch so the engineering decisions can be shown and
> measured without exposing anything that belongs to the company.

Photo of a ruled form in → structured digits out, **or an explicit
refusal**. Pure NumPy, no OCR engine, no system fonts, no network.

```
photo → normalize contrast → deskew → detect printed grid
      → cut into cells → read each cell with a voting ensemble
      → accept confident cells, abstain on the rest
```

## The design decision this exists to demonstrate

The engine is tuned so that **a reading it reports is correct**, not so
that it reports as many readings as possible. Cells it isn't sure about
come back as `None` and go to a human.

In a system that writes into a database, a silently wrong number costs
far more than a blank one: the blank gets noticed and filled in, the
wrong one gets trusted, propagated, and discovered months later during a
reconciliation nobody budgeted for.

## Measured behavior (these numbers come from the test suite, not from me)

Same form, same digits, increasing photo degradation:

| Condition | Coverage | Correct | **Wrong** |
|---|---|---|---|
| Clean render | 100% | 40/40 | **0** |
| Blur + noise σ=0.12 | 50% | 20 | **0** |
| Blur + noise σ=0.30 | 10% | 4 | **0** |
| Blur + noise + lighting gradient | — | — | **refused outright** |

Degradation is allowed to cost coverage. It is not allowed to cost
correctness.

### Why the strict threshold is set where it is

Measured on the same noisy image (σ=0.30), permissive vs shipped default:

| Ensemble setting | Coverage | **Wrong** |
|---|---|---|
| 1 of 6 variants must agree | 100% | **3** |
| 5 of 6 must agree + margin ≥ 0.25 | 10% | **0** |

`test_strict_threshold_prevents_the_wrong_answers_a_loose_one_produces`
asserts exactly this, so the justification stays true or the build goes
red.

**Honest caveat:** below roughly σ=0.25 on this synthetic data, the
permissive setting is *also* error-free, so the strict default is buying
nothing there and costing real coverage. The threshold is insurance
against the bad end of the input distribution, not a free improvement —
and the sweep above is how you find out which regime you're actually in.

## Things the pipeline refuses to do

- **Guess the grid.** If the printed rules can't be located, it raises
  `GridNotFoundError` instead of slicing the image into equal rectangles
  and reading whatever lands in them — which would produce a full sheet
  of confident, meaningless numbers.
- **Read a grid that doesn't match the template.** A form detected as
  6×8 when the template is 5×8 was mis-detected; reading it anyway shifts
  every value into the wrong field.
- **Invent a digit for an empty cell.**

## Run it

```bash
pip install numpy pytest
python -m pytest tests/ -v
```

14 tests: grid detection (including that dense handwriting isn't mistaken
for a printed rule), skew estimation and correction, contrast
normalization edge cases, the coverage-vs-correctness tradeoff above, and
every refusal path.

## Two bugs this repo's own tests caught while it was being written

Kept here because they're the useful part:

1. **`deskew` applied the correction with the wrong sign**, tilting every
   page twice as far instead of straightening it. The code looked
   obviously right; the segmentation test failed immediately because the
   cell count never recovered.
2. **A locally-adaptive contrast normalization made things measurably
   worse** and was removed. Subtracting a blurred background estimate
   cancels uneven lighting in general — but the window around a
   full-width printed rule is itself full of that rule, so the
   subtraction erased the lines it was meant to rescue. The committed
   version is a plain global stretch, and the lighting-gradient case is
   documented as unhandled rather than papered over.

## License

MIT.
