# 2026-08-12: Root-cause of 12 failing test_pmrc_alignment.py tests

Investigation of the 12 failures in `tests/test_pmrc_alignment.py` (committed
failing as WIP at original tip `d3776d4`, "defense draft updates 1/2";
functionally unchanged since). Not caused by the repo cleanup.

## Setup of the test

`tests/gold_fixtures.py::simulate_pmrc_rows` seeds the model state from the
gold CSV's first row (age, hd, tpa, ba), projects forward 1 year/step to
`end_age=35`, and at each row re-derives derived fields (V=volume_tvob via
`pmrc.yield_predict(unit="TVOB")`, DW, Dq, products). Tests compare the
model's rows to the gold CSVs in `test_csvs/`.

## Root cause 1 (dominant): TVOB volume equation divergence — all 6 scenarios

The gold CSV's `V` (TVOB) column is **not produced by the Python model's
`yield_predict(unit="TVOB")`**, even when fed the gold's exact (age,tpa,hd,ba):

| age | gold V | model TVOB | ratio |
|----:|-------:|-----------:|------:|
|  5 | 301.76 |  68.77 | 4.39 |
|  7 | 757.42 | 453.09 | 1.67 |
|  9 | 1304.81 | 1144.44 | 1.14 |
| 11 | 1891.48 | 1980.83 | 0.96 |
| 15 | 3095.97 | 3696.22 | 0.84 |
| 25 | 5993.45 | 7179.91 | 0.84 |
| 35 | 8582.90 | 9566.98 | 0.90 |

Pattern: model underpredicts 4.4x at age 5, crosses ~age 11, overpredicts
~15-22% at mid/late rotation. Signature of the `ln(TPA)/A`, `ln(HD)/A`,
`ln(BA)/A` correction terms dominating at young ages.

- The Python `yield_predict` **exactly matches the documented PMRC equation**
  (`docs/core/pmrc_model_technical_report.md` §7.1) and PUCP TVOB coefficients
  `(0.0, 0.268552, 1.368844, -7.466863, 8.934524, 3.553411)`.
- Those coefficients are **identical across all of git history** (never
  changed), and the old `tvob()` method used the same formula. So the model
  has NEVER produced 301.76 at age 5.
- **The gold CSV's `V` comes from an external source** (the R reference model
  / workbook), not from this Python model at any commit.
- Corroborating: the model's **DWIB matches the gold DW almost exactly from
  age 15+** (gold 38.26 vs model 38.22 at age 15) and **HD matches exactly**
  at all ages. So the model is in the right equation family and the growth
  projection is essentially sound; the **TVOB equation specifically** differs
  from the gold's generator.

A single coefficient change cannot fix it (the young-age 4.4x and mid-rotation
0.85x are inconsistent) → the gold's TVOB uses a **structurally different
equation/coefficients** than the documented one the Python implements.

## Root cause 2 (secondary): BA drift breaks the thin trigger — thin scenarios

Model's BA projection drifts ~4% below the gold by mid-rotation (e.g. at age
15: model BA=144.4 vs gold BA=150.9). The thin test infers
`ba_threshold = gold pre-thin BA = 150.9` and `should_thin` requires
`state.ba >= ba_threshold`. Since the model's BA at age 15 (144.4) < 150.9,
`should_thin` never fires → no duplicate-age thin row → model produces 31
rows vs gold's 32. (`test_pmrc_duplicate_age_rows` and the thin
`matches_gold_csv_rows` row-count checks fail here.)

The BA drift itself is small and likely upstream of the same equation-family
mismatch (the model's `ba_project`/`tpa_project` differ slightly from the gold's
generator; TPA drifts ~4% by age 12, BA follows).

## What is authoritative? (unresolved modeling decision)

AGENTS.md #1: "Preserve the deterministic PMRC model as the baseline truth."
But the gold CSVs are labeled "gold" reference data. Conflict:
- **Model authoritative** → gold CSVs are stale/external and should be
  regenerated from the current model (or tests marked xfail with rationale).
  Risk: the TVOB equation may genuinely be wrong vs the real PMRC R model.
- **Gold authoritative** → the model's TVOB equation/coefficients (and BA/TPA
  projection) have a real bug vs the R reference and must be fixed to match.
  Requires the R script (not in repo) to confirm the correct equation.

Evidence leans slightly toward "model faithfully implements the docs; gold is
external/stale" (coeffs match the technical report; DWIB + HD corroborate), but
the 4.4x young-age TVOB gap is large enough that a real model bug cannot be
ruled out without the R reference. **User must decide direction before any
model or test changes.**

## How to reproduce

```
uv run pytest tests/test_pmrc_alignment.py -q   # 12 failed, 5 passed
# isolated yield-equation check (feed gold inputs directly):
uv run python -c "from tests.gold_fixtures import *; from core.pmrc_model import PMRCModel; ..."
```
