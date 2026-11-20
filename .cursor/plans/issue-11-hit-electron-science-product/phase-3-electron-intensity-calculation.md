# Phase 3 — Electron intensities follow Algorithm Document Eq. 32

> **Depends on:** Phase 1 (`ElectronIntensityFactors` supplies rate indices, ΔE, G, b)
> **Blocks:** Phase 5 (the processor calls `calculate_electron_intensities`)
> **Doc references:**
> - [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): `assert_allclose` with default tolerance, explicit NaN assertions, assert the whole result
> - [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): type annotations, fail explicitly on shape mismatch
> - [`phase-1-electron-intensity-factors-ancillary.md`](phase-1-electron-intensity-factors-ancillary.md): factor definitions
> - [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf): Eq. 14 / Eq. 32 (intensity, the 60·Δt divisor), §9.3, Table 53

## Goal

Given the L1B `ialirtrates` (livetime-corrected counts per 1-minute frame) and their ±σ, together with the intensity factors, `calculate_electron_intensities` returns an `ElectronIntensities`. For each side it holds `(epoch, 6)` arrays of intensity and plus/minus uncertainty, computed as:

```
J[t, i]       = (q[t, idx_i] − b_i) / (60 · ΔE_i · G_i)
delta±[t, i]  = σ±[t, idx_i]        / (60 · ΔE_i · G_i)
```

Here `idx_i` is the A-side index for `sunward` and the B-side index for `antisunward`. A NaN rate gives NaN for that epoch and bin only.

## Gherkin Feature & Scenarios

```gherkin
Feature: Electron intensities follow Algorithm Document Eq. 32

  Background:
    Given ElectronIntensityFactors with a_side_rate_indices [0..5], b_side_rate_indices [10..15],
      delta_e_mev all 0.5, geometry_factor_cm2_sr all 2.0 and background all 0

  Scenario: A-side and B-side rates are converted to intensities with the 60 s frame divisor
    Given 2 epochs of ialirtrates where index k holds 60 * (k + 1) counts
    And rate uncertainties equal to 6 * (k + 1)
    When calculate_electron_intensities is called
    Then sunward intensity at every epoch is [1, 2, 3, 4, 5, 6]
    And antisunward intensity at every epoch is [11, 12, 13, 14, 15, 16]
    And sunward delta_plus and delta_minus at every epoch are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    And antisunward delta_plus and delta_minus at every epoch are [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

  Scenario: Per-bin background, bin width and geometry factor are each applied to their own bin
    Given delta_e_mev [0.3, 0.1, 0.4, 0.4, 0.8, 88], geometry_factor_cm2_sr from doc Table 53
      and background [6, 0, 0, 0, 0, 0]
    And 1 epoch of ialirtrates of 600 counts on every index
    When calculate_electron_intensities is called
    Then sunward intensity bin 0 is (600 - 6) / (60 * 0.3 * 1.7792)
    And sunward intensity bin i > 0 is 600 / (60 * delta_e_mev[i] * geometry_factor_cm2_sr[i])

  Scenario: A NaN rate yields NaN only in its own epoch and bin
    Given 2 epochs of ialirtrates where epoch 1, index 12 is NaN
    When calculate_electron_intensities is called
    Then antisunward intensity at epoch 1, bin 2 is NaN
    And no other sunward or antisunward intensity is NaN

  Scenario: Rates without 20 I-ALiRT entries per epoch are rejected
    Given ialirtrates of shape (2, 19)
    When calculate_electron_intensities is called
    Then a ValueError mentioning "20 I-ALiRT rates" is raised
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/electron/science/electron_intensity.py` *(new)*:
  - `SECONDS_PER_L1B_FRAME = 60`
  - `ElectronIntensities` dataclass: `sunward`, `sunward_delta_plus`, `sunward_delta_minus`, `antisunward`, `antisunward_delta_plus`, `antisunward_delta_minus`
  - `calculate_electron_intensities(rates: np.ndarray, rates_delta_plus: np.ndarray, rates_delta_minus: np.ndarray, factors: ElectronIntensityFactors) -> ElectronIntensities`

### Tests

- `tests/hit/l3/electron/science/test_electron_intensity.py` *(new)*

## TDD steps

### 3.1 Red — Scenario: A-side and B-side rates are converted to intensities with the 60 s frame divisor

- Write `test_converts_a_and_b_side_rates_to_intensities_with_frame_divisor`.
- Build the Background factors directly: `ElectronIntensityFactors(a_side_rate_indices=np.arange(6), b_side_rate_indices=np.arange(10, 16), ...)`.
- Rates are `np.tile(60 * (np.arange(20) + 1), (2, 1))` and the uncertainties are `np.tile(6 * (np.arange(20) + 1), (2, 1))`. Pass the same array for plus and minus. The plus/minus wiring is still checked because the minus array differs in step 3.3.
- Assert all six output arrays with `assert_allclose` against the expected `(2, 6)` arrays. With ΔE·G = 1 the divisor is exactly 60, so the expected values are exact.
- Fails: `ModuleNotFoundError`.

### 3.2 Green

- `per_bin_divisor = SECONDS_PER_L1B_FRAME * factors.delta_e_mev * factors.geometry_factor_cm2_sr`.
- `sunward = rates[:, factors.a_side_rate_indices] / per_bin_divisor`; deltas and antisunward follow the same pattern.
- Do **not** subtract the background yet. The Background has b = 0, so nothing here drives it; step 3.4 does.

### 3.3 Refactor + extend 3.1

- Change the 3.1 test so `rates_delta_minus` is `3 * (k + 1)`, distinct from plus, and expect `[0.05 … 0.3]` / `[0.55 … 0.8]` for the minus arrays. This proves plus and minus are not swapped. It belongs in the same scenario, not a new test.
- Extract a `_convert_side(rates, rates_delta_plus, rates_delta_minus, indices, factors) -> tuple[np.ndarray, np.ndarray, np.ndarray]` helper so the A and B sides share one code path.

### 3.4 Red — Scenario: Per-bin background, bin width and geometry factor are each applied to their own bin

- Write `test_applies_each_bins_background_width_and_geometry_factor`. Use the Table 53 ΔE and G values, background `[6, 0, 0, 0, 0, 0]` and a single epoch of 600 on all 20 indices.
- Compute the expected values in the test from the Scenario's formulas and assert the full sunward and antisunward `(1, 6)` arrays with `assert_allclose`.
- This drives correct per-bin broadcasting, catching any reversed or misaligned factor array. Phase 1 never subtracted a background, so this is the first scenario where a non-zero `b` changes the result.
- Fails: bin 0 is `600 / divisor` because the background is not subtracted yet.

### 3.4a Green

- Subtract `factors.background` from the selected rates in `_convert_side`, for the intensity only, not the uncertainties.

### 3.5 Red — Scenario: A NaN rate yields NaN only in its own epoch and bin

- Write `test_nan_rate_yields_nan_only_in_its_epoch_and_bin`, with rates of 2 epochs and `rates[1, 12] = np.nan`.
- Assert `np.argwhere(np.isnan(result.antisunward))` equals `[[1, 2]]` and `np.isnan(result.sunward).any()` is `False`. Also assert the non-NaN values with `assert_allclose`, so the result is fully checked.

### 3.6 Green

- Expected to pass already via numpy NaN propagation. If so, fold the NaN assertion into the 3.1 test instead of a separate test, because there is no new production behavior. Keep it separate only if it drove a change.

### 3.7 Red — Scenario: Rates without 20 I-ALiRT entries per epoch are rejected

- Write `test_rejects_rates_without_twenty_ialirt_entries`, with rates of shape `(2, 19)`, and assert `ValueError` with `"20 I-ALiRT rates"` in the message.
- Fails: `IndexError` on index 15+, or silently wrong output.

### 3.8 Green

- Validate that `rates.shape[1] == IALIRT_RATE_COUNT`, importing the constant from Phase 1's module. Raise `ValueError(f"Expected {IALIRT_RATE_COUNT} I-ALiRT rates per epoch, got {rates.shape[1]}")`.

## Done criteria

- [ ] Scenario "60 s frame divisor" passes with distinct plus/minus uncertainties, asserting all six arrays
- [ ] Scenario "per-bin factors" passes using doc Table 53 values and a non-zero background
- [ ] Scenario "NaN only in own bin" is covered, either as its own test or folded into 3.1 with the reason recorded
- [ ] Scenario "rejects wrong rate count" passes
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
