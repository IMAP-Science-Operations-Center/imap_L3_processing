# Phase 4 — Charge is computed without the Table 45 energy bounds

> **Depends on:** none
> **Blocks:** Phase 5 (the v001 Range 2 charge-fit tracks are not in Z order inside the old bounds, so the end-to-end run would hit the `assert` crash without this phase)
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — assert NaN handling explicitly, no conditional assertions
> - [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — workaround vs fix: NaN is the agreed per-event fill for an undefined interpolation, not a hidden failure
> - [`overview.md`](overview.md) — user decisions 2 (remove bounds) and 5 (NaN for undefined interpolation); Table 46 Si open issue
> - HIT Algorithm Document v1.11.05 §9.1.4 (Eq. 61–64, "Algorithm Application" steps 1–3), Table 45

## Goal

Per the HIT team (email note 4), the Table 45 E′/ΔE ranges are informational only. `compute_charge` must no longer return NaN just because ΔE or E′ fall outside them. The doc's interpolation (Eq. 62–64, with H–He / Fe–Ni extrapolation at the ends) runs for every Range 2/3/4 event.

Once the bounds are gone, inputs reach the interpolation where it is mathematically undefined:

- ΔE ≤ 0 or E′ ≤ 0. This is possible because the calibration is linear with a negative offset, and the power law has no real value there.
- Track values at E′ that are not strictly increasing in Z. This happens with today's tables outside the old bounds and with the new Table 46 inside them.

In those cases the event's charge is NaN, and the rest of the event and the file are processed normally. Today the non-monotone case raises `AssertionError("values are not increasing")`, which aborts the whole `direct-events` file.

## Gherkin Feature & Scenarios

```gherkin
Feature: Charge is computed without the Table 45 energy bounds

  Background:
    Given a range fit lookup whose tracks evaluated at E' give charges [3, 4, 5, 6]

  Scenario Outline: Charge is interpolated for energies outside the former Table 45 bounds
    Given the tracks evaluated at E' are [1, 2, 20, 200]
    When compute_charge is called for <detected_range> with delta_e <delta_e> and e_prime <e_prime>
    Then the range fit lookup is evaluated at e_prime <e_prime> for <detected_range>
    And the charge is A × delta_e^B using the pair of tracks at indices <index_1> and <index_2>

    Examples:
      | detected_range | delta_e | e_prime | index_1 | index_2 |
      | Range 2 side A | 430.1   | 860.1   | 2       | 3       |
      | Range 2 side B | 0.09    | 0.19    | 0       | 1       |
      | Range 3 side A | 860.1   | 4300.1  | 2       | 3       |
      | Range 3 side B | 0.19    | 0.9     | 0       | 1       |
      | Range 4 side A | 4300.1  | 4300.1  | 2       | 3       |
      | Range 4 side B | 0.9     | 0.9     | 0       | 1       |

  Scenario Outline: Charge is NaN when delta_e or e_prime is not positive
    Given the tracks evaluated at E' are [1, 2, 20, 200]
    When compute_charge is called for Range 2 side A with delta_e <delta_e> and e_prime <e_prime>
    Then the charge is NaN

    Examples:
      | delta_e | e_prime |
      | 0.0     | 50.0    |
      | -1.5    | 50.0    |
      | 5.0     | 0.0     |
      | 5.0     | -3.0    |

  Scenario Outline: Charge is NaN instead of an error when the tracks at E' are not strictly increasing
    Given the tracks evaluated at E' are <track_values>
    When compute_charge is called for Range 3 side A with delta_e 5.0 and e_prime 50.0
    Then the charge is NaN
    And no exception is raised

    Examples:
      | track_values        |
      | [1, 20, 2, 200]     |
      | [1, 2, 2, 200]      |
      | [1, NaN, 20, 200]   |
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/pha/science/calculate_pha.py`: `compute_charge`. Delete `ValidDetectorRange` and `valid_ranges`.

### Tests

- `tests/hit/l3/pha/science/test_calculate_pha.py`:
  - Delete `test_compute_charge_returns_nan_if_e_prime_or_delta_e_are_outside_of_range`; the behavior it specified is removed by this card.
  - Add the three tests below.
  - Keep `test_compute_charge`, the in-range interpolation and extrapolation cases. Update its `subTest` mock to the shared fixture only if that removes duplication without losing cases.

## TDD steps

### 4.1 Red — Scenario Outline: Charge is interpolated for energies outside the former Table 45 bounds

- Add `test_compute_charge_interpolates_energies_outside_former_table_45_bounds`:
  - For each Examples row, in a `subTest`, create `mock_lookup = Mock()` with `mock_lookup.evaluate_e_prime.return_value = (np.array([3, 4, 5, 6]), np.array([1, 2, 20, 200]))`.
  - `charge = compute_charge(detected_range, delta_e, e_prime, mock_lookup)`.
  - `mock_lookup.evaluate_e_prime.assert_called_once_with(detected_range, e_prime)`.
  - Compute the expected value independently with `math.log` in the test (B = log(Z₂/Z₁)/log(f₂/f₁), A = Z₁/f₁^B, expected = A·ΔE^B) and assert it with `self.assertAlmostEqual(expected, charge)`.
- These Examples come from the old bounds test: one just outside each min/max edge for every range, on both sides.
- Fails: the current gating returns NaN, and `evaluate_e_prime` is never called.

### 4.2 Green

- Delete `ValidDetectorRange` and `valid_ranges`. Remove the `if valid_ranges[...]` gate so `compute_charge` always evaluates the tracks and interpolates.
- Delete `test_compute_charge_returns_nan_if_e_prime_or_delta_e_are_outside_of_range`; it now contradicts the specified behavior.

### 4.3 Red — Scenario Outline: Charge is NaN when delta_e or e_prime is not positive

- Add `test_compute_charge_is_nan_when_delta_e_or_e_prime_is_not_positive`:
  - Same mock lookup returning increasing tracks.
  - For each Examples row in a `subTest`, assert `self.assertTrue(np.isnan(compute_charge(DetectedRange(DetectorRange.R2, DetectorSide.A), delta_e, e_prime, mock_lookup)))`.
- Fails:
  - ΔE ≤ 0 with Python floats: `(-1.5) ** B` gives a complex number, and `0.0 ** B` with negative B gives `ZeroDivisionError`, or a finite or complex value instead of NaN.
  - E′ ≤ 0: the mock still returns valid tracks, so a finite charge comes back.

### 4.4 Green

- At the top of `compute_charge`: `if delta_e <= 0 or e_prime <= 0: return np.nan`.

### 4.5 Red — Scenario Outline: Charge is NaN instead of an error when the tracks at E′ are not strictly increasing

- Add `test_compute_charge_is_nan_when_tracks_at_e_prime_are_not_strictly_increasing`:
  - For each `track_values` row in a `subTest`, the mock returns `(np.array([3, 4, 5, 6]), np.array(track_values, dtype=float))`.
  - Assert `np.isnan(compute_charge(DetectedRange(DetectorRange.R3, DetectorSide.A), 5.0, 50.0, mock_lookup))`. If the code raises, the test errors, which proves the "no exception" clause without a try/except.
- Fails: `[1, 20, 2, 200]` and `[1, NaN, 20, 200]` raise `AssertionError`; `[1, 2, 2, 200]` passes the `np.sort` check and then divides by `log(1) = 0`.

### 4.6 Green

- Replace `assert np.array_equal(deltas, np.sort(deltas)), ...` with `if not np.all(np.diff(deltas) > 0): return np.nan`. NaN entries make the comparison False, so they are covered too.

### 4.7 Refactor

- Rename locals to the doc's terms where it helps, e.g. `track_charges`, `track_values_at_e_prime`. `B` and `A` match Eq. 63–64, so keep them (lowercase only if ruff requires it).
- Annotate `compute_charge`'s parameters and return (`-> float`) if not already complete. Drop the now-unused `clip` import only if no longer used.
- Confirm `process_pha_event` still assigns `compute_charge`'s NaN straight into `EventOutput.charge`. `HitProcessor` already writes it into the NaN-initialised `charge` array, so the NaN reaches the CDF as fill with no extra code.

## Done criteria

- [ ] `ValidDetectorRange` and `valid_ranges` are removed, and so is the old bounds test
- [ ] Scenario Outline "outside the former bounds" passes for all 6 rows, with the lookup evaluated at the given E′ and the charge matching Eq. 62–64
- [ ] Scenario Outline "not positive" returns NaN for all 4 rows
- [ ] Scenario Outline "not strictly increasing" returns NaN for all 3 rows and raises nothing
- [ ] Existing `test_compute_charge` still passes
- [ ] `uv run python -m unittest tests.hit.l3.pha.science.test_calculate_pha` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
