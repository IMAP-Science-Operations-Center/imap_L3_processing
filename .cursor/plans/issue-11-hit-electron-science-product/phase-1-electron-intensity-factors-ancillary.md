# Phase 1 — HIT electron intensity factors ancillary provides per-bin conversion factors

> **Depends on:** none
> **Blocks:** Phase 2 (dependencies parse the ancillary with `ElectronIntensityFactors.from_file`), Phase 3 (the intensity calculation reads ΔE, G, b and rate indices from it)
> **Doc references:**
> - [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): test quality rules, full-result assertions, numpy assertions
> - [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): fail explicitly on malformed files, type annotations
> - [`documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv`](documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv): the attached ancillary (git-ignored; origin in `overview.md` → Source documents)
> - [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf): Table 53 (ΔE, G, b per I-ALiRT rate) and Table 52 (electron energy bins)

## Goal

The `electron-intensity-factors` ancillary CSV can be loaded into an `ElectronIntensityFactors` object. For each of the 6 electron energy bins the object exposes:

- the 0-based `ialirtrates` index for the A side and for the B side
- the effective bin width ΔE (MeV)
- the geometry factor G (cm² sr)
- the background b
- the bin's lower and upper energy bounds (MeV), derived from the cumulative ΔE
- the bin label

A file with the wrong header, the wrong number of rows, non-positive ΔE or G, or a rate number outside 1–20 raises a `ValueError` that names the file and the problem, instead of producing silently wrong intensities.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT electron intensity factors ancillary provides per-bin conversion factors

  Scenario: The attached ancillary is parsed into per-bin factors
    Given the file "imap_hit_electron-intensity-factors_20260513_v001.csv" with a UTF-8 BOM, the header
      "IALIRT_Rate_A_side,IALIRT_Rate_B_side,delta_E_MeV,Gfavg_cm2_sr,b" and the 6 rows from doc Table 53
    When ElectronIntensityFactors.from_file is called with that path
    Then a_side_rate_indices is [0, 1, 2, 3, 4, 5]
    And b_side_rate_indices is [10, 11, 12, 13, 14, 15]
    And delta_e_mev is [0.3, 0.1, 0.4, 0.4, 0.8, 88]
    And geometry_factor_cm2_sr is [1.7792, 0.39942, 0.20431, 0.011787, 0.0013381, 0.000034886]
    And background is [0, 0, 0, 0, 0, 0]

  Scenario: Energy bin bounds and labels are derived from the cumulative bin widths
    Given the attached ancillary has been loaded
    When the energy bin bounds are read from ElectronIntensityFactors
    Then energy_lower_mev is [0, 0.3, 0.4, 0.8, 1.2, 2.0]
    And energy_upper_mev is [0.3, 0.4, 0.8, 1.2, 2.0, 90.0]
    And energy_labels is ["<300 keV", "300-400 keV", "400-800 keV", "800-1200 keV", "1200-2000 keV", ">2000 keV"]

  Scenario Outline: A malformed ancillary is rejected with a clear error
    Given an electron intensity factors file where <defect>
    When ElectronIntensityFactors.from_file is called with that path
    Then a ValueError is raised whose message contains the file path and "<message fragment>"

    Examples:
      | defect                                           | message fragment              |
      | there are 5 data rows                            | Expected 6 energy bins        |
      | the header is "rate_a,rate_b,dE,G,b"             | Unexpected header             |
      | delta_E_MeV is 0 in the third row                | delta_E_MeV must be positive  |
      | Gfavg_cm2_sr is -1 in the first row              | Gfavg_cm2_sr must be positive |
      | IALIRT_Rate_B_side is 21 in the last row         | I-ALiRT rate must be in 1-20  |
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/electron/__init__.py` *(new, empty)*
- `imap_l3_processing/hit/l3/electron/science/__init__.py` *(new, empty)*
- `imap_l3_processing/hit/l3/electron/science/electron_intensity_factors.py` *(new)*:
  - `ElectronIntensityFactors` dataclass, with `from_file(path: Path) -> ElectronIntensityFactors` and `energy_lower_mev`, `energy_upper_mev` and `energy_labels` properties.
  - Constants `ELECTRON_ENERGY_BIN_COUNT = 6`, `IALIRT_RATE_COUNT = 20` and `EXPECTED_HEADER`.

### Tests

- `tests/hit/l3/electron/__init__.py`, `tests/hit/l3/electron/science/__init__.py` *(new, empty)*
- `tests/hit/l3/electron/science/test_electron_intensity_factors.py` *(new)*
- `tests/test_data/hit/electron/imap_hit_electron-intensity-factors_20260513_v001.csv` *(new fixture)*: a byte-for-byte copy of the attached CSV, BOM included

## TDD steps

### 1.1 Red — Scenario: The attached ancillary is parsed into per-bin factors

- Copy `documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv` (untracked; re-download from the origin link in `overview.md` if missing) to `tests/test_data/hit/electron/imap_hit_electron-intensity-factors_20260513_v001.csv`, keeping the BOM.
- In `test_electron_intensity_factors.py`, write `test_from_file_parses_attached_ancillary_into_per_bin_factors`:
  - Load via `get_test_data_path("hit/electron/imap_hit_electron-intensity-factors_20260513_v001.csv")`.
  - Assert every field with `np.testing.assert_array_equal` for the integer index arrays and `np.testing.assert_allclose` (default tolerance) for ΔE, G and b. Assert all five fields so the whole object is checked.
- Fails: `ModuleNotFoundError`.

### 1.2 Green

- Create the packages and `ElectronIntensityFactors` with fields:
  - `a_side_rate_indices: np.ndarray`
  - `b_side_rate_indices: np.ndarray`
  - `delta_e_mev: np.ndarray`
  - `geometry_factor_cm2_sr: np.ndarray`
  - `background: np.ndarray`
- `from_file`:
  - Open with `encoding="utf-8-sig"`.
  - Read with `csv.DictReader` or `np.genfromtxt(..., names=True)`.
  - Subtract 1 from the rate columns to make 0-based indices.

### 1.3 Red — Scenario: Energy bin bounds and labels are derived from the cumulative bin widths

- Add `test_energy_bin_bounds_and_labels_are_derived_from_cumulative_bin_widths`, which loads the same fixture.
- Assert `energy_lower_mev` and `energy_upper_mev` with `assert_allclose` (cumsum of floats), and assert `energy_labels` with `assertEqual` on the full list.
- Fails: `AttributeError`.

### 1.4 Green

- `energy_upper_mev = np.cumsum(delta_e_mev)` and `energy_lower_mev = energy_upper_mev - delta_e_mev`.
- `energy_labels`:
  - The first bin is `f"<{upper_kev:.0f} keV"`.
  - The last bin is `f">{lower_kev:.0f} keV"`.
  - The others are `f"{lower_kev:.0f}-{upper_kev:.0f} keV"`.
  - `kev = mev * 1000`, rounded so floating cumsum noise doesn't leak into the labels.

### 1.5 Red — Scenario Outline: A malformed ancillary is rejected with a clear error

- Add `test_from_file_rejects_malformed_ancillary`. Use `subTest` per Examples row (one behavior, one test function). Each subtest writes a defective CSV into a `TempFileTestCase` temp directory, derived from the good fixture text with one change.
- Assert `with self.assertRaises(ValueError) as context`, then `assertIn(str(path), str(context.exception))` and `assertIn(<message fragment>, str(context.exception))`.
- Fails: no validation (wrong shapes pass through, or an unrelated `KeyError` or `IndexError` is raised).

### 1.6 Green

- After reading, validate in this order:
  - header equals `EXPECTED_HEADER`
  - row count equals `ELECTRON_ENERGY_BIN_COUNT`
  - all `delta_E_MeV > 0`
  - all `Gfavg_cm2_sr > 0`
  - all rate numbers are in `1..IALIRT_RATE_COUNT`
- Raise `ValueError(f"... in {path}: ...")` with the message fragments from the Examples table.

### 1.7 Refactor

- Extract `_validate(path: Path, ...) -> None` if `from_file` gets long.
- Keep `from_file` the only public constructor path used by later phases. Tests in Phases 3–5 can construct `ElectronIntensityFactors(...)` directly with simple numbers.

## Done criteria

- [ ] Scenario "attached ancillary is parsed" passes, asserting all five fields
- [ ] Scenario "bounds and labels derived" passes for lower, upper and labels
- [ ] All five malformed-file Examples raise `ValueError` with the path and message fragment
- [ ] Fixture CSV committed under `tests/test_data/hit/electron/` with its BOM intact
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
