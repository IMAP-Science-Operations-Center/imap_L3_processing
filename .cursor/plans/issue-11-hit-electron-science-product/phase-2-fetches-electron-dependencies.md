# Phase 2 — HIT electron dependencies provide L1B I-ALiRT rates and intensity factors

> **Depends on:** Phase 1 (`ElectronIntensityFactors.from_file` must exist)
> **Blocks:** Phase 5 (the processor's `electron` branch calls `HitL3ElectronDependencies.fetch_dependencies`)
> **Doc references:**
> - [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): isolate `imap_data_access` in unit tests, assert fill → NaN explicitly, assert `.data` and the NaN positions
> - [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): fail explicitly on a missing required dependency, `MODULE` patch constant
> - [`phase-1-electron-intensity-factors-ancillary.md`](phase-1-electron-intensity-factors-ancillary.md): the ancillary this phase loads
> - `imap_processing/hit/l1b/hit_l1b.py` (`process_standard_rates_data`, `calculate_rates`): the source of `ialirtrates`
> - [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf): Table 23 (IALIRTRATES layout, index → L4Ai/L4Bi bins), §8.2 Table 36, §9.3 (L3 uses livetime-corrected L1B data)

## Goal

When the SDC runs the HIT L3 `electron` job, the dependency collection holds:

- a HIT L1B `standard-rates` science file
- the `electron-intensity-factors` ancillary

`HitL3ElectronDependencies.fetch_dependencies` downloads both. It reads the L1B file into `HitL1bIalirtRates`: epochs, `ialirtrates` with its `FILLVAL` replaced by NaN, and the plus/minus uncertainties. The real L1B file (see `overview.md` → Verified on a real L1B file) gives the uncertainty variables **no `FILLVAL` attribute**, so they are read as stored and then set to NaN wherever the rate is fill. It parses the ancillary with `ElectronIntensityFactors.from_file`. If either input is missing, fetching fails at once with `ValueError("Missing <descriptor> dependency.")` rather than a bare `IndexError`.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT electron dependencies provide L1B I-ALiRT rates and intensity factors

  Scenario: An L1B standard-rates CDF is read into I-ALiRT rates with fill values as NaN
    Given an L1B standard-rates CDF with 3 epochs and ialirtrates, ialirtrates_stat_uncert_plus and
      ialirtrates_stat_uncert_minus of shape (3, 20)
    And ialirtrates has FILLVAL -9.223372e+18 and the uncertainty variables have no FILLVAL attribute,
      as in the SDC file imap_hit_l1b_standard-rates_20260601_v001.0003.cdf
    And ialirtrates holds its FILLVAL at epoch 1, index 2
    When HitL1bIalirtRates.read_from_cdf is called with that path
    Then epoch equals the CDF's 3 epochs
    And rates, rates_delta_plus and rates_delta_minus equal the CDF values everywhere except epoch 1, index 2
    And rates, rates_delta_plus and rates_delta_minus at epoch 1, index 2 are NaN and no other entry is NaN

  Scenario: L1B rates and the intensity factors ancillary are downloaded, read and exposed
    Given a ProcessingInputCollection with "imap_hit_l1b_standard-rates_20260601_v001.cdf"
      and "imap_hit_electron-intensity-factors_20260513_v001.csv"
    When HitL3ElectronDependencies.fetch_dependencies is called
    Then both files are downloaded
    And HitL1bIalirtRates.read_from_cdf is called with the downloaded L1B path
    And ElectronIntensityFactors.from_file is called with the downloaded ancillary path
    And the returned dependencies hold the read rates and the parsed factors

  Scenario Outline: Fetching fails clearly when a required input is missing
    Given a ProcessingInputCollection that lacks the <missing> input
    When HitL3ElectronDependencies.fetch_dependencies is called
    Then a ValueError "Missing <descriptor> dependency." is raised

    Examples:
      | missing                      | descriptor                 |
      | L1B standard-rates           | standard-rates             |
      | electron intensity factors   | electron-intensity-factors |
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/electron/models.py` *(new)*:
  - `HitL1bIalirtRates` dataclass with `epoch`, `rates`, `rates_delta_plus` and `rates_delta_minus`
  - `read_from_cdf(cdf_file_path: Path | str) -> HitL1bIalirtRates`, using `read_numeric_variable`
- `imap_l3_processing/hit/l3/electron/hit_l3_electron_dependencies.py` *(new)*:
  - `HIT_L1B_STANDARD_RATES_DESCRIPTOR = "standard-rates"`
  - `HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR = "electron-intensity-factors"`
  - `HitL3ElectronDependencies` dataclass with `ialirt_rates: HitL1bIalirtRates` and `intensity_factors: ElectronIntensityFactors`, plus `fetch_dependencies`

### Tests

- `tests/hit/l3/electron/test_models.py` *(new)*: the L1B reader scenario. The `HitElectronDataProduct` tests join this file in Phase 4.
- `tests/hit/l3/electron/test_hit_l3_electron_dependencies.py` *(new)*

## TDD steps

### 2.1 Red — Scenario: An L1B standard-rates CDF is read into I-ALiRT rates with fill values as NaN

- In `test_models.py` (a `TempFileTestCase`), write `test_read_from_cdf_reads_ialirt_rates_and_replaces_fill_with_nan`.
- Use `spacepy.pycdf.CDF(path, "")` to write a CDF in the temp directory, shaped like the real SDC file:
  - `epoch`: 3 datetimes, 60 s apart
  - `ialirtrates`: `float32`, shape `(3, 20)`, distinct values, attribute `FILLVAL = np.float32(-9.223372e+18)`
  - `ialirtrates_stat_uncert_plus` and `ialirtrates_stat_uncert_minus`: `float32`, shape `(3, 20)`, distinct values, **no `FILLVAL` attribute**
  - put the FILLVAL into `ialirtrates[1, 2]`, with ordinary finite numbers at `[1, 2]` in both uncertainty arrays
- Assert `epoch` with `assert_array_equal`. Assert the three arrays with `assert_array_equal` against expected arrays that have `np.nan` at `[1, 2]`; `assert_array_equal` treats NaN == NaN. Also assert `np.argwhere(np.isnan(...))` equals `[[1, 2]]` for each of the three arrays.
- Fails: `ModuleNotFoundError`. Once the module exists, a naive `read_numeric_variable` on the uncertainties raises `KeyError: 'FILLVAL'`, and the uncertainties at `[1, 2]` stay finite.

### 2.2 Green

- Create `HitL1bIalirtRates.read_from_cdf`:
  - `epoch = cdf["epoch"][...]`
  - `rates = read_numeric_variable(cdf["ialirtrates"])`
  - read the uncertainties raw (`cdf[name][...].astype(float)`), then `np.where(np.isnan(rates), np.nan, ...)` so a fill rate's uncertainties are NaN too
  - don't use `read_numeric_variable` on the uncertainty variables; they have no `FILLVAL`

### 2.3 Red — Scenario: L1B rates and the intensity factors ancillary are downloaded, read and exposed

- In `test_hit_l3_electron_dependencies.py`, set `MODULE = "imap_l3_processing.hit.l3.electron.hit_l3_electron_dependencies"` and write `test_fetch_dependencies_downloads_reads_and_exposes_inputs`.
- Patch `f"{MODULE}.imap_data_access.download"` with `side_effect=[sentinel.l1b_path, sentinel.factors_path]`, plus `f"{MODULE}.HitL1bIalirtRates.read_from_cdf"` and `f"{MODULE}.ElectronIntensityFactors.from_file"`.
- Build the collection from the two file names with `ScienceInput` / `AncillaryInput`, following `tests/hit/l3/pha/test_l3_pha_dependencies.py`.
- Assert:
  - `mock_download.assert_has_calls([call(<l1b name>), call(<factors name>)])`
  - `read_from_cdf.assert_called_once_with(sentinel.l1b_path)`
  - `from_file.assert_called_once_with(sentinel.factors_path)`
  - `assertEqual(HitL3ElectronDependencies(ialirt_rates=read_from_cdf.return_value, intensity_factors=from_file.return_value), result)`
- Fails: `ModuleNotFoundError`.

### 2.4 Green

- Implement `fetch_dependencies`. Get the L1B file with `get_file_paths(source="hit", data_type="l1b", descriptor=HIT_L1B_STANDARD_RATES_DESCRIPTOR)` and the ancillary with `get_file_paths(source="hit", descriptor=HIT_ELECTRON_INTENSITY_FACTORS_DESCRIPTOR)`. Download the first entry of each, read, and return.

### 2.5 Red — Scenario Outline: Fetching fails clearly when a required input is missing

- Add `test_fetch_dependencies_raises_when_required_input_missing` with a `subTest` per Examples row. Build the collection with only the other input, patch `download` and the readers, and assert `ValueError` with the exact message `f"Missing {descriptor} dependency."`.
- Fails: `IndexError` from `[0]`.

### 2.6 Green

- Add a module-level helper `_download_required(dependencies: ProcessingInputCollection, descriptor: str, **query: str) -> Path`. It raises `ValueError(f"Missing {descriptor} dependency.")` when `get_file_paths` returns an empty list, and both inputs use it.

### 2.7 Refactor

- Confirm the L1B query passes `data_type="l1b"`, so a future L2 `standard-rates` file in the same collection can't be picked up. If needed, add that case to the 2.3 test's collection (an extra L2 file that must not be downloaded) instead of writing a separate test.

## Done criteria

- [ ] Scenario "L1B CDF read with fill as NaN" passes and asserts the NaN position exactly
- [ ] Scenario "downloaded, read and exposed" passes, asserting the whole returned dataclass
- [ ] Both missing-input Examples raise the exact `ValueError` message
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
