# Phase 6 — HIT L3 electron CDF is produced end to end from a real L1B file

> **Depends on:** Phase 5 (the `electron` branch in `HitProcessor`)
> **Blocks:** none
> **Doc references:**
> - [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): end-to-end runs against real files belong in `tests/integration/`
> - [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): verify, don't assume (inspect the real L1B file before relying on it)
> - [`phase-2-fetches-electron-dependencies.md`](phase-2-fetches-electron-dependencies.md): the L1B variable names and FILLVAL this phase confirms
> - [`phase-4-electron-data-product-and-cdf-attributes.md`](phase-4-electron-data-product-and-cdf-attributes.md): the `epoch_delta` assumption this phase confirms
> - `tests/integration/test_hit_processor_integration.py` (`_run_hit_l3_product`)

## Goal

Running `imap_l3_data_processor` with `--instrument hit --data-level l3 --descriptor electron` writes `imap_hit_l3_electron_<date>_v001.cdf`. The run uses the real SDC file `imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` and the `electron-intensity-factors` ancillary fixture. The output's intensities are finite where the L1B rates are finite, and its variables match Phase 4's list.

The L1B file was already inspected during planning (`overview.md` → Verified on a real L1B file):

- `ialirtrates` is `(1440, 20)` counts per frame with `FILLVAL -9.223372e+18`.
- The uncertainties have no `FILLVAL`.
- Epochs are 60 s apart with no `epoch_delta`.
- The file has no fill values and DT state is 0 all day.

So this phase is pure end-to-end wiring; nothing about the data is left to discover.

## Prerequisite (do first, before any Red step)

- Copy the file into `tests/test_data/hit/electron/`. At 961 KB it is committed directly, like the other HIT CDF fixtures; `.gitattributes` does not LFS-track `.cdf`.
- Where to get it: if the planning copy is gone, re-download it from the SDC with `imap_data_access.download("imap/hit/l1b/2026/06/imap_hit_l1b_standard-rates_20260601_v001.0003.cdf")`. That is a file download, so the user runs it or approves it first.
- If the SDC has reprocessed the file (a newer `v001.000N`), use the version the user agrees to and re-check the facts above with the `read-cdf` skill before writing the test.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT L3 electron CDF is produced end to end from a real L1B file

  Scenario: The electron product is written from real L1B standard-rates and the ancillary fixture
    Given the SDC file "imap_hit_l1b_standard-rates_20260601_v001.0003.cdf" with 1440 epochs
    And the fixture "imap_hit_electron-intensity-factors_20260513_v001.csv"
    When imap_l3_data_processor runs with instrument "hit", level "l3", descriptor "electron" and date 20260601
    Then "imap_hit_l3_electron_20260601_v001.cdf" exists
    And it contains exactly the 13 variables emitted by HitElectronDataProduct
    And electron_intensity_sunward and electron_intensity_antisunward have shape (1440, 6)
    And every intensity whose L1B rate is not fill is finite
```

## Affected files

### Production

- none expected. The data facts were verified during planning and are already built into Phases 2 and 4.

### Tests

- `tests/integration/test_hit_processor_integration.py`: add `test_hit_electron_product`
- `tests/test_data/hit/electron/imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` *(new fixture, 961 KB)*
- `tests/test_data/hit/electron/imap_hit_electron-intensity-factors_20260513_v001.csv` (from Phase 1)

## TDD steps

### 6.1 Red — Scenario: The electron product is written from real L1B standard-rates and the ancillary fixture

- Add `test_hit_electron_product` using the existing `_run_hit_l3_product(mock_parse_cli_arguments, input_files, "hit_electron_integration", "electron", "20260601")`.
- After the run, open the output with `spacepy.pycdf.CDF` (read-only) and assert:
  - the set of zVariable names equals the 13 Phase 4 names
  - both intensity shapes are `(1440, 6)`
  - `np.isfinite` is true everywhere in both intensity arrays, since the fixture has no fill (asserted with `assert_array_equal` against an all-True mask)
- This red run is expected to pass if Phases 1–5 are right. Its value is the end-to-end wiring. If it fails, find the root cause before changing anything.

### 6.2 Green

- Only if 6.1 fails: fix the root cause in the owning phase's code, guided by a unit test in that phase first. Do not loosen the integration assertions.

### 6.3 Refactor

- Optionally run `run_istp_compliance_check` on the output once by hand. It posts the file to an external service, so ask the user first. Do not add it to the automated test, because the other HIT integration tests don't call it.

## Done criteria

- [ ] `imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` copied into `tests/test_data/hit/electron/`. If a newer SDC version is used instead, it has been re-checked against `overview.md` → Verified on a real L1B file.
- [ ] `test_hit_electron_product` passes: file exists, exact variable set, shapes, finite where input is not fill
- [ ] `uv run python -m unittest tests.integration.test_hit_processor_integration` passes
- [ ] Full suite `uv run python -m unittest discover tests` passes (Docker-only GLOWS tests skipped if Docker isn't running)
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
