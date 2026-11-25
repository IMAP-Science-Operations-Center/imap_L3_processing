# Phase 5 — HIT L3 processor produces the electron product

> **Depends on:** Phase 2 (`HitL3ElectronDependencies.fetch_dependencies`), Phase 3 (`calculate_electron_intensities`), Phase 4 (`HitElectronDataProduct` and its YAML)
> **Blocks:** Phase 6 (the end-to-end run goes through this branch)
> **Doc references:**
> - [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): processor tests assert the `save_data` call and the full returned paths; full flag arrays
> - [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): `MODULE` patch constant, type annotations
> - [`phase-2-fetches-electron-dependencies.md`](phase-2-fetches-electron-dependencies.md), [`phase-3-electron-intensity-calculation.md`](phase-3-electron-intensity-calculation.md), [`phase-4-electron-data-product-and-cdf-attributes.md`](phase-4-electron-data-product-and-cdf-attributes.md)
> - [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf): §9.3 ("no L3 HIT electron data will be generated prior to 2026-05-13")
> - Existing tests: `tests/hit/l3/test_hit_processor.py` (`test_process_direct_event_product`, `test_raise_error_if_descriptor_doesnt_match`)

## Goal

When `HitProcessor.process()` runs with descriptor `electron` and a start date on or after 2026-05-13, it:

1. fetches the electron dependencies
2. computes intensities with Eq. 32
3. builds a `HitElectronDataProduct` with:
   - the L1B epochs
   - the energy bins from the ancillary
   - `hit_flags` all `NONE`
   - parent file names
4. saves it with `save_data` and returns `[cdf_path]`

For a start date before 2026-05-13 it logs that no electron data is produced before the L4 energy-slice change, fetches nothing, and returns `[]` (user decision). An unknown descriptor still raises, and the message now lists `electron` among the known products.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT L3 processor produces the electron product

  Background:
    Given InputMetadata for instrument "hit", level "l3", descriptor "electron"

  Scenario: An electron product is built from the dependencies and saved
    Given the start date is 2026-06-01
    And HitL3ElectronDependencies.fetch_dependencies returns 2 epochs of L1B I-ALiRT rates and intensity factors
    And calculate_electron_intensities returns known A-side and B-side arrays
    When HitProcessor.process is called
    Then calculate_electron_intensities is called with the L1B rates, delta_plus, delta_minus and the factors
    And save_data is called once with a HitElectronDataProduct whose epoch is the L1B epoch,
      energy, energy deltas and labels come from the factors, intensities come from the calculation,
      hit_flags is [NONE, NONE] and parent_file_names are the processor's parent file names
    And process returns [the path save_data returned]

  Scenario Outline: Start dates before 2026-05-13 produce no output files
    Given the start date is <start date>
    When HitProcessor.process is called
    Then process returns <result>
    And HitL3ElectronDependencies.fetch_dependencies <fetched>

    Examples:
      | start date | result                   | fetched           |
      | 2026-05-12 | []                       | is not called     |
      | 2026-05-13 | [the saved CDF path]     | is called once    |

  Scenario: An unknown descriptor names electron among the known products
    Given the descriptor is "not-a-product"
    When HitProcessor.process is called
    Then a ValueError is raised listing 'macropixel', 'direct-events' and 'electron'
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/hit_processor.py`:
  - add an `elif self.input_metadata.descriptor == "electron":` branch
  - add `ELECTRON_PRODUCT_START_DATE = datetime(2026, 5, 13)`
  - add a `process_electron_product(self, dependencies: HitL3ElectronDependencies) -> HitElectronDataProduct` method
  - update the unknown-descriptor message

### Tests

- `tests/hit/l3/test_hit_processor.py`:
  - add a `MODULE = "imap_l3_processing.hit.l3.hit_processor"` constant if it is missing, and use it for the new tests' patches
  - add the electron tests
  - update `test_raise_error_if_descriptor_doesnt_match`'s expected message
- `tests/hit/l3/hit_test_builders.py`: an electron data product / factors builder, if Phase 4's refactor moved one here

## TDD steps

### 5.1 Red — Scenario: An electron product is built from the dependencies and saved

- Write `test_process_electron_product_builds_and_saves_product`:
  - Patch `f"{MODULE}.HitL3ElectronDependencies.fetch_dependencies"`, `f"{MODULE}.calculate_electron_intensities"` and `f"{MODULE}.save_data"`.
  - Give the fetch mock a real `HitL3ElectronDependencies` holding a real `HitL1bIalirtRates` (2 epochs, `(2, 20)` arrays) and a real `ElectronIntensityFactors`, so the energy-bin properties are real.
  - Make `calculate_electron_intensities` return a real `ElectronIntensities` with distinct arrays.
  - Patch `HitProcessor.get_parent_file_names` (or build `ProcessingInputCollection` inputs) to give known names.
- Assert:
  - `calculate_electron_intensities.assert_called_once_with(...)`, using `NumpyArrayMatcher` for the three rate arrays and the factors object.
  - `save_data.assert_called_once()`, then check every field of the saved `HitElectronDataProduct` with numpy assertions: `epoch`, `energy` (midpoints), `energy_delta_plus/minus` (ΔE/2), `energy_labels`, all six intensity arrays, `hit_flags == np.full(2, HitL3Flags.NONE)`, `input_metadata` and `parent_file_names`.
  - `assertEqual([save_data.return_value], result)`.
- Fails: `ValueError("Don't know how to generate 'electron'...")`.

### 5.2 Green

- Add the `electron` branch:
  - `dependencies = HitL3ElectronDependencies.fetch_dependencies(self.dependencies)`
  - `product = self.process_electron_product(dependencies)`
  - set `product.parent_file_names = self.get_parent_file_names()`
  - `return [save_data(product)]`
- In `process_electron_product`, call `calculate_electron_intensities(...)`. Build the product with:
  - `energy = (lower + upper) / 2`
  - `energy_delta_plus = energy_delta_minus = factors.delta_e_mev / 2`
  - `hit_flags = np.full(len(epoch), HitL3Flags.NONE, dtype=np.uint16)`

### 5.3 Red — Scenario Outline: Start dates before 2026-05-13 produce no output files

- Write `test_electron_product_is_only_produced_from_2026_05_13` with a `subTest` per Examples row, patching fetch, calculate and `save_data` as in 5.1.
- For 2026-05-12, assert `result == []`, `fetch_dependencies.assert_not_called()` and `save_data.assert_not_called()`.
- For 2026-05-13, assert `result == [save_data.return_value]` and `fetch_dependencies.assert_called_once()`.
- Also assert, with `assertLogs`, that the 05-12 run logs an INFO message naming the 2026-05-13 cutoff.
- Fails: the 05-12 case currently fetches and saves.

### 5.4 Green

- At the top of the `electron` branch, check `if self.input_metadata.start_date < ELECTRON_PRODUCT_START_DATE:`. Inside, call `logger.info(f"No HIT L3 electron data is produced before {ELECTRON_PRODUCT_START_DATE:%Y-%m-%d} (L4 energy slices changed); start date {self.input_metadata.start_date:%Y-%m-%d}")` and `return []`.

### 5.5 Red — Scenario: An unknown descriptor names electron among the known products

- Update `test_raise_error_if_descriptor_doesnt_match` to expect `'electron'` in the known-products list. It exists today and asserts the message, so this is a change to living behavior, not a new test.
- Fails: the message lacks `'electron'`.

### 5.6 Green

- Update the message to `Known HIT l3 data products: 'macropixel', 'direct-events', 'electron'.`. While touching it, fix the literal `/n` typo to `\n`, and make sure the updated test asserts the corrected text.

### 5.7 Refactor

- If the descriptor chain grows unwieldy, extract each branch's body into a `_process_<product>() -> list[Path]` method so `process()` stays a thin dispatcher. Do this only if the existing branches move together; otherwise leave them as they are.

## Done criteria

- [ ] Scenario "built and saved" passes, asserting every field of the saved product, the calculation call and the returned paths
- [ ] Both start-date Examples pass, including "fetch not called" and the INFO log for 2026-05-12
- [ ] Unknown-descriptor test expects `'electron'` and the corrected newline
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
