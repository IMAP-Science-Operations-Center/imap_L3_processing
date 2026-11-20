# Phase 4 — HIT electron data product exposes ISTP variables

> **Depends on:** none
> **Blocks:** Phase 5 (the processor builds and saves `HitElectronDataProduct`)
> **Doc references:**
> - [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): assert the full `to_data_product_variables()` list; YAML must match every variable
> - [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): type annotations on dataclass fields
> - [`.cursor/rules/no-comments.mdc`](../../rules/no-comments.mdc): no comments in YAML either; use `VAR_NOTES` for explanations
> - [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf): §9.3 (A/B side propagation direction, 2026-05-13 cutoff, wording for `VAR_NOTES`), Table 52 (energy bin labels)
> - Existing patterns: `imap_l3_processing/hit/l3/models.py` (`HitDirectEventDataProduct`), `imap_l3_processing/cdf/config/imap_hit_l3_macropixel_variable_attrs.yaml`

## Goal

`HitElectronDataProduct` holds the per-epoch electron intensities for both sides and the energy-bin description. `to_data_product_variables()` returns every CDF variable in a fixed order. `ImapAttributeManager` can resolve global attributes for `imap_hit_l3_electron` and variable attributes for every variable the product emits, so `save_data` writes an ISTP-shaped file.

| Variable | Shape | Purpose |
|---|---|---|
| `epoch` | (n) | time |
| `epoch_delta` | (n) | 30 s in ns: half the 60 s L1B frame, the same convention as HIT L2 macropixel's 300 s (see `overview.md` → Verified on a real L1B file) |
| `electron_energy` | (6) | bin midpoint, MeV |
| `electron_energy_delta_plus`, `electron_energy_delta_minus` | (6) | ΔE/2, MeV |
| `electron_energy_label` | (6) | e.g. `300-400 keV` |
| `electron_intensity_sunward` | (n, 6) | A side, propagating toward the Sun |
| `electron_intensity_sunward_delta_plus`, `_delta_minus` | (n, 6) | ±σ |
| `electron_intensity_antisunward` | (n, 6) | B side, propagating away from the Sun |
| `electron_intensity_antisunward_delta_plus`, `_delta_minus` | (n, 6) | ±σ |
| `hit_flags` | (n) | quality flags |

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT electron data product exposes ISTP variables

  Scenario: The data product emits every electron CDF variable in order
    Given a HitElectronDataProduct with 2 epochs, 6 energy bins, sunward and antisunward intensities
      with plus and minus uncertainties, and hit_flags
    When to_data_product_variables is called
    Then the result is exactly the list of DataProductVariables
      epoch, epoch_delta, electron_energy, electron_energy_delta_plus, electron_energy_delta_minus,
      electron_energy_label, electron_intensity_sunward, electron_intensity_sunward_delta_plus,
      electron_intensity_sunward_delta_minus, electron_intensity_antisunward,
      electron_intensity_antisunward_delta_plus, electron_intensity_antisunward_delta_minus, hit_flags
    And each variable's value is the matching field of the data product
    And epoch_delta is 30 seconds in nanoseconds for every epoch

  Scenario: Every emitted variable has CDF attributes and the product has global attributes
    Given the HIT L3 electron attribute YAML files
    When ImapAttributeManager.add_instrument_attrs("hit", "l3", "electron") is called
    Then get_variable_attributes succeeds for every variable name the product emits
    And the global attributes include Logical_source_description for "imap_hit_l3_electron"
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/electron/models.py`: add the variable-name constants and the `HitElectronDataProduct(DataProduct)` dataclass. The fields are:
  - `epoch`
  - `energy`, `energy_delta_plus`, `energy_delta_minus`, `energy_labels`
  - `sunward`, `sunward_delta_plus`, `sunward_delta_minus`
  - `antisunward`, `antisunward_delta_plus`, `antisunward_delta_minus`
  - `hit_flags`
- `imap_l3_processing/cdf/config/imap_hit_l3_electron_global_cdf_attrs.yaml` *(new)*:
  - `Data_type: L3_Electron>Level-3 Science data`
  - `Logical_source_description: IMAP Mission HIT Instrument Level-3 Electron Intensity Science Data`
  - other fields as in the macropixel file
- `imap_l3_processing/cdf/config/imap_hit_l3_electron_variable_attrs.yaml` *(new)*: one entry per variable, modeled on the macropixel YAML.
  - The intensities are `CDF_REAL4` with `DEPEND_0: epoch`, `DEPEND_1: electron_energy`, `LABL_PTR_1: electron_energy_label` and `DISPLAY_TYPE: spectrogram`.
  - Units are `"#/(cm^2-s-sr-MeV)"` and the FILLVAL is `-1.0000E+31`.
  - Each intensity carries `DELTA_PLUS_VAR` / `DELTA_MINUS_VAR`.
  - `electron_energy` is NRV with `DELTA_PLUS_VAR` / `DELTA_MINUS_VAR`. `hit_flags` copies the macropixel entry with electron wording.
  - `VAR_NOTES` on the intensities states:
    - the side and propagation direction
    - Eq. 32 and the 60 s frame
    - that no data is produced before 2026-05-13

### Tests

- `tests/hit/l3/electron/test_models.py`: add the data product tests next to the Phase 2 reader test.

## TDD steps

### 4.1 Red — Scenario: The data product emits every electron CDF variable in order

- Write `test_to_data_product_variables_emits_every_electron_variable_in_order`. Build the product with distinct arrays per field (for example `np.full((2, 6), k)` for a different `k` each) and `input_metadata=Mock()`.
- Assert the result against the full expected list. Loop with `zip` over expected `(name, value)` pairs, checking `assertEqual` on `.name` and `np.testing.assert_array_equal` on `.value`, and first assert `len(result) == 13` so missing or extra variables fail. `DataProductVariable` equality does not compare numpy arrays.
- The `epoch_delta` expected value is `np.full(2, 30_000_000_000)`.
- Fails: `ImportError`.

### 4.2 Green

- Add the constants (`ELECTRON_ENERGY_VAR_NAME = "electron_energy"`, etc.), reuse `EPOCH_VAR_NAME`, `EPOCH_DELTA_VAR_NAME` and `HIT_FLAGS_CDF_VAR_NAME`, and add the dataclass with `to_data_product_variables`.
- `epoch_delta` is `np.full(len(self.epoch), ELECTRON_EPOCH_DELTA_NS)` with `ELECTRON_EPOCH_DELTA_NS = 30 * 1_000_000_000`.

### 4.3 Red — Scenario: Every emitted variable has CDF attributes and the product has global attributes

- Write `test_attribute_manager_resolves_every_emitted_variable_and_global_attrs`:
  - Create `ImapAttributeManager()` and call `add_instrument_attrs("hit", "l3", "electron")`.
  - For each `.name` in the 4.1 product's `to_data_product_variables()`, call `get_variable_attributes(name)` in a `subTest`, which raises on a missing key or schema violation.
  - Assert the global `Logical_source_description` equals the YAML string.
- Fails: `FileNotFoundError` or `KeyError`, since the YAML doesn't exist.

### 4.4 Green

- Write both YAML files.
- Copy `epoch`, `epoch_delta`, `hit_flags` and the `*_energy` / `*_energy_label` entry shapes from `imap_hit_l3_macropixel_variable_attrs.yaml`. Then write the six intensity and uncertainty entries.

### 4.5 Refactor

- If the 4.3 test can share the product built in 4.1, move the builder to a `_build_electron_data_product()` helper in `test_models.py`, or into `tests/hit/l3/hit_test_builders.py` if Phase 5 also needs it. Phase 5 does, for its `save_data` assertion.

## Done criteria

- [ ] Scenario "emits every variable in order" passes, asserting all 13 names and values
- [ ] Scenario "attributes resolve" passes for every emitted variable plus the global attrs
- [ ] YAML has no comments; explanations live in `VAR_NOTES`
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
