# Issue 11 — HIT: Electron science product

## Card reference

- **Card:** issue-11
- **GitHub issue:** [IMAP-Science-Operations-Center/imap_L3_processing#11](https://github.com/IMAP-Science-Operations-Center/imap_L3_processing/issues/11), "HIT: Electron science product (Phase D Deferred)"
- **Source documents:** these live in [`documents/`](documents/), which is git-ignored. Download them again from the links below if the folder is empty.
  - [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf). This is the latest version attached to the issue (comment of 2026-11-11). Origin: <https://github.com/user-attachments/files/33721925/HIT_Algorithm_Document-v1p11p05.pdf>. The plan relies on:
    - §9.3 "Lev3 Electron Science Products", Tables 52–53
    - Eq. 14 / Eq. 32 (intensity)
    - §8.2 (I-ALiRT rate definitions, Table 36)
    - Table 23 (science-frame IALIRTRATES layout)
    - §2.5 (I-ALiRT apertures are unaffected by dynamic threshold)
  - [`documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv`](documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv), the ancillary factors attached to the issue. Origin: <https://github.com/user-attachments/files/32026832/ialirt_electron_intensity_factors_v1_2026_09_09.csv>. Its values match doc Table 53 to within rounding.
  - Superseded versions also attached to the issue (not downloaded): `HIT_Algorithm_Document-v1.11.02.pdf` and `HIT_Algorithm_Document-v1.11.04.pdf`.
  - The real L1B file `imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` came from the SDC (`imap/hit/l1b/2026/06/`). It is data, not a source document, so it isn't kept in `documents/`; Phase 6 commits it as a test fixture.
- **Scope note:** the HIT team split the real-time I-ALiRT intensities into a separate ticket. This card covers only the L3 *science* electron product built from the continuous SSR science packets.

## Goal

The HIT L3 job with descriptor `electron` produces `imap_hit_l3_electron_<date>_v<NNN>.cdf`. It contains science-quality electron intensities in **6 energy bins** for both sides:

- **A side:** electrons propagating toward the Sun.
- **B side:** electrons propagating away from the Sun.

The intensities are computed from the livetime-corrected I-ALiRT aperture rates in HIT L1B `standard-rates`, using doc Eq. 32 reduced to one rate per bin:

```
J_i = (q_i − b_i) / (60 · Δt · ΔE_i · G_i)        Δt = 1 minute (one L1B epoch)
```

- `q_i` is the L1B `ialirtrates` value for bin `i`.
- `ΔE_i`, `G_i` and `b_i` come from the new `electron-intensity-factors` ancillary.

The L1B ±1σ statistical uncertainties scale by the same factor. Days before **2026-05-13**, when the L4 energy slices were redefined, produce **no output files**.

## What the doc and L1B code say, and what it means for code

| Source | Finding | Code impact |
|---|---|---|
| Doc §9.3 | L3 uses the livetime-corrected I-ALiRT data generated at L1B. DELTA PLUS/MINUS and livetime were removed from L3 in v1.11.04 because L1B already does them. | L3 reads L1B; no livetime step at L3 |
| `imap_processing` 1.0.35 `hit_l1b.process_standard_rates_data` | `ialirtrates` (20 per epoch) and `ialirtrates_stat_uncert_plus/minus` are divided by the **fractional** livetime (`livetime_counter / 270`). The values are livetime-corrected **counts per 1-minute frame**, not counts per second. | Eq. 32's `60 · Δt` (60 s) divisor **is** applied at L3 |
| Doc Table 23 / Table 36 | `ialirtrates` index 0–5 = L4Ai ΔE bins 0–5 (A side), 6–9 = L4Ai vs L3A, 10–15 = L4Bi ΔE bins 0–5 (B side), 16–19 = L4Bi vs L3B | The ancillary's 1-based rate numbers (1–6, 11–16) map to 0-based indices (0–5, 10–15) |
| Doc §2.5 | Dynamic-threshold states affect only the science apertures; the I-ALiRT apertures are unaffected | No DT geometry-factor scaling and no 1-minute DT offset for electrons |
| Doc §9.3 / Table 52 | No L3 electron data before 2026-05-13 | Processor returns `[]` for earlier start dates (user decision) |
| Real L1B file `imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` (inspected 2026-11-20) | See **Verified on a real L1B file** below | Phase 2 reader reads the uncertainties without fill masking; Phase 6 uses this file as its fixture |
| Doc Table 53 / CSV | ΔE = 0.3, 0.1, 0.4, 0.4, 0.8, 88 MeV. Their cumulative sum gives exactly the bin edges 0, 0.3, 0.4, 0.8, 1.2, 2.0, 90 MeV. b = 0 everywhere. | Energy bin bounds are derived from the ancillary ΔE; b is still subtracted so a future non-zero background is a data change only |

## Verified on a real L1B file

On 2026-11-20 I downloaded `imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` (961 KB, released, `ground_software_version` 1.0.35.post7.dev0) from the SDC and inspected it read-only. It is not yet in the repo; Phase 6 adds it as the integration fixture.

**SDC coverage:** the SDC has 81 L1B `standard-rates` files from 2026-05-13 to 2026-07-31. None are later than 2026-07-31.

**`ialirtrates`:**
- `float32`, shape `(1440, 20)`, `DEPEND_1: ialirtrates_index` (0–19).
- `UNITS: counts`, `FILLVAL: -9.223372e+18`, `DELTA_PLUS_VAR` / `DELTA_MINUS_VAR` set.
- **Values are livetime-corrected counts per 1-minute frame**, as the source suggested: whole counts slightly above an integer, e.g. 45.0114 and 2.0008. That confirms the `60 · Δt` divisor.

**Uncertainties (`ialirtrates_stat_uncert_plus` / `_minus`):**
- `float32`, shape `(1440, 20)`, `UNITS: counts`.
- **They have no `FILLVAL`, `CATDESC`, `VAR_TYPE` or `FIELDNAM`.** `read_numeric_variable` indexes `attrs["FILLVAL"]`, so it would raise `KeyError` on them.
- They are asymmetric, Poisson-style: at 0 counts, plus is 2.0008 and minus is 0.

**`epoch`:**
- TT2000, 1440 records, spacing exactly 60 s. The first record is 00:00:40.09 and the last 23:59:40.02.
- There is no `epoch_delta` variable and no `DELTA_*` attributes, so the file itself doesn't say whether `epoch` is the frame start or its centre.
- The `imap_processing` code answers it: `decom_hit.py` sets the epoch to the mean packet time of the frame, so it is the frame centre. The HIT L2 macropixel fixture (10-minute cadence) likewise uses `epoch_delta` = 300 s, half its cadence. The plan's 30 s `epoch_delta` follows both.

**Other:**
- `ialirtrates_index_label` values are just `"0"` … `"19"`.
- `dynamic_threshold_state` is 0 for the whole day.
- There are no fill values and no NaNs in `ialirtrates` that day.

**Daily median counts per frame on 2026-06-01:**

| Bin | A side (index 0–5) | B side (index 10–15) |
|---|---|---|
| <300 keV | 0 (max 1) | 0 (max 0, all day) |
| 300–400 keV | 45 | 57 |
| 400–800 keV | 29 | 33 |
| 800–1200 keV | 22 | 22 |
| 1200–2000 keV | 11 | 10 |
| >2000 keV | 6 | 4 |

The <300 keV bin is empty or close to it, which fits §8.2 leaving it out of the I-ALiRT products.

## Affected areas

### Production

- `imap_l3_processing/hit/l3/electron/__init__.py` *(new)*
- `imap_l3_processing/hit/l3/electron/science/__init__.py` *(new)*
- `imap_l3_processing/hit/l3/electron/science/electron_intensity_factors.py` *(new)*: `ElectronIntensityFactors` ancillary reader
- `imap_l3_processing/hit/l3/electron/science/electron_intensity.py` *(new)*: `calculate_electron_intensities`, `ElectronIntensities`
- `imap_l3_processing/hit/l3/electron/models.py` *(new)*: `HitL1bIalirtRates` (L1B reader) and `HitElectronDataProduct`
- `imap_l3_processing/hit/l3/electron/hit_l3_electron_dependencies.py` *(new)*: `HitL3ElectronDependencies`
- `imap_l3_processing/hit/l3/hit_processor.py`: `electron` descriptor branch and the pre-2026-05-13 no-output rule
- `imap_l3_processing/cdf/config/imap_hit_l3_electron_global_cdf_attrs.yaml` *(new)*
- `imap_l3_processing/cdf/config/imap_hit_l3_electron_variable_attrs.yaml` *(new)*

### Tests

- `tests/hit/l3/electron/__init__.py`, `tests/hit/l3/electron/science/__init__.py` *(new)*
- `tests/hit/l3/electron/science/test_electron_intensity_factors.py` *(new)*
- `tests/hit/l3/electron/science/test_electron_intensity.py` *(new)*
- `tests/hit/l3/electron/test_models.py` *(new)*
- `tests/hit/l3/electron/test_hit_l3_electron_dependencies.py` *(new)*
- `tests/hit/l3/test_hit_processor.py`
- `tests/integration/test_hit_processor_integration.py`
- New fixtures in `tests/test_data/hit/electron/`:
  - `imap_hit_electron-intensity-factors_20260513_v001.csv`: the attached CSV, renamed to the SDC ancillary convention
  - a small synthetic L1B `standard-rates` CDF, written by the test, for the L1B reader unit test
  - the real `imap_hit_l1b_standard-rates_20260601_v001.0003.cdf` (961 KB, committed directly like the other HIT CDF fixtures) for the integration test

## Phase summary

| # | Phase file | Gherkin Feature | Description |
|---|---|---|---|
| 1 | [`phase-1-electron-intensity-factors-ancillary.md`](phase-1-electron-intensity-factors-ancillary.md) | HIT electron intensity factors ancillary provides per-bin conversion factors | New reader for the `electron-intensity-factors` CSV: rate indices per side, ΔE, G, b, derived energy-bin bounds. Fails loudly on malformed files. |
| 2 | [`phase-2-fetches-electron-dependencies.md`](phase-2-fetches-electron-dependencies.md) | HIT electron dependencies provide L1B I-ALiRT rates and intensity factors | L1B `standard-rates` reader (`ialirtrates` + uncertainties, fill → NaN) and `HitL3ElectronDependencies.fetch_dependencies`, with clear errors when an input is missing |
| 3 | [`phase-3-electron-intensity-calculation.md`](phase-3-electron-intensity-calculation.md) | Electron intensities follow Algorithm Document Eq. 32 | `J = (q − b)/(60·ΔE·G)` per side and bin, uncertainties scaled the same way, NaN propagated |
| 4 | [`phase-4-electron-data-product-and-cdf-attributes.md`](phase-4-electron-data-product-and-cdf-attributes.md) | HIT electron data product exposes ISTP variables | `HitElectronDataProduct.to_data_product_variables()` plus global and variable attribute YAML for every variable |
| 5 | [`phase-5-processor-produces-electron-product.md`](phase-5-processor-produces-electron-product.md) | HIT L3 processor produces the electron product | `electron` descriptor branch: fetch, calculate, build product, `save_data`. Returns `[]` for start dates before 2026-05-13. |
| 6 | [`phase-6-electron-product-end-to-end.md`](phase-6-electron-product-end-to-end.md) | HIT L3 electron CDF is produced end to end from a real L1B file | Real L1B fixture plus ancillary fixture, integration test through `imap_l3_data_processor` |

## Dependency graph

```
Phase 1 (intensity factors ancillary)
   ├─▶ Phase 2 (fetch L1B + ancillary dependencies)
   │      └─▶ Phase 5 (processor electron branch)
   └─▶ Phase 3 (Eq. 32 intensity calculation)
          └─▶ Phase 5
Phase 4 (data product + CDF attrs) ───────▶ Phase 5
                                             └─▶ Phase 6 (end-to-end integration)
```

| Phase | Depends on | Blocks |
|---|---|---|
| 1 | none | 2, 3 |
| 2 | 1 | 5 |
| 3 | 1 | 5 |
| 4 | none | 5 |
| 5 | 2, 3, 4 | 6 |
| 6 | 5 | none |

Phases 2 and 3 can run in either order once Phase 1 is done. Phase 4 is independent of Phases 1–3.

## Decisions recorded (from the user, 2026-11-20)

1. **Energy bins:** all 6 bins per side, including <300 keV (rates 1/11), following §9.3. The I-ALiRT product list's 5-bin choice is a separate product.
2. **Before 2026-05-13:** `process()` returns no output files (`[]`) and logs why. It does not raise.
3. **Output shape:** descriptor `electron`, one 2D `(epoch, energy)` intensity variable per side with plus/minus uncertainty variables and a shared energy coordinate.
4. **Plan prefix:** `issue-11`.

## Assumptions (flag if wrong)

- **Input product:** HIT L1B `standard-rates` (`source="hit"`, `data_type="l1b"`, `descriptor="standard-rates"`) with `ialirtrates`, `ialirtrates_stat_uncert_plus` and `ialirtrates_stat_uncert_minus`, each of shape `(epoch, 20)`. *Verified* on the 2026-06-01 file.
- **Δt = 1 minute:** each L1B epoch is one HIT science frame, so the 60·Δt divisor is the constant 60 s. *Verified*: the epoch spacing is exactly 60 s and the values are counts per frame.
- **Uncertainty fill handling:** the L1B uncertainty variables have no `FILLVAL`. The reader takes them as stored and sets them to NaN wherever `ialirtrates` is fill, so a fill rate never comes with a finite uncertainty.
- **Variable naming:**
  - `electron_intensity_a_side` is the A side (doc: "particles propagating toward the Sun").
  - `electron_intensity_b_side` is the B side ("propagating away from the Sun").
  - Names follow the instrument team's A/B side terminology. CATDESC and `VAR_NOTES` spell out the propagation direction.
- **Energy coordinate:**
  - `electron_energy` is the arithmetic midpoint of each bin's bounds, with `electron_energy_delta_minus` and `electron_energy_delta_plus` both equal to ΔE/2.
  - Bin bounds are the cumulative sum of the ancillary ΔE: 0–0.3, 0.3–0.4, 0.4–0.8, 0.8–1.2, 1.2–2.0 and 2.0–90 MeV.
  - Labels are `<300 keV`, `300-400 keV`, `400-800 keV`, `800-1200 keV`, `1200-2000 keV` and `>2000 keV`.
  - The open-ended bins (<300, >2000) make the midpoint a convention, not a physical mean energy.
- **Uncertainties:** `delta_± = σ_±(L1B) / (60·ΔE·G)`. The background has no uncertainty term (b = 0 and no σ_b is given).
- **Units:** `#/(cm^2-s-sr-MeV)`.
- **Ancillary:**
  - SDC name `imap_hit_electron-intensity-factors_20260513_v001.csv`, descriptor `electron-intensity-factors`, start date set to the 2026-05-13 cutoff.
  - The file keeps the attached header (`IALIRT_Rate_A_side,IALIRT_Rate_B_side,delta_E_MeV,Gfavg_cm2_sr,b`) and UTF-8 BOM. It is read with `encoding="utf-8-sig"`.
- **Epoch and epoch delta:**
  - The output `epoch` is the L1B `epoch`, copied through unchanged. It doesn't enter the intensity calculation, since Δt is the constant 60 s. The 2026-05-13 cutoff uses `input_metadata.start_date`, not `epoch`.
  - `epoch_delta` = 30 s is wired as both `DELTA_PLUS_VAR` and `DELTA_MINUS_VAR` of `epoch`, so each record claims to cover `epoch ± 30 s`.
  - That matches how L1A builds `epoch`: `imap_processing/hit/l0/decom_hit.py` sets each science frame's epoch to the mean of its 20 packet times, "to use as the data collection time". `epoch` is therefore the frame centre, and HIT L2 macropixel uses the same convention (300 s for its 10-minute cadence).
- **Quality flags:** `hit_flags` is written as all `HitL3Flags.NONE`. No electron-specific flag is defined by the doc.
- **Fill handling:** L1B FILLVAL → NaN on read. NaN in a rate yields NaN in that epoch/bin's intensity and uncertainties, written as the CDF FILLVAL.

## Open issues to raise with the HIT team (do not block implementation)

- **6 vs 5 bins:** §9.3 says 6 bins per side, but the I-ALiRT list in §8.2 drops the <300 keV bin. This plan follows §9.3 (decision 1). On 2026-06-01 that bin holds 0 counts on the B side all day and at most 1 count per frame on the A side, so it will be almost all zeros. Worth confirming with the HIT team before release.
- **L1B `epoch` vs accumulation window (low priority):** L1A stamps each frame at the mean of its packet times. If the counts telemetered in a frame were accumulated over the *previous* minute, the true collection centre would be about 60 s earlier. The doc's 1-minute dynamic-threshold offset (§9.1) hints at this kind of lag for singles rates. It would shift only the time tags, not the intensities.
- **Proton contamination:** §8.2 says the channels "will have different proton contamination levels that will vary with time", yet `b = 0` everywhere. Ask whether a time-dependent background ancillary is planned. Background is already subtracted, so a non-zero `b` would be a data-only change.
- **Energy coordinate convention:** ask whether the midpoint is acceptable or a spectrum-weighted mean energy (e.g. for the assumed E⁻³ spectrum) is wanted, especially for the open-ended bins.
- **Doc inconsistencies:**
  - Tables 37 and 52 number the same bins 1–6 and 0–5.
  - The pre-2026-05-13 slice list gives six edges but the table shows five bins.
- **L1B uncertainty attributes:** `ialirtrates_stat_uncert_plus/minus` on the SDC file have no `FILLVAL`, `CATDESC`, `VAR_TYPE` or `FIELDNAM`. That's an L1B ISTP gap to report to the `imap_processing` HIT maintainers. The reader handles it (see Assumptions) and doesn't depend on it being fixed.

## Doc references

- [`.cursor/rules/test-quality-coverage.mdc`](../../rules/test-quality-coverage.mdc): behavioral tests, numpy assertions, fill/NaN handling, full-result assertions, full `to_data_product_variables()` assertion, no regression pins
- [`.cursor/rules/agent-guidelines.mdc`](../../rules/agent-guidelines.mdc): fail explicitly (missing/malformed inputs), type annotations, `MODULE` patch-path constant, imports at top
- [`.cursor/rules/no-comments.mdc`](../../rules/no-comments.mdc): no code comments
- [`.cursor/README.md`](../../README.md): card → phased plan → TDD → review workflow
- [`documents/HIT_Algorithm_Document-v1p11p05.pdf`](documents/HIT_Algorithm_Document-v1p11p05.pdf): §2.5, §8.2, §9.3, Eq. 14/32, Tables 23, 36, 52, 53 (git-ignored; see Source documents above)
- [`documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv`](documents/ialirt_electron_intensity_factors_v1_2026_09_09.csv): ancillary factors (git-ignored)
