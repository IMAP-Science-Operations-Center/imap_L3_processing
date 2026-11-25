# Card 3485 — HIT: Update direct events to Algorithm Document v1.11.05

## Card reference

- **Card:** 3485
- **GitHub issue:** [IMAP-Science-Operations-Center/imap_L3_processing#69](https://github.com/IMAP-Science-Operations-Center/imap_L3_processing/issues/69) — "HIT: Update direct events - Card 3485"
- **Source documents:** `.cursor/plans/update-direct-events/HIT_Algorithm_Document-v1p11p05.pdf` (new, §9.1 "PHA Products") compared against `.cursor/plans/update-direct-events/HIT_Algorithms_v1p10p19_11_21_2025.pdf` (the version the current code implements), plus the HIT-team email notes captured in the issue comments.

## Goal

The HIT L3 `direct-events` product computes per-event incident energy, deposited energies, ΔE / E′ and charge the way HIT Algorithm Document v1.11.05 §9.1 and the HIT team's email clarifications describe:

- **Incident energy (`energy`)** is the sum of the uncorrected calibrated MeV of the participating detector segments, each multiplied by its window (foil) correction factor `k = WINCORR[address] / 256`. Range 2 uses WINCORR2 and Ranges 3 and 4 use WINCORR3. The WINCORR arrays come from a new `wincorr` ancillary file.
- **Per-detector energies (`energy_in_detector`)** are the uncorrected calibrated MeV. The cosine correction no longer applies to them or to `energy`.
- **ΔE and E′ (`delta_e`, `e_prime`)** keep the cosine correction K, which is still applied only for the charge calculation.
- **Charge** is computed for every Range 2/3/4 event, with no Table 45 E′/ΔE bounds check. When the power-law interpolation is undefined (ΔE ≤ 0, E′ ≤ 0, or charge tracks not strictly increasing at E′), that event's charge is NaN and processing continues. Today this case raises an `AssertionError` that stops the whole file.
- The processor runs end to end on the v1.11.05 ancillary set: the new WINCORR, cosine v001, lo-gain 2026-05-14 v001, event-type v003, and charge-fit v001 built from doc Tables 46–48.

## What changed between v1.10.19 and v1.11.05 (and what it means for code)

| Change in doc / email | Code impact |
|---|---|
| New §9.1.1 WINCORR2 / WINCORR3 foil correction (Table 40), applied when summing `energy` (email: per segment, by detector address, `k = WINCORR/256`) | **New** ancillary reader, dependency and energy sum (Phases 1–3) |
| §9.1.1 no longer says incident energy "factors in a cosine correction" | **Change:** `energy` and `energy_in_detector` are not cosine-corrected (Phase 3, user decision) |
| Email #4: Table 45 E′/ΔE bounds are informational only | **Change:** remove `valid_ranges` gating in `compute_charge` (Phase 4, user decision) |
| Removing the bounds exposes E′ where tracks are not in Z order (current tables: R2 E′ > ~5650 MeV, R3 E′ < ~0.003 MeV, R4 E′ > ~11 700 MeV; new Table 46: inside the bounds too) | **Change:** NaN instead of `assert` crash (Phase 4, user decision) |
| Tables 46–48 charge-fit coefficients changed | **Ancillary only:** v001 charge-fit files built from the doc (Phase 5, user decision). Format unchanged. |
| Tables 42–44 cosine K / power-law index `a` changed; email #7: new cosine tables apply for all time | **Ancillary only:** cosine v001 files from issue #69. Format unchanged, comment line already supported on `origin/main`. |
| Table 41 now written `E = a·PHA − b` with positive `b`; 2026-05-14 L1A/B14 low-gain offset drop (email #6) | **Ancillary only:** the lo-gain files keep the `a·adc + b` / negative-`b` convention, so `calculate_mev` is unchanged. The SDC selects the dated lo-gain file. |
| Email #1: PENA? / PENB? are NOCALC | **Already deployed:** SDC `hit-event-type-lookup_20250228_v003.csv` already has `NOCALC` for those rows. Only the repo's v000 fixture is stale (Phase 5 adds the v003 fixture). |
| Email #2: compute range only from groups present in the PHA words | **No change:** `HitEventTypeLookup.lookup_range` already uses only the set of groups present |
| Email #5: A0/B0 (I-ALiRT aperture) events remain NOCALC | **No change:** every L1A0/L1B0 row is already NOCALC, and the A0/B0 cosine/charge tables (Tables 49–51) are not used |
| Issue requirement: check units of `energy_in_detector` | **No change:** MeV, fixed earlier in #128; the gain tables are MeV |
| Table 39 event classification | **No change:** row-by-row comparison against `hit-event-type-lookup` CSV found no differences besides the PENA?/PENB? range |

## Prerequisite

Local `main` is **3 commits behind `origin/main`**. The missing commits are #253's comment-stripping in the cosine/gain readers and the side-B cosine table fix `bf08c6bc`. Implementation must start from a branch based on `origin/main`, because the new cosine and lo-gain files begin with a `#` comment line.

## Affected areas

### Production

- `imap_l3_processing/hit/l3/pha/science/window_correction_lookup_table.py` *(new)*
- `imap_l3_processing/hit/l3/pha/hit_l3_pha_dependencies.py`
- `imap_l3_processing/hit/l3/pha/science/calculate_pha.py`
- `imap_l3_processing/hit/l3/hit_processor.py`
- `imap_l3_processing/cdf/config/imap_hit_l3_direct-events_variable_attrs.yaml` (`energy`, `energy_in_detector`, `delta_e`, `e_prime` CATDESC and VAR_NOTES)
- `run_local.py` (`create_hit_direct_event_cdf`), `run_local_with_upload_and_download.py` (HIT direct-events dependency list): developer scripts, no tests

### Tests

- `tests/hit/l3/pha/science/test_window_correction_lookup_table.py` *(new)*
- `tests/hit/l3/pha/test_l3_pha_dependencies.py`
- `tests/hit/l3/pha/science/test_calculate_pha.py`
- `tests/hit/l3/test_hit_processor.py`
- `tests/integration/test_hit_processor_integration.py`
- New fixtures in `tests/test_data/hit/pha_events/`:
  - `imap_hit_wincorr_20250901_v000.csv`
  - `imap_hit_range-{2A,2B,3A,3B,4A,4B}-cosine-lookup_20250203_v001.csv`
  - `imap_hit_lo-gain-lookup_20260514_v001.csv`
  - `imap_hit_range-{2A,2B,3A,3B,4A,4B}-charge-fit-lookup_20250319_v001.csv`
  - `imap_hit_hit-event-type-lookup_20250228_v003.csv`

## Phase summary

| # | Phase file | Gherkin Feature | Description |
|---|---|---|---|
| 1 | [`phase-1-wincorr-lookup-table.md`](phase-1-wincorr-lookup-table.md) | HIT WINCORR ancillary provides per-range detector window correction factors | New reader turning the two 64-entry WINCORR rows into `k = value/256` by range and detector address, failing loudly on malformed files |
| 2 | [`phase-2-fetches-wincorr-dependency.md`](phase-2-fetches-wincorr-dependency.md) | HIT direct-events dependencies include the WINCORR lookup | `HitL3PhaDependencies.fetch_dependencies` downloads and parses the `wincorr` ancillary, and raises a clear error when it is missing |
| 3 | [`phase-3-incident-energy-window-correction.md`](phase-3-incident-energy-window-correction.md) | Direct-event energies follow the v1.11.05 incident energy calculation | `energy` = Σ k·MeV with no cosine correction; `energy_in_detector` uncorrected; ΔE/E′ keep cosine K; the processor passes the WINCORR lookup through |
| 4 | [`phase-4-charge-without-energy-bounds.md`](phase-4-charge-without-energy-bounds.md) | Charge is computed without the Table 45 energy bounds | Remove the bounds gating; NaN (not a crash) when the power-law interpolation is undefined |
| 5 | [`phase-5-direct-events-end-to-end-v1p11-ancillaries.md`](phase-5-direct-events-end-to-end-v1p11-ancillaries.md) | HIT L3 direct-events product is produced from the v1.11.05 ancillary set | Add the new ancillary fixtures (including charge-fit v001 built from Tables 46–48), a direct-events integration test, and updated run_local helpers |
| 6 | [`phase-6-address-pr-255-review.md`](phase-6-address-pr-255-review.md) | Direct-events metadata and PHA event processing follow the PR 255 review | Short CATDESC + VAR_NOTES for the energy variables, restructure `process_pha_event` (early return, `cosine_correction` next to its use), reply to review questions, hand off sds-data-manager PR and 2026-07-30 sample day |

## Dependency graph

```
Phase 1 (wincorr lookup table)
   └─▶ Phase 2 (fetch wincorr dependency)
          └─▶ Phase 3 (incident energy uses WINCORR, cosine only on ΔE/E′)
                 └─▶ Phase 5 (end-to-end with v1.11.05 ancillaries)
                        └─▶ Phase 6 (PR 255 review)
Phase 4 (charge without energy bounds) ───────────▶ Phase 5
```

| Phase | Depends on | Blocks |
|---|---|---|
| 1 | none | 2, 3 |
| 2 | 1 | 3, 5 |
| 3 | 1, 2 | 5, 6 |
| 4 | none | 5 |
| 5 | 2, 3, 4 | 6 |
| 6 | 3, 5 | none |

Phase 4 is independent of Phases 1–3 and can be done in any order relative to them.

## Decisions recorded (from the user, 2026-11-18)

1. **Cosine scope:** cosine K applies only to ΔE and E′. `energy` and `energy_in_detector` use uncorrected calibrated MeV, and `energy` additionally applies WINCORR `k`.
2. **E bounds:** remove the Table 45 bounds check in `compute_charge`.
3. **PENA? / PENB?:** handled by a new ancillary version, not by code. The SDC already has it as `hit-event-type-lookup_20250228_v003.csv`.
4. **Charge-fit:** build v001 charge-fit ancillaries from doc Tables 46–48 (A and B sides identical, as in v000).
5. **Non-monotone tracks / non-positive energies:** charge = NaN for that event; processing continues.

## Assumptions (flag if wrong)

- **WINCORR file layout:** the attached `imap_hit_wincorr_20250901_v000.csv` has `#` comment lines followed by exactly two comma-separated rows of 64 integers. Row 1 is WINCORR2 (Range 2) and row 2 is WINCORR3 (Ranges 3 and 4). The file has no labels; the order is inferred from the values matching doc Table 40 (390 vs 419 for L1x14). The attached values differ slightly from doc Table 40 (391/420 vs 421/424 for L1x0). The file comes from the newer `WINCORR_tables_20260630.dat`, so the file wins.
- **WINCORR applies only to `energy`** (the email says so), not to `energy_in_detector`, `delta_e` or `e_prime`. It is looked up for every summed segment by address; non-L1 addresses are 256 (k = 1).
- **`energy` sums the same words as today:** the highest-signal unsaturated word in each included detector group (`EventAnalysis.words_with_highest_energy`).
- **Charge-fit v001 start date** stays `20250319`, the same as v000, so it supersedes v000 for all time, like the cosine v001 files. The A side and B side files are identical per range. Only Tables 46–48 (A14/B14 apertures) are used; Tables 49–51 (A0/B0) are not, because those events are NOCALC.
- **"Strictly increasing"** is the monotonicity criterion. Equal adjacent track values make the log-ratio interpolation divide by zero, so they also yield NaN.

## Open issues to raise with HIT / Jon (do not block implementation)

- **Table 46 (Range 2, A14/B14) Si row looks wrong.** `14, 3.307547e+02, -1.016750e-01, 2.434280e+04, -3.028693e-09, -2.299653e-01`: `a1` is ~4× its neighbors (Al 71.0, S 93.5), and the old value was 79.7. With this row the Si track crosses Al and S across about half of the Range 2 E′ bounds (0.2–860 MeV). After Phase 4 that gives NaN charge for those Range 2 events instead of crashing. Confirm or correct before the charge-fit v001 files are uploaded to the SDC.
- **The lo-gain 2026-05-14 v001 attachment has the same values as the deployed `lo-gain-lookup_20250203_v000.csv`**; only the comment line differs. Doc §9.1.2 says the L1A/B14 low-gain offsets dropped by 8 on 2026-05-14, so either v000 already holds the adjusted offsets or the wrong file was attached.
- **Doc §9.1.2 mentions "a further adjustment … on <Month> <Day>, 2026"** to the low-gain offsets, which is still a placeholder. That will be another dated lo-gain ancillary, with no code change.
- **WINCORR v000 naming:** Jon asked whether the first WINCORR file should be v000 or v001. The plan uses the attached name `imap_hit_wincorr_20250901_v000.csv`.

## Doc references

- [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — behavioral tests, numpy assertions, fill/NaN handling, full-result assertions, no regression pins
- [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — fail explicitly (malformed/missing WINCORR), type annotations, `MODULE` patch-path constant, imports at top
- [`.claude/rules/no-comments.md`](../../rules/no-comments.md) — no code comments (CSV `#` provenance lines in ancillary fixtures are data, not code)
- [`.claude/README.md`](../../README.md) — card → phased plan → TDD → review workflow
- HIT Algorithm Document v1.11.05 §9.1 (Tables 39–48, Eq. 60–64) and issue #69 email notes
