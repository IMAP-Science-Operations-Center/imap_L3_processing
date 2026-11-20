# Drift

Deviations from the plan, recorded as they happen.

## 1. Bounds and labels folded into the parse test (Phase 1.3)

**When:** 2026-11-20
**Original plan:** A separate `test_energy_bin_bounds_and_labels_are_derived_from_cumulative_bin_widths` loads the same fixture and asserts the bounds and labels.
**What changed:** Those assertions were added to `test_from_file_parses_attached_ancillary_into_per_bin_factors_and_energy_bins`.
**Why:** The setup is the same load, and the bounds and labels are part of that result.

## 2. `with_tempdir` instead of `TempFileTestCase` (Phase 1.5)

**When:** 2026-11-20
**Original plan:** Each malformed-file subtest writes into a `TempFileTestCase` temp directory.
**What changed:** `test_from_file_rejects_malformed_ancillary` uses `@with_tempdir`.
**Why:** That matches `test_window_correction_lookup_table.py`, which already writes defective lookup files this way.

## 3. Shape checks only (Phase 1.5–1.6)

**When:** 2026-11-20
**Original plan:** Reject non-positive ΔE, non-positive G, and an I-ALiRT rate outside 1–20, and define `IALIRT_RATE_COUNT`. An extra A-side rate of 0 was also going to be tested.
**What changed:** `from_file` checks the header and that there are 6 data rows. The ΔE, G, and rate-range checks, their subtests, and `IALIRT_RATE_COUNT` were not kept.
**Why:** The ancillary is a known file. Header and row count are the shape check; the value checks were more validation than that file needs. Phase 3 can add `IALIRT_RATE_COUNT` when the rates-shape check needs it.

## 4. L2 decoy lives in the fetch test (Phase 2.7)

**When:** 2026-11-20
**Original plan:** During refactor, confirm the L1B query passes `data_type="l1b"`, and if needed add an extra L2 file to the 2.3 test.
**What changed:** The fetch test's collection includes an L2 `standard-rates` file from the start, and the download assertion is the exact call list.
**Why:** That makes `data_type="l1b"` a behavior the test requires, instead of a later optional check.

## 5. Expected paths come from `construct_path` (Phase 2.3)

**When:** 2026-11-20
**Original plan:** Assert `download` with `assert_has_calls` on the file names.
**What changed:** The test builds `ScienceFilePath` and `AncillaryFilePath` objects, passes their filenames into the collection, and asserts `call_args_list` equals `construct_path()` for the L1B file and the ancillary.
**Why:** `get_file_paths` returns those constructed paths, and `construct_path` is how `imap_data_access` builds them.

## 6. Missing-input check stays inline (Phase 2.6)

**When:** 2026-11-20
**Original plan:** A module-level `_download_required` raises `ValueError` when `get_file_paths` is empty, and both inputs use it.
**What changed:** Each lookup checks for an empty list and raises inside `fetch_dependencies`.
**Why:** The helper only wrapped two short lookups.
