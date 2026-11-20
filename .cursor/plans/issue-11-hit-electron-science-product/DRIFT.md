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
