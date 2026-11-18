# Drift

Deviations from the plan, recorded as they happen.

## 1. Malformed-file test uses `with_tempdir` (Phase 1, step 1.3)

**When:** 2026-11-18
**Original plan:** Write each malformed WINCORR file with `tempfile.TemporaryDirectory()` or `TempFileTestCase`.
**What changed:** `test_rejects_wincorr_file_without_two_rows_of_64_values` uses the existing `@with_tempdir` helper from `tests/test_helpers.py`.
**Why:** That helper is the repo's temporary-directory fixture and supplies the same per-test directory.

## 2. A jagged WINCORR row keeps numpy's error (Phase 1, step 1.4)

**When:** 2026-11-18
**Original plan:** Wrap a failed `np.loadtxt` in `ValueError(..., from error)` so the message names the file path, including a row of 63 values.
**What changed:** `from_file` lets that `ValueError` propagate. The path is included only when the file loads and `shape != (2, 64)`. The 63-value case asserts `ValueError` and does not require the path.
**Why:** Requested during implementation: inline the shape error and re-raise numpy's error for a file `loadtxt` cannot parse.
