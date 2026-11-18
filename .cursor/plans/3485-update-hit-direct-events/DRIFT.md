# Drift

Deviations from the plan, recorded as they happen.

## 1. Inline ancillary file names (Phase 2.3)

**When:** 2026-11-18
**Original plan:** Pull the ancillary file names into `_ancillary_file_names`, shared by the happy-path test and the missing-wincorr test.
**What changed:** `test_fetch_dependencies` keeps each ancillary file name inline, with the wincorr CSV as the last entry.
**Why:** A second descriptor list plus a helper was more to keep in sync than the one file-path list the test already had.

## 2. No dedicated missing-wincorr error (Phases 2.3–2.5)

**When:** 2026-11-18
**Original plan:** If the wincorr path list is empty, raise `ValueError("Missing wincorr dependency.")`, covered by `test_fetch_dependencies_raises_when_wincorr_dependency_missing`. Optionally extract `_download_required_ancillary` and use it only for wincorr.
**What changed:** WINCORR is downloaded with `get_file_paths(...)[0]`, the same as the other ancillaries. The missing-wincorr test and the helper were not kept.
**Why:** An empty path list already fails at `[0]`. A separate check would be a second failure path only for this ancillary.

## 3. Return annotation on fetch_dependencies (Phase 2.2)

**When:** 2026-11-18
**Original plan:** Did not mention a return annotation.
**What changed:** `fetch_dependencies` is annotated `-> HitL3PhaDependencies`.
**Why:** Modified functions need a return annotation.

## 4. MODULE constant on the NOCALC patches (Phase 3.1)

**When:** 2026-11-18
**Original plan:** Add `MODULE` and use it for the rewritten test's `compute_charge` and `analyze_event` patches.
**What changed:** `test_process_event_for_event_with_invalid_detector_sequences` uses the same `MODULE` constant for its two patches. Each subTest also resets the cosine mock and restores `cosine_k`, because that test asserts the cosine lookup was called once.
**Why:** The NOCALC test was already being edited for the new argument, and `assert_called_once` across subTests needs a fresh cosine mock.

## 5. No per-word MeV cache (Phase 3.5)

**When:** 2026-11-18
**Original plan:** Compute `calculate_mev` once per word and reuse that value for `energies`, `total_energy`, `e_delta`, and `e_prime`.
**What changed:** The uncorrected `energies` list is computed once, above the analyzed/NOCALC branch. `total_energy` is a loop that looks up the window factor, converts that word to MeV, and appends the product. `e_delta` and `e_prime` still call `calculate_mev`.
**Why:** `PHAWord` is an unfrozen dataclass, so a reuse map would be keyed by `id()`. The loop names the window factor and the deposited energy. `calculate_mev` is a multiply-add.
