# Phase 6 — Direct-events metadata and PHA event processing follow the PR 255 review

> **Depends on:** Phase 3 (restructures the `process_pha_event` code it introduced), Phase 5 (the integration test produces the CDF whose attributes are checked here)
> **Blocks:** none
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — only write tests that drive production changes; test behavior, not implementation
> - [`.claude/rules/no-comments.md`](../../rules/no-comments.md) — the refactor must read clearly without comments
> - [`phase-3-incident-energy-window-correction.md`](phase-3-incident-energy-window-correction.md), [`phase-4-charge-without-energy-bounds.md`](phase-4-charge-without-energy-bounds.md), [`phase-5-direct-events-end-to-end-v1p11-ancillaries.md`](phase-5-direct-events-end-to-end-v1p11-ancillaries.md)
> - Review on [IMAP-Science-Operations-Center/imap_L3_processing#255](https://github.com/IMAP-Science-Operations-Center/imap_L3_processing/pull/255) by jtniehof, 2026-11-20 (changes requested)

## Goal

Address the PR 255 review quickly, with no change to the science output:

- **Metadata:** `energy`, `energy_in_detector`, `delta_e` and `e_prime` get a short `CATDESC`, and the correction details move to `VAR_NOTES`.
- **Readability:** in `process_pha_event`, the cosine correction is computed right where it is applied, and is named `cosine_correction`. The NOCALC case returns early, so the calculation is no longer nested under an `if` whose `else` sits 25 lines below.
- **Review questions:** reply to the comments that only ask a question, and handle the two follow-ups outside this repo.

## Review comments and how this phase handles them

| # | Where | Comment (summary) | Action |
|---|---|---|---|
| 1 | YAML `energy` (L56) | CATDESC "Incident Energy"; VAR_NOTES "Sum of window-corrected energies deposited in each participating detector" | 6.1 |
| 2 | YAML `energy_in_detector` (L74) | CATDESC "Energy deposited in detector_id detector"; VAR_NOTES "no cosine correction" | 6.1 |
| 3 | YAML `delta_e` (L107) | CATDESC "Delta E (energy deposited in last detector that particle fully penetrates)"; VAR_NOTES "Cosine corrected for species lookup table" | 6.1 |
| 4 | YAML `e_prime` (L125) | CATDESC "E prime (energy deposited in stopping detector)"; VAR_NOTES "Cosine corrected for species lookup table" | 6.1 |
| 5 | `calculate_pha.py` L167 | Move the cosine `correction` down to where it is applied and rename it `cosine_correction` | 6.2 |
| 6 | `calculate_pha.py` L179 | Use `if not event_analysis: return ...` and dedent the calculation | 6.2 |
| 7 | `test_calculate_pha.py` L547 | No explicit check of how many times the window correction is called/applied | 6.3 (reply, no test change) |
| 8 | `test_calculate_pha.py` L625 | What is the significance of the not-strictly-increasing test? | 6.3 (reply) |
| 9 | `window_correction_lookup_table.py` L22, `test_calculate_pha.py` L607, `run_local.py` L543 | 👍 / agreement / diff remark | No action |
| R1 | Review body | sds-data-manager PR to add the new `wincorr` dependency | 6.4 (outside repo) |
| R2 | Review body | Make a sample day (2026-07-30) and send it to Grant, Alessandro and Eric with a link to the metadata suggestions, in one email | 6.4 (outside repo) |

## Gherkin Feature & Scenarios

```gherkin
Feature: Direct-events metadata and PHA event processing follow the PR 255 review

  Scenario Outline: Direct-events energy variables carry a short CATDESC and the correction detail in VAR_NOTES
    Given the HIT L3 direct-events CDF written by test_hit_direct_events_product
    When the attributes of <variable> are read
    Then its CATDESC is "<catdesc>"
    And its VAR_NOTES is "<var_notes>"

    Examples:
      | variable           | catdesc                                                                     | var_notes                                                              |
      | energy             | Incident Energy                                                             | Sum of window-corrected energies deposited in each participating detector |
      | energy_in_detector | Energy deposited in detector_id detector                                    | No cosine correction                                                   |
      | delta_e            | Delta E (energy deposited in last detector that particle fully penetrates)  | Cosine corrected for species lookup table                              |
      | e_prime            | E prime (energy deposited in stopping detector)                             | Cosine corrected for species lookup table                              |

  Scenario: process_pha_event output is unchanged after the restructure
    Given the existing process_pha_event tests for analyzed and NOCALC events
    When process_pha_event computes the cosine correction just before applying it and returns early for NOCALC events
    Then every existing test in test_calculate_pha passes without modification
```

## Affected files

### Production

- `imap_l3_processing/cdf/config/imap_hit_l3_direct-events_variable_attrs.yaml`: `CATDESC` and new `VAR_NOTES` for `energy`, `energy_in_detector`, `delta_e`, `e_prime`.
- `imap_l3_processing/hit/l3/pha/science/calculate_pha.py`: `process_pha_event` only.

### Tests

- None changed. The metadata is checked by reading the integration-test output CDF (6.1), and the refactor is covered by the existing `process_pha_event` tests (6.2).

## TDD steps

### 6.1 Scenario Outline: Direct-events energy variables carry a short CATDESC and the correction detail in VAR_NOTES

This is a metadata-only change. No unit test asserts YAML text, and adding one would only pin strings, so there is no Red step.

- In the YAML, set each variable's `CATDESC` and add a `VAR_NOTES` key directly after it, using the Examples table. `energy_in_detector` uses "No cosine correction" (capitalised as a sentence, the reviewer's wording otherwise).
- The `hit_flags` entry in the same file already uses `VAR_NOTES`, so `ImapAttributeManager` writes it through with no code change.
- Check: run `uv run python -m unittest tests.integration.test_hit_processor_integration.TestHitProcessorIntegration.test_hit_direct_events_product` (confirm the class name first), then use the `read-cdf` skill on the output `imap_hit_l3_direct-events_20100105_v001.cdf` and confirm all four variables show the new `CATDESC` and `VAR_NOTES`.

### 6.2 Scenario: process_pha_event output is unchanged after the restructure (Refactor only)

All tests in `tests/hit/l3/pha/science/test_calculate_pha.py` are green before starting. Then:

- Replace `if event_analysis: ... else: return EventOutput(...None...)` with an early return:
  - `if not event_analysis:` returns the NOCALC `EventOutput` (`energies` set, every other output `None`).
  - The analyzed-event calculation follows, dedented one level.
- Order the analyzed-event calculation so each correction sits next to its use:
  1. Window-corrected `total_energy` loop.
  2. `cosine_correction = cosine_table.get_cosine_correction(event_analysis.range, event_analysis.l1_detector, event_analysis.l2_detector)`.
  3. `e_delta = cosine_correction * ...` and `e_prime = cosine_correction * ...`.
  4. `compute_charge(...)` and the analyzed `EventOutput`.
- `energies` stays computed once above the early return, because both outputs use it.
- Re-run `uv run python -m unittest tests.hit.l3.pha.science.test_calculate_pha`. All tests must pass unchanged. If any test needs editing, behavior changed and the refactor is wrong.

### 6.3 Replies to the review questions (no code change)

Draft these for the user to post. Do not post them without the user's OK.

- **Comment 7 (window correction call count):** The test already pins which factors are applied, and how many times. It uses a real `WindowCorrectionLookupTable` whose factors are non-unity for both summed addresses (19 and 8) and differ between the Range 2 row and the Range 3/4 row. It runs all three ranges. The excluded word (address 7) would add `mev_7` if it were summed. A factor applied twice would square it. Each of those mistakes changes `total_energy`, so the exact `assert_allclose` on `total_energy` catches it. Asserting the lookup's call count as well would test the implementation rather than the result, so the test is left as is. If the reviewer still wants it, wrap the table in `Mock(wraps=...)` and assert `get_correction_factor.call_args_list == [call(range, 19), call(range, 8)]`.
- **Comment 8 (not-strictly-increasing test):** `compute_charge` evaluates every species' track at the measured E′, which gives the ΔE each species would deposit at that E′. It then interpolates the charge between the two neighboring tracks that bracket the measured ΔE (Eq. 62–64). That only works if the track ΔE values increase with Z at that E′. If two tracks cross, the bracketing pair, and the log ratio in Eq. 63, are not defined. This used to be an `assert` that aborted the whole file. Once the Table 45 bounds were removed it became reachable, both outside the old bounds with the old tables and inside them with the new Table 46: the Si row crosses Al and S for roughly 0.2–860 MeV E′ in Range 2. Those events now get NaN charge and the rest of the file is processed. The test covers a crossed pair, an equal pair (log of 1 is 0) and a NaN track.
- **Comment 9:** Optionally acknowledge it. No reply is needed.

### 6.4 Follow-ups outside this repo

These are tracked here so the review can be closed, but they are not code in this repo:

- **R1 — sds-data-manager:** open a PR there adding the `wincorr` ancillary as a HIT L3 `direct-events` dependency (file `imap_hit_wincorr_20250901_v000.csv`). Its scope is a separate repo, so it gets its own session or branch.
- **R2 — sample day 2026-07-30:** after 6.1 and 6.2 are pushed, produce `imap_hit_l3_direct-events_20260730_v001.cdf`. Run `imap_l3_data_processor.py` with `--start-date 20260730` and the same ancillary list as `run_local_with_upload_and_download.py`'s `hit` / `direct-events` case, plus the 2026-07-30 L1A direct-events file (look up its current version with `imap-data-access`). Do not upload to the SDC. The user sends the file, with a link to this PR's metadata comments, to Grant, Alessandro and Eric in one email.

## Done criteria

- [ ] The four YAML entries match the Examples table, and `read-cdf` on the integration-test output shows the new `CATDESC` and `VAR_NOTES`
- [ ] `process_pha_event` returns early for NOCALC events and computes `cosine_correction` immediately before `e_delta` / `e_prime`
- [ ] `uv run python -m unittest tests.hit.l3.pha.science.test_calculate_pha` passes with no test edits
- [ ] `uv run python -m unittest tests.integration.test_hit_processor_integration` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] Replies for comments 7 and 8 are drafted and given to the user to post
- [ ] R1 and R2 are handed to the user (sds-data-manager PR, sample-day CDF ready to send)
