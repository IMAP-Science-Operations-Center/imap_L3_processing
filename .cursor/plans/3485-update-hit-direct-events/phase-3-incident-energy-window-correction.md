# Phase 3 — Direct-event energies follow the v1.11.05 incident energy calculation

> **Depends on:** Phase 1 (`WindowCorrectionLookupTable` and its `get_correction_factor`), Phase 2 (`HitL3PhaDependencies.window_correction_lookup` exists for the processor to pass through)
> **Blocks:** Phase 5 (the end-to-end run exercises the new energy calculation), Phase 6 (restructures `process_pha_event` per the PR 255 review)
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — numpy float assertions, assert every field, `NumpyArrayMatcher` / mock-call patterns
> - [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — type annotations, `MODULE` patch constant
> - [`phase-1-wincorr-lookup-table.md`](phase-1-wincorr-lookup-table.md) — WINCORR lookup API
> - [`phase-2-fetches-wincorr-dependency.md`](phase-2-fetches-wincorr-dependency.md) — the `window_correction_lookup` dependency field
> - [`overview.md`](overview.md) — user decision: cosine K applies only to ΔE / E′
> - HIT Algorithm Document v1.11.05 §9.1.1–9.1.3; issue #69 WINCORR email notes

## Goal

For every event classified as Range 2, 3 or 4:

- `energy_in_detector` (`EventOutput.energies`) is the **uncorrected** calibrated MeV of each PHA word.
- `energy` (`EventOutput.total_energy`) is the sum over the participating segments, meaning the highest-signal word in each included group (`words_with_highest_energy`), of `k × MeV`. Here `k = WINCORR[address] / 256`, taken from WINCORR2 for Range 2 and WINCORR3 for Ranges 3 and 4. No cosine correction is applied.
- `delta_e` and `e_prime` remain `K × MeV`, where K is the cosine correction for the event's L1/L2 segment pair, and charge is still computed from them.

NOCALC events are unchanged: they keep uncorrected per-detector energies and have no total energy, range, ΔE, E′ or charge. The processor passes the fetched WINCORR lookup into `process_pha_event` for every event. The CDF variable descriptions for `energy` and `energy_in_detector` stop claiming a cosine correction.

## Gherkin Feature & Scenarios

```gherkin
Feature: Direct-event energies follow the v1.11.05 incident energy calculation

  Background:
    Given gain coefficients for detector addresses 19 (L1A1a, low gain), 8 (L2A1, high gain) and 7 (L2A2, high gain)
    And a cosine correction table that returns K for the event's L1/L2 segment pair
    And a WINCORR lookup whose Range 2 row and Range 3/4 row give different, non-unity k for addresses 19 and 8
    And a raw PHA event with words at addresses 19, 8 and 7
    And event analysis selects the words at addresses 19 and 8 as the highest-energy words, 19 as the ΔE word and 8 as the E′ word

  Scenario Outline: Incident energy sums window-corrected uncorrected MeV with the range's WINCORR row
    Given event analysis classifies the event as <detected_range>
    When process_pha_event is called
    Then energies are the uncorrected MeV of the words at addresses 19, 8 and 7, in event order
    And total_energy is k(<wincorr_row>, 19) × MeV(19) + k(<wincorr_row>, 8) × MeV(8)
    And delta_e is K × MeV(19) and e_prime is K × MeV(8)
    And compute_charge is called with <detected_range>, K × MeV(19), K × MeV(8) and the range fit lookup
    And charge is compute_charge's result and detected_range is <detected_range>

    Examples:
      | detected_range | wincorr_row |
      | Range 2 side A | WINCORR2    |
      | Range 3 side A | WINCORR3    |
      | Range 4 side B | WINCORR3    |

  Scenario: The processor passes the WINCORR lookup to every direct event
    Given HitL3PhaDependencies with a window_correction_lookup
    And L1A direct-event binaries that decode into three raw PHA events
    When HitProcessor processes the "direct-events" descriptor
    Then process_pha_event is called for each raw event with the dependencies' cosine, gain, window correction, range fit and event type lookups
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/pha/science/calculate_pha.py`: `process_pha_event` gains a `window_correction_table: WindowCorrectionLookupTable` parameter placed after `gain_table`. Energies and total energy follow the new rules.
- `imap_l3_processing/hit/l3/hit_processor.py`: `process_direct_event_product` passes `direct_event_dependencies.window_correction_lookup`.
- `imap_l3_processing/cdf/config/imap_hit_l3_direct-events_variable_attrs.yaml`: `CATDESC` for `energy`, `energy_in_detector`, `delta_e` and `e_prime`.

### Tests

- `tests/hit/l3/pha/science/test_calculate_pha.py`: rewrite `test_process_pha_event` as the Scenario Outline. Add the new positional argument to `test_process_event_for_event_with_invalid_detector_sequences`'s call, which currently passes four `Mock()`s.
- `tests/hit/l3/test_hit_processor.py`: `test_process_direct_event_product`'s `mock_process_pha_event.assert_has_calls`.

## TDD steps

### 3.1 Red — Scenario Outline: Incident energy sums window-corrected uncorrected MeV with the range's WINCORR row

- In `test_calculate_pha.py`, add `MODULE = "imap_l3_processing.hit.l3.pha.science.calculate_pha"` and use it for this test's patch targets (`f"{MODULE}.compute_charge"`, `f"{MODULE}.analyze_event"`).
- Replace `test_process_pha_event` with `test_process_pha_event_sums_window_corrected_energies_and_cosine_corrects_only_delta_e_and_e_prime`:
  - Words: `Detector.from_address(19)` (low gain), `Detector.from_address(8)` (high gain), `Detector.from_address(7)` (high gain), each with a distinct ADC value and a distinct `Gain(a, b)` in the gain table, keyed by `DetectorGain` and address.
  - Build a real `WindowCorrectionLookupTable(range_2_wincorr, range_3_and_4_wincorr)` from two 64-element integer arrays that are 256 everywhere except `[19]` and `[8]`, with different values in the two rows, e.g. R2 row `[19]=300, [8]=270` and R3/4 row `[19]=500, [8]=400`. A real table is used instead of a Mock so the test checks the range-to-row behavior rather than a call pattern.
  - `mock_cosine_correction_table.get_cosine_correction.return_value = cosine_k` (e.g. 0.93).
  - For each Examples row, use `subTest` with a real `DetectedRange` (`R2/A`, `R3/A`, `R4/B`). Inside the subTest, `mock_analyze_event.reset_mock()` / `mock_compute_charge.reset_mock()` and set `mock_analyze_event.return_value = EventAnalysis(range=detected_range, l1_detector=..., l2_detector=..., e_delta_word=word19, e_prime_word=word8, words_with_highest_energy=[word19, word8])`.
  - Call `process_pha_event(raw_event, mock_cosine_table, gain_table, window_table, sentinel.range_fit_lookup, sentinel.rule_lookup)`.
  - Assert **every** `EventOutput` field:
    - `original_event`, `detected_range` and `charge` (== `mock_compute_charge.return_value`) with `assertEqual`
    - `energies` with `np.testing.assert_allclose(actual, [mev19, mev8, mev7])`, the uncorrected values
    - `total_energy` with `assert_allclose(actual, k_row[19] / 256 * mev19 + k_row[8] / 256 * mev8)`
    - `e_delta` / `e_prime` with `assert_allclose(actual, cosine_k * mev19)` / `cosine_k * mev8`
  - Assert `mock_cosine_table.get_cosine_correction.assert_called_once_with(detected_range, l1_detector, l2_detector)`.
  - Assert `mock_compute_charge.assert_called_once_with(detected_range, cosine_k * mev19, cosine_k * mev8, sentinel.range_fit_lookup)`. The expected ΔE/E′ use the same single multiplication as production, so exact equality holds.
- Fails: `TypeError` (unexpected positional argument), then wrong energies and total because cosine K is still applied and WINCORR is not.

### 3.2 Green

- In `calculate_pha.py`:
  - Import `WindowCorrectionLookupTable`.
  - Change the signature to `process_pha_event(event: RawPHAEvent, cosine_table: CosineCorrectionLookupTable, gain_table: GainLookupTable, window_correction_table: WindowCorrectionLookupTable, range_fit_lookup: RangeFitLookup, rule_lookup: HitEventTypeLookup) -> EventOutput`.
  - In the analyzed branch:
    - `energies = [calculate_mev(word, gain_table) for word in event.pha_words]`
    - `total_energy = sum(window_correction_table.get_correction_factor(event_analysis.range.range, word.detector.address) * calculate_mev(word, gain_table) for word in event_analysis.words_with_highest_energy)`
    - `e_delta = cosine_correction * calculate_mev(event_analysis.e_delta_word, gain_table)`; same pattern for `e_prime`.
- Update `test_process_event_for_event_with_invalid_detector_sequences` to pass one more `Mock()`. That test's NOCALC behavior is unchanged and it is not rewritten.

### 3.3 Red — Scenario: The processor passes the WINCORR lookup to every direct event

- In `tests/hit/l3/test_hit_processor.py::test_process_direct_event_product`, change each expected `call(...)` in `mock_process_pha_event.assert_has_calls` to `call(raw_pha_event_n, deps.cosine_correction_lookup, deps.gain_lookup, deps.window_correction_lookup, deps.range_fit_lookup, deps.event_type_lookup)`.
- Fails: the processor still passes 5 arguments. `autospec=True` makes the mock enforce the new 6-parameter signature.

### 3.4 Green

- In `hit_processor.py::process_direct_event_product`, pass `direct_event_dependencies.window_correction_lookup` as the fourth argument.

### 3.5 Refactor

- In `process_pha_event`, extract a local helper or comprehension so `calculate_mev` is computed once per word and reused for `energies`, `total_energy`, `e_delta` and `e_prime`. Today it is recomputed up to three times per word. Keep the branch structure readable.
- Update YAML `CATDESC`s in `imap_hit_l3_direct-events_variable_attrs.yaml` (metadata only, no test):
  - `energy`: e.g. `Incident energy (sum of window-corrected energies deposited in each participating detector)`
  - `energy_in_detector`: e.g. `Energy deposited in detector_id detector (no cosine correction)`
  - `delta_e`: append `, cosine corrected`
  - `e_prime`: append `, cosine corrected`
- Confirm `UNITS: MeV` remains on `energy`, `energy_in_detector`, `delta_e` and `e_prime`. This answers the issue's "check units of `energy_in_detector`" requirement.

## Done criteria

- [ ] Scenario Outline passes for Range 2 (WINCORR2), Range 3 (WINCORR3) and Range 4 (WINCORR3), with every `EventOutput` field asserted
- [ ] `energy_in_detector` values are uncorrected MeV; `energy` = Σ k·MeV over `words_with_highest_energy`; `delta_e` / `e_prime` are cosine-corrected
- [ ] Scenario "processor passes the WINCORR lookup" passes (`test_process_direct_event_product`)
- [ ] `test_process_event_for_event_with_invalid_detector_sequences` still passes with the new argument
- [ ] YAML `CATDESC`s for `energy`, `energy_in_detector`, `delta_e` and `e_prime` describe the new behavior, and units are MeV
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
