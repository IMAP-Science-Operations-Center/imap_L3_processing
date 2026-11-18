# Phase 2 — HIT direct-events dependencies include the WINCORR lookup

> **Depends on:** Phase 1 (`WindowCorrectionLookupTable.from_file` must exist)
> **Blocks:** Phase 3 (the processor passes `window_correction_lookup` to `process_pha_event`), Phase 5 (the end-to-end run needs the `wincorr` ancillary to be fetched)
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — test quality rules, isolate `imap_data_access` in unit tests
> - [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — fail explicitly on missing required dependency, `MODULE` patch constant
> - [`phase-1-wincorr-lookup-table.md`](phase-1-wincorr-lookup-table.md) — the table this phase loads

## Goal

When the SDC invokes the HIT L3 `direct-events` job, the dependency collection now includes the `wincorr` ancillary. `HitL3PhaDependencies.fetch_dependencies` downloads it, parses it with `WindowCorrectionLookupTable.from_file`, and exposes it as `window_correction_lookup`. If the collection has no `wincorr` file, fetching fails immediately with `ValueError("Missing wincorr dependency.")`, mirroring the existing L1A check. It does not raise a bare `IndexError` or silently skip the correction.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT direct-events dependencies include the WINCORR lookup

  Background:
    Given a ProcessingInputCollection with a HIT L1A direct-events science file
    And the cosine, gain, charge-fit and event-type ancillaries HIT direct events already require

  Scenario: The WINCORR ancillary is downloaded, parsed and exposed on the dependencies
    Given the collection also contains "imap_hit_wincorr_20100105_v001.csv"
    When HitL3PhaDependencies.fetch_dependencies is called
    Then the WINCORR file is downloaded from the HIT ancillary directory after the other ancillaries
    And WindowCorrectionLookupTable.from_file is called with the downloaded WINCORR path
    And window_correction_lookup on the returned dependencies is the parsed table
    And every other dependency field is unchanged

  Scenario: Fetching fails clearly when the WINCORR ancillary is missing
    Given the collection does not contain a "wincorr" ancillary
    When HitL3PhaDependencies.fetch_dependencies is called
    Then a ValueError "Missing wincorr dependency." is raised
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/pha/hit_l3_pha_dependencies.py`: add `HIT_L3_WINCORR_LOOKUP_DESCRIPTOR = "wincorr"`, the `window_correction_lookup: WindowCorrectionLookupTable` field, and the fetch, missing check and parse in `fetch_dependencies`.

### Tests

- `tests/hit/l3/pha/test_l3_pha_dependencies.py`
- `tests/hit/l3/test_hit_processor.py`: only if it builds `HitL3PhaDependencies(...)` directly. It currently mocks `fetch_dependencies`, so a new required field is not expected to break it, but check.
- `run_local.py`: constructs `HitL3PhaDependencies(...)` directly in `create_hit_direct_event_cdf`. It is not a test, but it will stop working once the field is required; it is updated in Phase 5.

## TDD steps

### 2.1 Red — Scenario: The WINCORR ancillary is downloaded, parsed and exposed on the dependencies

- In `tests/hit/l3/pha/test_l3_pha_dependencies.py`:
  - Add a `MODULE = "imap_l3_processing.hit.l3.pha.hit_l3_pha_dependencies"` constant. Convert this test's `@patch` targets to `f"{MODULE}...."`; the rule calls for it, and it is the test being edited.
  - Extend `test_fetch_dependencies`:
    - Add `@patch(f"{MODULE}.WindowCorrectionLookupTable.from_file")`.
    - Append `sentinel.wincorr_lookup_path` as the **last** entry of `mock_download.side_effect`.
    - Append `f"{mission}_{instrument}_{HIT_L3_WINCORR_LOOKUP_DESCRIPTOR}_{start_date}_{version}.csv"` as the last entry of `file_paths`, which makes the existing `mock_download.assert_has_calls` assert the new download and its order.
    - Assert `mock_wincorr_from_file.assert_called_once_with(sentinel.wincorr_lookup_path)`.
    - Assert `self.assertEqual(mock_wincorr_from_file.return_value, hit_l3_pha_dependencies.window_correction_lookup)`. Keep every existing field assertion so the whole result is still checked.
  - Import `HIT_L3_WINCORR_LOOKUP_DESCRIPTOR`.
- Fails: `ImportError` on the descriptor, then the missing field / download.

### 2.2 Green

- In `hit_l3_pha_dependencies.py`:
  - Add `HIT_L3_WINCORR_LOOKUP_DESCRIPTOR = "wincorr"`.
  - Add the field `window_correction_lookup: WindowCorrectionLookupTable` to the dataclass.
  - After the event-type download, fetch `dependencies.get_file_paths(source="hit", descriptor=HIT_L3_WINCORR_LOOKUP_DESCRIPTOR)`, download its first entry, build `WindowCorrectionLookupTable.from_file(path)`, and pass it to `cls(...)`.

### 2.3 Red — Scenario: Fetching fails clearly when the WINCORR ancillary is missing

- Add `test_fetch_dependencies_raises_when_wincorr_dependency_missing`. Build a `ProcessingInputCollection` containing the L1A science input and every existing ancillary but no `wincorr`. Patch `download`, `HitL1Data.read_from_cdf` and the lookup builders with `MODULE` targets so no real file I/O happens.
- `with self.assertRaises(ValueError) as context: HitL3PhaDependencies.fetch_dependencies(collection)`, then `self.assertEqual(f"Missing {HIT_L3_WINCORR_LOOKUP_DESCRIPTOR} dependency.", str(context.exception))`.
- Fails: `IndexError` from `[0]` on the empty list.

### 2.4 Green

- Check for the missing case before indexing: if `get_file_paths(...)` for `wincorr` is empty, `raise ValueError(f"Missing {HIT_L3_WINCORR_LOOKUP_DESCRIPTOR} dependency.")`.

### 2.5 Refactor

- If the missing check reads cleaner as a small helper, e.g. `_download_required_ancillary(dependencies: ProcessingInputCollection, descriptor: str) -> Path`, extract it and use it **only for `wincorr`** in this card. Moving the other 15 ancillaries onto it would change their failure from `IndexError` to `ValueError`. That is a behavior change without its own tests, so offer it to the user as a follow-up rather than doing it silently.
- Remove the `sentinel.*_cdf_path` naming confusion only if it is touched (the ancillaries are CSVs). Otherwise leave it.

## Done criteria

- [ ] Scenario "downloaded, parsed and exposed" passes: the download order includes WINCORR last, `from_file` is called with the downloaded path, `window_correction_lookup` is set, and all existing field assertions still hold
- [ ] Scenario "missing WINCORR" passes with the exact message `Missing wincorr dependency.`
- [ ] `test_l3_pha_dependencies.py` uses a `MODULE` constant for its patch targets
- [ ] `uv run python -m unittest discover -s tests/hit -t .` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
