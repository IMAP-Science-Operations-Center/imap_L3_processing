# Phase 1 — HIT WINCORR ancillary provides per-range detector window correction factors

> **Depends on:** none
> **Blocks:** Phase 2 (the dependency loader builds this table), Phase 3 (the incident energy sum queries it)
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — test quality rules
> - [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — fail explicitly on malformed input, type annotations, imports at top
> - [`.claude/rules/no-comments.md`](../../rules/no-comments.md) — no code comments
> - [`overview.md`](overview.md) — WINCORR layout assumption (row 1 = WINCORR2, row 2 = WINCORR3)
> - HIT Algorithm Document v1.11.05 §9.1.1, Table 40 (WINCORR2 / WINCORR3 arrays), Table 8 (detector addresses)

## Goal

The HIT team ships the flight-software window (foil) correction arrays as an ancillary file, `imap_hit_wincorr_<date>_v<NNN>.csv`. After `#` comment lines it holds two comma-separated rows of 64 integers indexed by detector address: WINCORR2 for Range 2 events and WINCORR3 for Range 3 and Range 4 events. Each entry is `256 × corr`. This phase adds a reader that turns that file into a lookup returning the correction factor `k = WINCORR[address] / 256` for a detected range and a detector address. A file that does not have exactly two rows of 64 values raises a `ValueError` that names the file.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT WINCORR ancillary provides per-range detector window correction factors

  Background:
    Given the WINCORR ancillary file "imap_hit_wincorr_20250901_v000.csv"
    And it begins with a "#" provenance comment line
    And its first data row is WINCORR2 and its second data row is WINCORR3, each 64 integers indexed by detector address

  Scenario Outline: The correction factor is the range's WINCORR entry for the detector address divided by 256
    When WindowCorrectionLookupTable.from_file reads the file
    And get_correction_factor is called with range <range> and detector address <address>
    Then the returned factor is <wincorr_value> / 256

    Examples:
      | range | address | detector | wincorr_value |
      | R2    | 16      | L1A0a    | 391           |
      | R2    | 19      | L1A1a    | 390           |
      | R2    | 32      | L1B0a    | 391           |
      | R2    | 9       | L2A0     | 256           |
      | R3    | 18      | L1A0c    | 420           |
      | R3    | 35      | L1B1a    | 419           |
      | R4    | 30      | L1A4c    | 419           |
      | R4    | 63      | L3Bo     | 256           |

  Scenario Outline: A WINCORR file without exactly two rows of 64 values is rejected
    Given a WINCORR file whose data rows are <shape>
    When WindowCorrectionLookupTable.from_file reads the file
    Then a ValueError is raised whose message names the file path

    Examples:
      | shape                                  |
      | a single row of 64 values              |
      | three rows of 64 values                |
      | two rows where the second has 63 values |
```

## Affected files

### Production

- `imap_l3_processing/hit/l3/pha/science/window_correction_lookup_table.py` *(new)*: `WindowCorrectionLookupTable`

### Tests

- `tests/hit/l3/pha/science/test_window_correction_lookup_table.py` *(new)*
- `tests/test_data/hit/pha_events/imap_hit_wincorr_20250901_v000.csv` *(new fixture)*: a byte-for-byte copy of the issue #69 attachment, including its `# From WINCORR_tables_20260630.dat, A. Bruno email 2026-09-10` line. It was downloaded during planning to `C:\Users\Harrison\AppData\Local\Temp\claude\0cb615bc458e791d\399e7e02-6f31-41f6-84ff-3006ee73f25d\scratchpad\issue69_attachments\imap_hit_wincorr_20250901_v000.csv`; re-download from the issue if the scratchpad has been cleared.

## TDD steps

### 1.1 Red — Scenario: The correction factor is the range's WINCORR entry for the detector address divided by 256

- Copy the WINCORR attachment into `tests/test_data/hit/pha_events/imap_hit_wincorr_20250901_v000.csv`.
- In `tests/hit/l3/pha/science/test_window_correction_lookup_table.py`, write `test_returns_wincorr_entry_over_256_for_range_and_detector_address`:
  - Load the fixture with `WindowCorrectionLookupTable.from_file(get_test_data_path("hit/pha_events/imap_hit_wincorr_20250901_v000.csv"))`.
  - Loop over the Examples table with `self.subTest(range=..., address=...)` and `self.assertEqual(wincorr_value / 256, table.get_correction_factor(DetectorRange.<range>, address))`. The values are exact integer ratios, so exact equality is correct.
  - The R3 and R4 rows prove that Range 3 and Range 4 share the second row. The R2 rows with 391 versus the R3 row with 420 prove the rows are not swapped. The 256 rows prove that non-L1 addresses give `k = 1`.
- Fails: the module does not exist.

### 1.2 Green

- Create `window_correction_lookup_table.py` with:
  - `class WindowCorrectionLookupTable` whose annotated members are `_range_2_factors: np.ndarray` and `_range_3_and_4_factors: np.ndarray` (float arrays of length 64, already divided by 256).
  - `__init__(self, range_2_wincorr: np.ndarray, range_3_and_4_wincorr: np.ndarray) -> None`
  - `@classmethod from_file(cls, path: Path) -> "WindowCorrectionLookupTable"`: read with `np.loadtxt(path, delimiter=",", comments="#", ndmin=2)` (whitespace after commas is accepted) and build from row 0 and row 1.
  - `get_correction_factor(self, detector_range: DetectorRange, detector_address: int) -> float`: R2 uses `_range_2_factors`; R3 and R4 use `_range_3_and_4_factors`.
- Import `DetectorRange` from `cosine_correction_lookup_table`, where it is defined today.

### 1.3 Red — Scenario: A WINCORR file without exactly two rows of 64 values is rejected

- Add `test_rejects_wincorr_file_without_two_rows_of_64_values` in the same test file. Write each malformed file into a temporary directory with `tempfile.TemporaryDirectory()` (or `TempFileTestCase` from `tests/temp_file_test_case.py`). Each file gets a `#` comment line followed by:
  - one row of 64 values,
  - three rows of 64 values,
  - two rows where the second has 63 values.
- For each case, use `subTest` and `with self.assertRaises(ValueError) as context:` around `from_file`, then `self.assertIn(str(path), str(context.exception))`.
- Fails: one row → `IndexError`; three rows → silently accepted; 63 values → `np.loadtxt` raises `ValueError` but without the file path.

### 1.4 Green

- In `from_file`, wrap the `np.loadtxt` call and validate the result. If loading fails, or `wincorr.shape != (2, 64)`, raise `ValueError(f"Expected 2 rows of 64 WINCORR values in {path}, got ...")`. When re-raising a `np.loadtxt` error, chain it with `from`.

### 1.5 Refactor

- Name the magic numbers as module constants, e.g. `WINCORR_SCALE = 256` and `DETECTOR_ADDRESS_COUNT = 64`.
- Confirm the class, method signatures and members are fully type-annotated and that there are no comments.

## Done criteria

- [ ] `tests/test_data/hit/pha_events/imap_hit_wincorr_20250901_v000.csv` is checked in and identical to the issue #69 attachment
- [ ] Scenario "correction factor … divided by 256" passes for all 8 Examples rows (`test_returns_wincorr_entry_over_256_for_range_and_detector_address`)
- [ ] Scenario "without exactly two rows of 64 values is rejected" passes for all 3 shapes, and each message includes the file path
- [ ] `uv run python -m unittest tests.hit.l3.pha.science.test_window_correction_lookup_table` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
