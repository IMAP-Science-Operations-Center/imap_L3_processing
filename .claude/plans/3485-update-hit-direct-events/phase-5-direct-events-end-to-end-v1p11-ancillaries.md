# Phase 5 — HIT L3 direct-events product is produced from the v1.11.05 ancillary set

> **Depends on:** Phase 2 (the `wincorr` ancillary must be fetched), Phase 3 (the new energy calculation is exercised), Phase 4 (the v001 Range 2 charge-fit tracks are not in Z order inside the old bounds, which would trip the old `assert`)
> **Blocks:** none
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — end-to-end runs against real files belong in `tests/integration/`
> - [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — verify fixtures before relying on them
> - [`overview.md`](overview.md) — decision 4 (charge-fit v001 from doc Tables 46–48), open issue on the Table 46 Si row, PENA?/PENB? already NOCALC in deployed v003
> - [`phase-2-fetches-wincorr-dependency.md`](phase-2-fetches-wincorr-dependency.md), [`phase-3-incident-energy-window-correction.md`](phase-3-incident-energy-window-correction.md), [`phase-4-charge-without-energy-bounds.md`](phase-4-charge-without-energy-bounds.md)
> - `tests/integration/integration_test_helpers.py` — `mock_imap_data_access`
> - HIT Algorithm Document v1.11.05 Tables 46, 47, 48

## Goal

The repo carries the full v1.11.05 ancillary set as fixtures:

- WINCORR v000 (added in Phase 1)
- the six cosine v001 files and lo-gain 2026-05-14 v001 from issue #69
- the deployed event-type v003, where PENA?/PENB? are NOCALC
- six charge-fit v001 files built from doc Tables 46–48

A new integration test runs `imap_l3_data_processor` for `hit` / `l3` / `direct-events` against a real L1A direct-events CDF and that ancillary set, and the L3 CDF is written. The developer scripts `run_local.py` and `run_local_with_upload_and_download.py` build direct events from the same set, including WINCORR.

This is the card's acceptance test. The unit tests in Phases 2–3 mock file loading, so this phase is the only check that real files go through fetch → parse → process → save.

## Gherkin Feature & Scenarios

```gherkin
Feature: HIT L3 direct-events product is produced from the v1.11.05 ancillary set

  Background:
    Given the L1A direct-events file "imap_hit_l1a_direct-events_20100105_v009.cdf"
    And the ancillaries
      | file                                                   |
      | imap_hit_wincorr_20250901_v000.csv                     |
      | imap_hit_range-2A-cosine-lookup_20250203_v001.csv      |
      | imap_hit_range-3A-cosine-lookup_20250203_v001.csv      |
      | imap_hit_range-4A-cosine-lookup_20250203_v001.csv      |
      | imap_hit_range-2B-cosine-lookup_20250203_v001.csv      |
      | imap_hit_range-3B-cosine-lookup_20250203_v001.csv      |
      | imap_hit_range-4B-cosine-lookup_20250203_v001.csv      |
      | imap_hit_lo-gain-lookup_20260514_v001.csv              |
      | imap_hit_hi-gain-lookup_20250203_v000.csv              |
      | imap_hit_range-2A-charge-fit-lookup_20250319_v001.csv  |
      | imap_hit_range-3A-charge-fit-lookup_20250319_v001.csv  |
      | imap_hit_range-4A-charge-fit-lookup_20250319_v001.csv  |
      | imap_hit_range-2B-charge-fit-lookup_20250319_v001.csv  |
      | imap_hit_range-3B-charge-fit-lookup_20250319_v001.csv  |
      | imap_hit_range-4B-charge-fit-lookup_20250319_v001.csv  |
      | imap_hit_hit-event-type-lookup_20250228_v003.csv       |

  Scenario: The direct-events L3 CDF is written from the v1.11.05 ancillaries
    When imap_l3_processor runs with instrument "hit", data level "l3", descriptor "direct-events", start date "20100105" and version "v001"
    Then "imap_hit_l3_direct-events_20100105_v001.cdf" exists at its canonical science path
```

## Affected files

### Production

- `run_local.py`, `create_hit_direct_event_cdf` (developer script, no tests):
  - Build `WindowCorrectionLookupTable.from_file(...)` from the WINCORR fixture and pass it as `window_correction_lookup`.
  - Switch to the v001 cosine / lo-gain / charge-fit fixtures and the v003 event-type fixture.
  - **Fix the existing argument-order bug.** `CosineCorrectionLookupTable(...)` is called with 2A, 2B, 3A, 3B, 4A, 4B, but the constructor order is 2A, 3A, 4A, 2B, 3B, 4B. The v001 A and B files are identical, so the bug has no numeric effect today, but it is still wrong.
- `run_local_with_upload_and_download.py`, the `"hit", "direct-events"` dependency JSON: add `imap_hit_wincorr_20250901_v000.csv` and update the cosine, lo-gain, charge-fit and event-type file names to the versions above.

### Tests

- `tests/integration/test_hit_processor_integration.py`: new `test_hit_direct_events_product`.
- New fixtures in `tests/test_data/hit/pha_events/`:
  - `imap_hit_range-{2A,2B,3A,3B,4A,4B}-cosine-lookup_20250203_v001.csv`: byte-for-byte copies of the issue #69 attachments (downloaded during planning to `C:\Users\Harrison\AppData\Local\Temp\claude\0cb615bc458e791d\399e7e02-6f31-41f6-84ff-3006ee73f25d\scratchpad\issue69_attachments\`; re-download from the issue if cleared)
  - `imap_hit_lo-gain-lookup_20260514_v001.csv`: issue #69 attachment
  - `imap_hit_hit-event-type-lookup_20250228_v003.csv`: copy of the deployed SDC file cached at `data/imap/ancillary/hit/imap_hit_hit-event-type-lookup_20250228_v003.csv` (rows `PENA?` and `PENB?` have Range `NOCALC`)
  - `imap_hit_range-{2A,2B,3A,3B,4A,4B}-charge-fit-lookup_20250319_v001.csv`: built from doc Tables 46–48, contents below

## Charge-fit v001 file contents

The format matches v000: `#` header lines, then `Z,a1,b1,a2,b2,gamma,` with a trailing comma. The A and B sides are identical per range, as in v000. The values were extracted with `pdftotext` from `HIT_Algorithm_Document-v1p11p05.pdf`, and the Table 46 Si row was re-checked with `pdftotext -raw`. **Before uploading to the SDC, re-verify against the PDF and get HIT to confirm the Table 46 Si (Z=14) row** (see overview open issues).

`imap_hit_range-2A-charge-fit-lookup_20250319_v001.csv` and `imap_hit_range-2B-charge-fit-lookup_20250319_v001.csv`:

```
# From HIT Algorithm Document v1.11.05 (2026-10-19) Table 46, Range2 (A14/B14 apertures)
# Format: ion_charge, a1, b1, a2, b2, gamma
1,1.479963e+00,8.209081e-02,1.056083e+00,-6.688255e-01,-1.759770e+00,
2,4.964089e+00,-2.347052e-02,1.119467e+01,-6.892635e-01,-2.222412e+00,
6,2.587211e+01,1.218594e-02,4.106214e+03,-1.256034e+00,-8.434640e-01,
7,3.208165e+01,7.707944e-03,2.944465e+03,-1.113833e+00,-1.038501e+00,
8,3.863437e+01,9.539366e-03,8.653277e+03,-1.227574e+00,-9.272040e-01,
10,5.128670e+01,1.664116e-02,4.316894e+03,-9.913076e-01,-1.239285e+00,
11,5.800006e+01,1.720292e-02,4.251641e+03,-9.382665e-01,-1.389592e+00,
12,6.463724e+01,1.945454e-02,6.811090e+03,-9.786677e-01,-1.283539e+00,
13,7.101556e+01,2.499757e-02,5.451054e+03,-9.004415e-01,-1.434989e+00,
14,3.307547e+02,-1.016750e-01,2.434280e+04,-3.028693e-09,-2.299653e-01,
16,9.348524e+01,4.418190e-03,7.976853e+02,-4.986653e-01,-4.782615e+00,
18,1.085192e+02,7.100202e-03,9.061990e+02,-4.675327e-01,-5.881363e+00,
20,1.212440e+02,1.199842e-02,9.543358e+02,-4.481641e-01,-5.864268e+00,
26,1.634267e+02,1.807242e-02,1.057107e+03,-3.715106e-01,-1.066283e+01,
28,1.779868e+02,1.780692e-02,1.133637e+03,-3.632466e-01,-1.176421e+01,
```

`imap_hit_range-3A-charge-fit-lookup_20250319_v001.csv` and `imap_hit_range-3B-charge-fit-lookup_20250319_v001.csv`:

```
# From HIT Algorithm Document v1.11.05 (2026-10-19) Table 47, Range3 (A14/B14 apertures)
# Format: ion_charge, a1, b1, a2, b2, gamma
1,1.453953e+00,-3.433355e-01,2.174537e+00,-7.499576e-01,-6.333077e+00,
2,7.645239e+00,-5.772561e-02,2.961728e+01,-7.905325e-01,-1.967359e+00,
6,4.251565e+01,-8.075919e-03,8.445316e+02,-8.396609e-01,-1.579686e+00,
7,5.331862e+01,-3.690031e-03,1.454459e+03,-8.563961e-01,-1.484515e+00,
8,6.532380e+01,-1.759982e-02,1.506928e+03,-8.017355e-01,-1.839473e+00,
10,8.841751e+01,-4.672032e-03,2.680709e+03,-7.942925e-01,-1.802791e+00,
11,1.022556e+02,3.122387e-03,5.270435e+03,-8.482708e-01,-1.466593e+00,
12,1.138565e+02,1.648505e-03,5.199846e+03,-8.158559e-01,-1.603033e+00,
13,1.267025e+02,1.216113e-02,6.798835e+03,-8.160702e-01,-1.497994e+00,
14,1.394042e+02,7.434322e-03,8.372908e+03,-8.202076e-01,-1.495869e+00,
16,1.666386e+02,4.861300e-03,8.498519e+03,-7.709579e-01,-1.683556e+00,
18,2.004655e+02,1.796604e-03,9.676462e+03,-7.350743e-01,-1.893294e+00,
20,2.204714e+02,6.721033e-03,9.789824e+03,-7.082093e-01,-1.971441e+00,
26,3.133529e+02,6.018827e-03,1.434066e+04,-6.626533e-01,-2.181724e+00,
28,3.368751e+02,1.208747e-02,1.826999e+04,-6.730212e-01,-2.002748e+00,
```

`imap_hit_range-4A-charge-fit-lookup_20250319_v001.csv` and `imap_hit_range-4B-charge-fit-lookup_20250319_v001.csv`:

```
# From HIT Algorithm Document v1.11.05 (2026-10-19) Table 48, Range4 (A14/B14 apertures)
# Format: ion_charge, a1, b1, a2, b2, gamma
1,1.154340e+01,-7.114543e-02,1.011661e+02,-1.052054e+00,-1.717067e+00,
2,4.811216e+01,-1.103459e-02,6.323176e+02,-7.755388e-01,-1.607142e+00,
6,2.676431e+02,-4.494929e-04,1.493469e+04,-7.874666e-01,-1.444953e+00,
7,3.437643e+02,-6.705931e-03,1.568531e+04,-7.352394e-01,-1.666771e+00,
8,4.254100e+02,-1.157973e-02,7.921619e+03,-5.849140e-01,-2.449569e+00,
10,5.902855e+02,-1.705847e-03,1.939047e+04,-6.267098e-01,-1.954604e+00,
11,6.974519e+02,-2.227489e-03,3.906509e+04,-6.850924e-01,-1.737538e+00,
12,7.861641e+02,-4.283877e-03,2.107218e+04,-5.731322e-01,-2.201842e+00,
13,8.908970e+02,2.216163e-03,8.528126e+04,-7.245672e-01,-1.523295e+00,
14,9.850535e+02,-1.797150e-03,7.608972e+04,-6.950917e-01,-1.738658e+00,
16,1.232022e+03,-7.479155e-03,3.545218e+04,-5.505602e-01,-2.394372e+00,
18,1.500582e+03,2.210960e-03,1.567151e+05,-6.830830e-01,-1.572172e+00,
20,1.717848e+03,-5.895197e-03,1.173455e+05,-6.349220e-01,-1.998707e+00,
26,2.592495e+03,-5.837627e-04,4.148593e+05,-6.875006e-01,-1.597780e+00,
28,2.864045e+03,-3.316593e-03,1.538460e+05,-5.585982e-01,-2.167948e+00,
```

## TDD steps

### 5.1 Red — Scenario: The direct-events L3 CDF is written from the v1.11.05 ancillaries

- Add the fixtures listed under **Affected files → Tests**.
- Before writing the test, check the fixtures: `RangeFitLookup.from_files(...)` loads each v001 charge-fit file into a `(15, 6)` array, and `CosineCorrectionLookupTable(...)` and `GainLookupTable.from_file(...)` load the v001 files. Do this as a one-off check in a REPL, not as new unit tests; those would only pin fixture contents.
- In `tests/integration/test_hit_processor_integration.py`, add `test_hit_direct_events_product`, modelled on `test_hit_macropixel_product`:
  - `input_files` = `get_test_data_path("hit/pha_events/<name>")` for the L1A file and every ancillary in the Background table.
  - `with mock_imap_data_access(get_run_local_data_path("hit_direct_events_integration"), input_files):` build `ProcessingInputCollection(*(generate_imap_input(f.name) for f in input_files))`. Set `mock_arguments` to `instrument="hit"`, `data_level="l3"`, `descriptor="direct-events"`, `start_date="20100105"`, `end_date=None`, `repointing=None`, `version="v001"`, `dependency=collection.serialize()`, `upload_to_sdc=False`, then call `imap_l3_data_processor.imap_l3_processor()`.
  - `self.assertTrue(ScienceFilePath("imap_hit_l3_direct-events_20100105_v001.cdf").construct_path().exists(), ...)`.
- Use **v009** of the L1A file. `imap_hit_l1a_direct-events_20100105_v010.cdf` fails to open with spacepy (`NOT_A_CDF_OR_NOT_SUPPORTED`), while v009 reads 61 epochs.
- Expected result when this is run before Phases 2–4 land: errors, either from the old `assert` in `compute_charge` on a v001 Range 2 event or from a missing `window_correction_lookup`. After Phases 2–4 it should pass. If it fails, investigate the real cause rather than adjusting fixtures to make it pass.

### 5.2 Green

- No new production code should be needed beyond Phases 1–4. If the run fails, trace the root cause, for example a fixture parse issue, the CDF attribute YAML, or a processor path, and fix it with a focused unit test in the relevant module first.
- Update `run_local.py::create_hit_direct_event_cdf` and `run_local_with_upload_and_download.py` as described under **Affected files → Production**, including the cosine argument-order fix. Run `create_hit_direct_event_cdf` once by hand if the referenced `data/imap/hit/l1a/...` input exists locally. If it does not, say so in the review summary rather than skipping silently.

### 5.3 Refactor

- If `test_hit_direct_events_product` and `test_hit_macropixel_product` now share most of their setup, such as the CLI-argument mock and the logging config, extract a small helper in the test module.
- Leave the v000 fixtures in place; existing unit tests still reference them.

## Done criteria

- [ ] All new fixtures are checked in; the cosine v001, lo-gain v001 and WINCORR files are byte-identical to the issue #69 attachments, and event-type v003 is byte-identical to the deployed copy
- [ ] The charge-fit v001 files match Tables 46–48 as listed above, with A side == B side
- [ ] Scenario "direct-events L3 CDF is written" passes: `uv run python -m unittest tests.integration.test_hit_processor_integration`
- [ ] `run_local.py` and `run_local_with_upload_and_download.py` reference the v1.11.05 set, include WINCORR, and the cosine argument order is fixed
- [ ] The full suite passes: `uv run python -m unittest discover tests`, with the Docker-only GLOWS tests skipped if Docker is not running
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] Review summary reminds the user of the open HIT questions: the Table 46 Si row, whether lo-gain v001 really differs from v000, and WINCORR v000 vs v001 naming
