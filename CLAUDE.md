# IMAP L3 Processing

`imap_l3_processing` produces Level 3 science data products for the IMAP (Interstellar Mapping and Acceleration Probe) mission. It is a command-line program, run inside a Docker container by the IMAP Science Data Center (SDC) batch pipeline, that reads L1/L2 (and some L3) CDF dependencies plus ancillary calibration files and SPICE kernels, runs instrument-specific science algorithms, and writes ISTP-compliant L3 CDF files that are optionally uploaded back to the SDC.

- **Language / runtime:** Python `>= 3.10` (CI and the Docker image use `3.12`), managed with [`uv`](https://docs.astral.sh/uv/).
- **Science stack:** numpy, scipy, xarray, astropy / astropy_healpix, lmfit, uncertainties, numba, spacepy (CDF I/O), spiceypy (SPICE geometry), `imap-data-access` (SDC queries, downloads, uploads), `imap-processing`.
- **Version:** dynamic, derived from git tags via `uv-dynamic-versioning` (see `pyproject.toml`).

## Table of contents (delegated rules)

Detailed conventions live in dedicated always-applied rules under `.claude/rules/` (loaded automatically by Claude Code) — follow them rather than duplicating here:

- [Agent guidelines](.claude/rules/agent-guidelines.md) — working principles, git safety, error handling, debugging, tool use, communication.
- [Test quality & coverage](.claude/rules/test-quality-coverage.md) — behavioral testing, numpy/CDF assertions, coverage expectations, patch cleanup, anti-patterns.
- [No comments](.claude/rules/no-comments.md) — the code must be self-documenting; do not add code comments.
- [Local environment](.claude/rules/local-environment.md) — macOS vs Windows command conventions, chosen from the current OS.
- [Development workflow](.claude/README.md) — the card → phased plan → TDD → review loop.

## This is a Test-Driven Development project

Production code is written **test-first** using strict Red → Green → Refactor cycles:

1. **Red** — write a failing test that describes the desired behavior.
2. **Green** — write the minimum production code to make it pass.
3. **Refactor** — clean up while keeping tests green.

Every change to production code should be driven by, and covered by, a test. See the [test quality & coverage rule](.claude/rules/test-quality-coverage.md) for how tests must be structured, and the [development workflow](.claude/README.md) for the end-to-end phased-plan/TDD process.

## Architecture

- **Entry point:** `imap_l3_data_processor.py` → `imap_l3_processor()` parses the SDC CLI arguments (`--instrument`, `--data-level`, `--descriptor`, `--start-date`, `--end-date`, `--repointing`, `--version`, `--dependency`, `--upload-to-sdc`), deserializes the dependency JSON into an `imap_data_access` `ProcessingInputCollection`, furnishes SPICE kernels, builds an `InputMetadata`, and dispatches to the instrument processor (or, for Hi/Lo/Ultra map descriptors, to a map initializer that decides which maps to produce).
- **Instrument packages** under `imap_l3_processing/`: `codice`, `glows`, `hi`, `hit`, `lo`, `swapi`, `swe`, `ultra`, plus shared `maps` (Hi/Lo/Ultra sky-map products) and `cdf` (CDF writing and attribute management).
- **Processor pattern:** each instrument has a `*Processor` subclass of `imap_l3_processing.processor.Processor` whose `process() -> list[Path]` produces output files. Inputs are gathered by `*Dependencies` dataclasses with a `fetch_dependencies(ProcessingInputCollection)` classmethod that downloads and reads the required files. Science algorithms live in `science/` subpackages; output data is modeled as `DataProduct` subclasses (`imap_l3_processing/models.py`) and written with `save_data` (`imap_l3_processing/utils.py`).
- **CDF metadata:** global and variable attributes come from YAML files in `imap_l3_processing/cdf/config/` (e.g. `imap_swapi_l3a_global_cdf_attrs.yaml`), loaded by `ImapAttributeManager`. New or renamed output variables need matching YAML entries.
- **Quality flags:** per-instrument `quality_flags.py` modules define flag enums built on `FlagNameMixin` (e.g. `SweL3Flags`, `SwapiL3Flags`, `MapL3Flags`) that are set on output products.
- **SPICE:** `furnish_local_spice` and `furnish_spice_metakernel` in `imap_l3_processing/utils.py` manage kernels; `PredictedEphemerisTracker` in `imap_l3_processing/predicted_ephemeris_tracker.py` detects when a lookup only succeeds with predicted-ephemeris kernels.

## Running the tests (with `uv`)

Tests use the standard-library `unittest` framework and are run with `unittest`. Many fixtures (calibration `.dat` files, SPICE kernels) are stored in **Git LFS** — run `git lfs pull` first, or tests fail with confusing parse/SPICE errors.

```bash
# Run the full suite
uv run python -m unittest discover tests

# Run one instrument's unit tests
uv run python -m unittest discover -s tests/swe -t .

# Run one instrument's integration test
uv run python -m unittest tests.integration.test_swe_processor_integration

# Run a single test module, case, or method
uv run python -m unittest tests.swe.test_swe_processor
uv run python -m unittest tests.swe.test_swe_processor.TestSweProcessor
uv run python -m unittest tests.swe.test_swe_processor.TestSweProcessor.test_some_behavior
```

Tests live under `tests/`, mirroring the `imap_l3_processing/` package layout (e.g. `imap_l3_processing/swe/swe_processor.py` → `tests/swe/test_swe_processor.py`). Integration tests that run a full processor end-to-end against real data live in `tests/integration/`. When discovering a subdirectory, pass `-t .` so test modules import as `tests.<instrument>...`.

The tests decorated with `@run_test_in_docker` in `tests/integration/test_glows_processor_integration.py` (`test_l3bcde_first_time_processing`, `test_l3bcde_automatic_reprocessing`, `test_glows_l3abcde_from_l2`, `test_run_glows_l3be_against_prod`) build and run `Dockerfile_glows_integration` and need a running Docker daemon. If Docker is not available, skip them:

```bash
uv run pytest -k "not test_l3bcde_first_time_processing and not test_l3bcde_automatic_reprocessing and not test_glows_l3abcde_from_l2 and not test_run_glows_l3be_against_prod"
```

A `failed to connect to the docker API` error from these tests means Docker is not running. Leave them skipped.

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format --check .
```

`pre-commit` runs both on staged files (line length 99, target `py310`).

## Running the processor locally

```bash
# Through Docker, matching the SDC batch job
./run_local_using_docker.sh {instrument-name} {data-level}

# Directly, with explicit SDC-style arguments
uv run python imap_l3_data_processor.py --instrument <instrument> --data-level <level> --descriptor <descriptor> --start-date <YYYYMMDD> --version <vNNN> --dependency '<dependency JSON or path to .json>'
```

`run_local.py` contains helpers for building specific products from local data without going through the SDC.

## Test infrastructure

Read these before writing new tests so you reuse existing helpers instead of reinventing them.

### `tests/test_helpers.py`

- **Test-data paths:** `get_test_data_path`, `get_test_data_folder`, `get_integration_test_data_path`, `get_spice_data_path`, `get_integration_test_spice_data_path`, `get_test_instrument_team_data_path`, `get_run_local_data_path` (files under `run_local_input_data/`).
- **Assertions and matchers:** `NumpyArrayMatcher` (numpy-aware argument matching in `assert_called_with`), `assert_dict_close`, `assert_dataclass_fields`.
- **Mocks and fakes:** `create_dataclass_mock`, `create_mock_query_results` and `create_mock_version_map` (for `imap_data_access` queries), `environment_variables` decorator.
- **Builders:** SWE-specific builders such as `build_swe_configuration`, `build_moments`, `build_moment_fit_results`, `build_swe_moment_data`.
- **Utilities:** `with_tempdir`, `run_periodically` (tests that only run on a schedule).

### Base test cases

- `tests/spice_test_case.py` — `SpiceTestCase` furnishes the local SPICE kernels in `setUpClass` and clears them in `tearDownClass`.
- `tests/temp_file_test_case.py` — `TempFileTestCase` creates and removes `tests/temp_test_files/` around each test.

### Builders and integration helpers

- `tests/maps/test_builders.py` — factories for map domain objects (`create_intensity_map_data`, rectangular/HEALPix map data, pointing sets, GLOWS L3e inputs).
- `tests/integration/integration_test_helpers.py` — `mock_imap_data_access`, `ImapQueryPatcher`, `RequestsGetPatcher`, `stage_input_file`, `create_metakernel`, and `run_istp_compliance_check` for end-to-end processor tests.
