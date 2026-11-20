---
name: implement-tdd
description: >-
  Implement a feature using strict Red-Green-Refactor TDD cycles driven by a
  provided plan or instructions. Use when the user asks to execute a plan
  phase, implement TDD steps, or build feature code with a test-first
  approach driven by a provided plan file.
disable-model-invocation: false
---

# Implement TDD

Implement a feature using strict Red-Green-Refactor TDD cycles driven by provided instructions. The user will provide implementation instructions or a plan that specifies what to build and which tests to write. Follow the phases below to execute those instructions using TDD.

## When to Use

- User explicitly invokes this skill via `/implement-tdd`
- A plan file under `.cursor/plans/` is ready to be executed
- You are Step 3 (planning) or Step 4 (building) of the standard IMAP L3 processing workflow (see `.cursor/README.md`)

**Do NOT use for:**
- Ad-hoc code changes without a plan
- Starting work without a clear list of TDD steps to follow

---

## Phase 1: Analyze the instructions and existing code

1. **Read the instructions** provided by the user (inline text, a plan file, or a linked document). Identify:
   - Which production files will change and what behavior is being added/modified.
   - Which test files correspond to the production files.
   - The TDD steps and their order (Red steps, Green steps, Refactor steps).

2. **Read every existing test file** that will receive new tests — in full, not just the diff. For each planned test case, determine:
   - **Can it be folded into an existing test?** If an existing test already exercises the same setup or a superset scenario (e.g. an existing round-trip test that already uses the relevant enum value), add the new assertions there instead of creating a separate test function. Prefer this whenever the existing test's scope naturally covers the new behavior.
   - **Can planned tests be consolidated?** If two or more planned tests assert different aspects of the same behavior (e.g. visibility for two enum values of the same category), merge them into a single test. In particular, when multiple tests share the same setup and exercise sequential steps of one scenario (e.g. initial state → add → remove → boundary enforcement), combine them into a single **lifecycle test** that walks through the full sequence, asserting at each step.
   - **Is the test genuinely new?** Only create a new test function when the behavior under test is distinct from everything already in the file.

   **Prefer behavioral/functional tests.** Every test — whether new or extended — must be resilient to implementation changes. The test should fail if the production code it covers is commented out or removed. If a test can still pass after the relevant production line is deleted, it is not testing real behavior and must be rewritten. Concretely:
   - Assert on observable outcomes (return values, output arrays, quality flags, data product variables, written CDF contents, output paths), not on internal bookkeeping.
   - For processor tests, drive behavior through `process()` or the public science function and assert on what it produces — never rely solely on reading back a value that was just written to an internal field.

3. **Produce a test plan** before writing any code. For each Red step, state:
   - The test file and whether it is a new test function or additions to an existing one.
   - The specific assertions that will be added.
   - Brief justification if creating a new function (what distinct behavior it covers).

Present this test plan to the user and wait for confirmation before proceeding.

---

## Phase 2: Execute TDD cycles

Work through the TDD steps from the instructions in order. For each cycle:

### Red

1. Write the test(s) for this step — either new test functions or additions to existing ones, per the confirmed test plan.
2. Run only the affected test file to confirm the new tests **fail** for the expected reason (missing method, missing field, wrong value — not an import error or unrelated crash).
3. If a test fails for the wrong reason, fix the test before proceeding.

### Green

1. Write the **minimum production code** needed to make the failing tests pass.
2. Run the affected test file again to confirm all tests pass (both new and pre-existing).
3. Do not add code that is not required by a failing test.

### Refactor (after each Green, and at the end)

1. Look for duplication, unclear naming, or structural issues in the code you just wrote.
2. Run the full set of affected test files after any refactor to confirm nothing regressed.

---

## Phase 3: Final verification

After all TDD cycles are complete:

1. Run every unit test file that was touched or that corresponds to a changed production file. Skip files under `tests/integration/`.
2. Confirm no regressions — especially tests that assert defaults or round-trip the full config, since new fields with default values must not break them.
3. Run `ReadLints` on all changed production and test files. Fix any lint errors introduced by the changeset.

---

## Test quality rules (always apply)

These rules govern every test written or modified during execution. Violations must be fixed before moving on.

- **Behavior, not implementation**: Tests verify what the code does from the caller's perspective. Do not test private methods directly.
- **Name tests after behaviors**: e.g. `test_returns_empty_list_when_no_data`, not `test_process_data`.
- **No redundancy**: Never duplicate coverage already provided by existing tests. Prefer extending an existing test over creating a new one when the behavior overlaps.
- **Proper granularity**: Each test function verifies a meaningful behavioral unit. Do not split one logical scenario across multiple functions. Do not create a separate test for a trivial assertion that is already covered by a broader test. When separate tests share the same setup and test sequential steps of one scenario (initial state, add, remove, boundary), consolidate them into a single lifecycle test that asserts at each step.
- **Science data testing**:
  - Compare arrays with `np.testing.assert_array_equal` / `assert_allclose`; match array arguments in mock calls with `NumpyArrayMatcher` from `tests.test_helpers`.
  - Include fill values / NaNs in inputs when the code handles them, and assert where they end up in the output.
  - Assert full quality-flag arrays, including the epochs that must not be flagged.
  - When a `DataProduct`'s variables change, assert the full `to_data_product_variables()` output and update the matching YAML in `imap_l3_processing/cdf/config/`.
  - Patch `imap_data_access` and SPICE in unit tests; use `SpiceTestCase` only when real kernels are part of the behavior under test.
- **No defensive programming in tests**: No try/except (except `assertRaises`), no conditional assertions, no overly tolerant assertions, no silent early returns, no optional/fallback logic.
- **Only write tests that drive production changes**:
   * Do not write tests for pinning regressions.
   * No re-pin regression pins, only test living code.

---

## Phase 4: Run the unit test suite

After final verification passes, run the unit tests to catch regressions beyond the directly affected files. Do not run integration tests. Anything under `tests/integration/` is out of scope for this skill, including the Docker GLOWS tests.

1. Run: `uv run pytest tests --ignore=tests/integration`
2. Lint and format with the pre-commit ruff hooks. They apply fixes, and `.pre-commit-config.yaml` excludes `.cursor/` and `.claude/`:
   - `uv run pre-commit run ruff-check --all-files`
   - `uv run pre-commit run ruff-format --all-files`
   If a hook rewrites a file, re-run the affected tests, then run that hook again until it passes. Do not run `ruff` on the whole tree; that also formats Markdown under `.cursor/` and `.claude/`.
3. If any test fails, determine whether it is caused by the changes in this session.
   - If yes, fix the regression and re-run the unit suite with the same command.
   - If no (pre-existing failure), note it for the user and move on.
4. Do not consider the task complete until the unit suite passes (or all failures are confirmed pre-existing).
   **NOTE:** Failures like `ValueError: could not convert string 'version' ...` or `SpiceNOLEAPSECONDS` usually mean Git LFS files are still pointer stubs — run `git lfs pull` rather than changing code.

---

## Phase 5: Document drift in `DRIFT.md` (only if a plan was used)

If the instructions for this session came from a plan directory under `.cursor/plans/<plan-name>/`, record any deviations from the plan in a `DRIFT.md` file inside that directory.

**Skip this phase entirely** when the work was driven by inline instructions or an ad-hoc document with no plan directory — drift is only meaningful when there is a plan to drift from.

1. **Identify deviations** introduced during this session. Common sources:
   - A planned test was folded into an existing test, consolidated with another planned test, or otherwise restructured during Phase 1.
   - A Red/Green/Refactor cycle deviated from the planned files, test names, or assertions.
   - Production code took a different shape than the plan described (different class structure, API, or module location).
   - A planned step was found unnecessary, redundant, or already implemented and was skipped.
   - Work was added mid-cycle that was not in the plan (e.g. an additional refactor or fix that surfaced during Green).

2. **Update `DRIFT.md`** in the plan directory:
   - Path: `.cursor/plans/<plan-name>/DRIFT.md`.
   - If the file does not exist, create it with a `# Drift` heading and a one-line description: `Deviations from the plan, recorded as they happen.`
   - For each deviation, append a numbered entry using this template:

     ```
     ## N. <Short title> (<Phase or step reference>)

     **When:** <YYYY-MM-DD>
     **Original plan:** <what the plan said>
     **What changed:** <what was actually done>
     **Why:** <reasoning>
     ```

   - Keep entries concise — one short paragraph per field. Record the decision, not a re-summary of the diff.
   - Append new entries at the bottom; do not edit or renumber existing entries from prior sessions.

3. **Do not modify the original plan files.** The plan stays as-is so future agents can compare intent (the plan) against reality (`DRIFT.md`).

4. **If no deviations occurred**, do not create or modify `DRIFT.md`. An empty drift log is not informative.

---

## Execution order

1. **Phase 1** → Read instructions and existing tests. Produce test plan. Wait for user confirmation.
2. **Phase 2** → Execute Red-Green-Refactor cycles in the order specified by the instructions.
3. **Phase 3** → Final verification: run all affected tests, check lints.
4. **Phase 4** → Run the unit test suite (`pytest tests --ignore=tests/integration`) to catch regressions. Do not run `tests/integration/`.
5. **Phase 5** → If a plan was used, record any deviations in `DRIFT.md` inside the plan directory. Skip when no plan directory exists.

Do not skip phases. Keep changes scoped to what the instructions specify.
