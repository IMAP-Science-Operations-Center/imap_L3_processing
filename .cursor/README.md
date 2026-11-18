# IMAP L3 Processing Development Workflow

## Step 1: Review Card Context

Read the card (ticket/issue) thoroughly. Understand the requirements, acceptance criteria, and any linked science documents, algorithm descriptions, or instrument-team reference outputs before proceeding.

## Step 2: Create Phased Plan with `/create-phased-plan`

Run `/create-phased-plan` and provide the card context: card number, description, goals, implementation details, acceptance criteria, algorithm references, and any other relevant information. The command will explore the codebase, decompose the work into BDD phases with TDD steps, and produce a structured plan directory under `.cursor/plans/`.

## Step 3: Plan with `/implement-tdd`

Run `/implement-tdd` in **Plan Mode** with a step per phase. This analyzes the plan files, reads existing tests, and produces a test plan for confirmation before any code is written.

## Step 4: Build in Agent Mode

Execute the plan in **Agent Mode**. The agent works through each phase's TDD cycles (Red → Green → Refactor), writing tests first, then the minimum production code to pass them, then cleaning up.

## Step 5: Review the changeset

- Run the `test-quality-coverage` subagent to audit coverage and test quality of the diff.
- Run the `comment-remover` subagent to find comments that should be removed and names that aren't self-documenting.
- Run `uv run ruff check .` and `uv run ruff format --check .`.
- Run the affected instrument's unit tests (e.g. `uv run python -m unittest discover -s tests/swe -t .`), its integration test if one exists (e.g. `uv run python -m unittest tests.integration.test_swe_processor_integration`), and finally the full suite with `uv run python -m unittest discover tests`.

## FYIs
- Keep an eye on your current agent's context window. If you use up a lot of the context in the planning phase, you will likely want to create a new agent window to begin execution of the plan.
- Test fixtures are stored in Git LFS. Run `git lfs pull` before running tests on a fresh clone.
