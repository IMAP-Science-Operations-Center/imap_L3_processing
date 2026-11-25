---
name: create-phased-plan
description: >-
  Create a Gherkin-BDD-phased implementation plan with TDD steps from a card
  description. Use when the user provides card context (card number,
  description, goals, designs, acceptance criteria, implementation details)
  and wants a structured plan produced under .claude/plans/.
disable-model-invocation: true
---

# Create Phased Plan

Produce a Gherkin-BDD-phased implementation plan with TDD steps from a card description. The user will provide card context — card number, description, goals, implementation details, designs, and any other relevant information. Use this to produce a structured plan directory under `.claude/plans/`.

## When to Use

- User explicitly invokes this skill via `/create-phased-plan` (optionally with a card number argument, e.g. `/create-phased-plan 3101`)
- User provides a card (ticket/issue) and asks for a phased implementation plan
- You are Step 2 of the standard IMAP L3 processing workflow (see `.claude/README.md`)

**Do NOT use for:**
- Ad-hoc planning that doesn't warrant a full plan directory
- Implementing code — this skill only produces plans, never code

---

## Argument: card number (optional)

The skill accepts an optional **card number** as the first argument:

- `/create-phased-plan 3101` → card number is `3101`
- `/create-phased-plan` → no slash-command argument

If no slash-command argument is given, scan the user's message for a card number (e.g. "ticket #3101", "card 3101", "IMAP-3101") and treat that as the card number.

If neither the argument nor the message contains a card number, omit the prefix from the directory name. Do not invent or pad a card number.

The card number controls the plan directory name (see Phase 4) and the **Card reference** field in `overview.md`.

---

## Phase 1: Understand the card

1. **Capture the card number**, if any, from the slash-command argument or the user's message.
2. **Read everything the user provided** — description, acceptance criteria, goals, implementation details, linked designs, and any additional context.
3. **Explore the codebase** to understand the areas that will be affected. Read relevant production files, existing tests, and related modules to build a clear picture of the current state.
4. **Identify ambiguities or gaps** in the requirements. If anything is unclear, ask the user before proceeding.

---

## Phase 2: Decompose the card into Gherkin features

Each **BDD phase** corresponds to a **Gherkin Feature** — a coherent, user-facing capability that can be implemented and verified independently. Express each feature using standard Gherkin syntax:

```gherkin
Feature: <user-facing capability>

  Scenario: <what the user can observe>
    Given <starting state or precondition>
    When <action the user or system takes>
    Then <observable outcome>
    And <additional observable outcome>
```

Conventions:
- Use `Given` / `When` / `Then` / `And` / `But` exactly as Gherkin specifies.
- Each `Then` / `And` / `But` after `Then` must describe an **outcome observable at the test boundary** — return values, output array contents, quality-flag values, CDF variables and attributes, output file paths, mock-call patterns. Do not write `Then` clauses about purely internal bookkeeping that no test could observe.
- Use `Background:` for setup shared across all Scenarios in a Feature.
- Use `Scenario Outline:` with `Examples:` when the same behavior is exercised across a table of inputs.

Guidelines for phases:
- One Gherkin Feature → one BDD phase → one phase markdown file.
- Each Scenario should map to one (or a small handful of) test functions in Phase 3.
- If a Feature has more than ~5 Scenarios, consider splitting it into two Features.
- Order phases so dependencies are respected — foundational features first, dependent features later.
- Name phases after the feature they deliver in `kebab-case` (e.g. `moment-predictive-ephemeris-flag`, `fetches-swapi-l3a-dependency`).

### Example

```gherkin
Feature: SWE L3 moments flag epochs that used predicted ephemeris

  Background:
    Given SweL3Dependencies with three SWE L2 epochs

  Scenario: An epoch whose DPS to RTN rotation needs a predicted kernel is flagged
    Given the SPICE lookup for the second epoch only succeeds with a predicted-ephemeris kernel
    When SweProcessor.calculate_moment_products is called
    Then swe_flags for the second epoch includes SweL3Flags.PREDICTIVE_EPHEMERIS
    And swe_flags for the first and third epochs does not include it

  Scenario: An epoch with no usable SPICE coverage is filled
    Given the SPICE lookup for the first epoch raises SpiceyError even with predicted kernels
    When SweProcessor.calculate_moment_products is called
    Then the RTN moment outputs for the first epoch are fill values
```

---

## Phase 3: Break each Feature into TDD steps

For each Gherkin Scenario, define **TDD (Red / Green / Refactor) steps** that implement it:

- **Red**: which test(s) to write, what they assert, and which test file they belong to. The test's setup / exercise / assertions should mirror the Scenario's `Given` / `When` / `Then` clauses. Name the Scenario in the step heading so the link from spec to test stays explicit.
- **Green**: the minimum production code needed to make the failing tests pass.
- **Refactor**: any cleanup, extraction, or structural improvement expected after Green.

Each TDD step must reference specific files (production and test) and describe the behavioral change being introduced.

---

## Phase 4: Create the plan directory

### Directory name

Construct the plan directory name from the card number (if any) and a concise kebab-case title:

- **With card number:** `.claude/plans/<card-number>-<concise-title>/` — e.g. `.claude/plans/3101-stack-plot-followup/`
- **Without card number:** `.claude/plans/<concise-title>/` — e.g. `.claude/plans/fix-save-session-skip-missing-files/`

The card number prefix is **only** added when a card number was found in the argument or the user's message.

### Directory layout

```
.claude/plans/<directory-name>/
├── overview.md              # Card reference, feature summary, dependency graph
├── phase-1-<name>.md        # Gherkin Feature + TDD steps for phase 1
├── phase-2-<name>.md        # Gherkin Feature + TDD steps for phase 2
├── ...                      # One markdown file per BDD phase
└── mock_designs/            # Mock designs and wireframes (if applicable)
```

### `overview.md` — required sections

- **Card reference** — card number (if any) and title.
- **Goal** — what the card achieves from the user's perspective.
- **Affected areas** — which production files, modules, and test files are involved, grouped under `Production` and `Tests` subheadings.
- **Phase summary** — a numbered table of all phases with the Gherkin Feature title and a one-line description of each. Do not paste the full Gherkin blocks here — they live in the phase files.
- **Dependency graph** — a visual representation (ASCII tree, table, or list) of which phases depend on which and which phases block which. Must be consistent with the per-phase `Depends on` / `Blocks` headers.
- **Doc references** — links to relevant rules under `.claude/rules/` and any cross-cutting design notes the plan as a whole leans on.

### Phase files (`phase-N-<name>.md`) — required sections

Every phase file **must begin** with a header block that captures dependencies and references at a glance, immediately under the title:

```markdown
# Phase N — <Feature Title>

> **Depends on:** Phase X (<one-line reason>), Phase Y (<one-line reason>) — or `none`
> **Blocks:** Phase A (<one-line reason>), Phase B (<one-line reason>) — or `none`
> **Doc references:**
> - [`.claude/rules/test-quality-coverage.md`](../../rules/test-quality-coverage.md) — test quality rules
> - [`.claude/rules/agent-guidelines.md`](../../rules/agent-guidelines.md) — agent guidelines (if relevant to this phase)
> - [`phase-X-<name>.md`](phase-X-<name>.md) — context this phase builds on (if any)
> - <any other relevant rule, design doc, or sibling phase>
```

Rules for the header block:
- `Depends on` and `Blocks` are **always** present. If there are no dependencies or no downstream blocks, write `none` explicitly — do not omit the line.
- The `Doc references` section is **always** present. At minimum, link `.claude/rules/test-quality-coverage.md`. Add `.claude/rules/agent-guidelines.md` and any sibling phase files when they are genuinely relevant; do not pad with irrelevant links.
- Use relative paths (`../../rules/...` from a phase file, `phase-X-<name>.md` for siblings in the same plan directory).
- The header block must stay consistent with the dependency graph in `overview.md`.

After the header block, the phase file must include:

- **Goal** — the user-facing capability this phase delivers (one paragraph).
- **Gherkin Feature & Scenarios** — the full Gherkin block from Phase 2 in a fenced ` ```gherkin ` code block. This is the contract for the phase; every Scenario must be covered by a TDD step and every Done-criteria item.
- **Affected files** — production and test files that will be created or modified, listed under `Production` and `Tests` subheadings.
- **TDD steps** — ordered list of Red / Green / Refactor cycles. Each Red step's heading should name the Scenario it covers (e.g. `### 2.1 Red — Scenario: Two data blocks separated by a gap render as two line segments`).
- **Done criteria** — a checklist that, when complete, proves every Scenario in the Gherkin block passes.

### `mock_designs/`

If the card involves UI changes, create or place mock designs in this subdirectory. Reference them from the relevant phase files.

---

## Phase 5: Present the plan for review

After creating all files, present a summary to the user:
1. List all phases with their Gherkin Feature titles and one-line descriptions.
2. Show the dependency graph (the same one used in `overview.md`).
3. Call out any assumptions or decisions made.
4. Ask the user to review and confirm before proceeding to `/implement-tdd`.

---

## Execution order

1. **Phase 1** → Capture the card number argument; understand the card; explore affected code; clarify ambiguities.
2. **Phase 2** → Decompose into Gherkin Features with Scenarios.
3. **Phase 3** → Define TDD steps within each Feature, mapping each Scenario onto specific tests.
4. **Phase 4** → Create the plan directory (with optional card-number prefix), `overview.md`, per-phase markdown files (each opening with the `Depends on` / `Blocks` / `Doc references` header block), and `mock_designs/` if needed.
5. **Phase 5** → Present the plan summary and wait for user confirmation.

Do not skip phases. Do not begin implementation — this skill only produces the plan.
