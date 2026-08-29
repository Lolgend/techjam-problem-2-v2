# Project Workflow

## Guiding Principles

1.  **The Plan is the Source of Truth:** All work must be tracked in `plan.md`
2.  **The Tech Stack is Deliberate:** Changes to the tech stack must be
    documented in `tech-stack.md` *before* implementation
3.  **Test-Driven Development:** Write unit tests before implementing
    functionality
4.  **High Code Coverage:** Aim for >80% code coverage for all modules
5.  **User Experience First:** Every decision should prioritize user experience and developer ergonomics
6.  **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use
    `CI=true` for watch-mode tools (tests, linters) to ensure single execution.

## Task Workflow

All tasks follow a strict lifecycle:

### Standard Task Workflow

1.  **Select Task:** Choose the next available task from `plan.md` in sequential
    order

2.  **Mark In Progress:** Before beginning work, edit `plan.md` and change the
    task from `[ ]` to `[~]`

3.  **Write Failing Tests (Red Phase):**

    -   Create a new test file for the feature or bug fix.
    -   Write one or more unit tests that clearly define the expected behavior
        and acceptance criteria for the task.
    -   **CRITICAL:** Run the tests and confirm that they fail as expected. This
        is the "Red" phase of TDD. Do not proceed until you have failing tests.

4.  **Implement to Pass Tests (Green Phase):**

    -   Write the minimum amount of application code necessary to make the
        failing tests pass.
    -   Run the test suite again and confirm that all tests now pass. This is
        the "Green" phase.

5.  **Refactor (Optional but Recommended):**

    -   With the safety of passing tests, refactor the implementation code and
        the test code to improve clarity, remove duplication, and enhance
        performance without changing the external behavior.
    -   Rerun tests to ensure they still pass after refactoring.

6.  **Verify Coverage:** Run coverage reports using the project's chosen tools:
    `uv run pytest --cov=src --cov-report=term-missing` Target: >80% coverage for new code.

7.  **Document Deviations:** If implementation differs from tech stack:

    -   **STOP** implementation
    -   Update `tech-stack.md` with new design
    -   Add dated note explaining the change
    -   Resume implementation

8.  **Commit Code Changes:**

    -   Stage all code changes related to the task.
    -   Propose a clear, concise commit message e.g, `feat(retriever): implement Google Search candidate model retriever`.
    -   Perform the commit.

9.  **Attach Task Summary with Git Notes:**

    -   **Step 9.1: Get Commit Hash:** Obtain the hash of the *just-completed commit* (`git log -1 --format="%H"`).
    -   **Step 9.2: Draft Note Content:** Create a detailed summary for the completed task. This should include the task name, a summary of changes, a list of all created/modified files, and the core "why" for the change.
    -   **Step 9.3: Attach Note:** Use the `git notes` command to attach the summary to the commit:
        `git notes add -m "<note content>" <commit_hash>`

10. **Get and Record Task Commit SHA:**

    -   **Step 10.1: Update Plan:** Read `plan.md`, find the line for the completed task, update its status from `[~]` to `[x]`, and append the first 7 characters of the *just-completed commit's* commit hash.
    -   **Step 10.2: Write Plan:** Write the updated content back to `plan.md`.

11. **Commit Plan Update:**

    -   **Action:** Stage the modified `plan.md` file.
    -   **Action:** Commit this change with a descriptive message (e.g., `conductor(plan): Mark task 'Implement retriever agent' as complete`).

### Phase Completion Verification and Checkpointing Protocol

**Trigger:** This protocol is executed immediately after a task is completed that also concludes a phase in `plan.md`.

1.  **Announce Protocol Start:** Inform the user that the phase is complete and the verification and checkpointing protocol has begun.

2.  **Ensure Test Coverage for Phase Changes:**

    -   **Step 2.1: Determine Phase Scope:** Find the starting point. Read `plan.md` to find the Git commit SHA of the *previous* phase's checkpoint. If no previous checkpoint exists, the scope is all changes since the first commit.
    -   **Step 2.2: List Changed Files:** Execute `git diff --name-only <previous_checkpoint_sha> HEAD` to get a precise list of all files modified during this phase.
    -   **Step 2.3: Verify and Create Tests:** For each code file in the list, verify a corresponding test file exists. If a test file is missing, create one.

3.  **Execute Automated Tests with Proactive Debugging:**

    -   Before execution, announce the exact shell command: `uv run pytest`
    -   Execute the command.
    -   If tests fail, inform the user and debug up to 2 attempts before stopping for guidance.

4.  **Propose a Detailed, Actionable Manual Verification Plan:**

    -   Generate a step-by-step verification plan with specific test commands and expected outcomes.
    -   Prompt user for explicit verification feedback.

5.  **Await Explicit User Feedback:**
    -   Pause and await user confirmation.

6.  **Identify Target Commit for Report:**
    -   Identify the hash of the last functional commit made during this phase.

7.  **Attach Auditable Verification Report using Git Notes:**
    -   Attach verification report to target commit with `git notes`.

8.  **Get and Record Phase Checkpoint SHA:**
    -   Update `plan.md` heading with `[checkpoint: <sha>]`.

9.  **Commit Plan Update:**
    -   Commit plan update: `conductor(plan): Mark phase '<PHASE NAME>' as complete`.

10. **Announce Completion.**

### Quality Gates

Before marking any task complete, verify:

-   [ ] All tests pass (`uv run pytest`)
-   [ ] Code coverage meets requirements (>80%)
-   [ ] Code follows project's code style guidelines (`code_styleguides/`)
-   [ ] All public functions/methods are documented
-   [ ] Type safety is enforced (`uv run mypy src`)
-   [ ] No linting or formatting errors (`uv run ruff check src` and `uv run ruff format --check src`)
-   [ ] Documentation updated if needed
-   [ ] No security vulnerabilities or unhandled exceptions

## Development Commands

### Setup

```bash
uv venv
uv sync
```

### Daily Development

```bash
uv run pytest
uv run ruff check src
uv run ruff format src
uv run mypy src
```

### Before Committing

```bash
uv run ruff check src
uv run mypy src
uv run pytest --cov=src --cov-report=term-missing
```
