---
description: Writes and runs tests for a specific Finance Portfolio Pipeline module. Pass the spec name as argument e.g. /test-feature 02-data-cleaning
allowed-tools: Bash(python -m pytest)
---

Run the full testing pipeline for the module specified
in $ARGUMENTS.

If no argument is provided, stop immediately and say:
> "Please provide a spec name. Usage: /test-feature
> <spec-name> e.g. /test-feature 02-data-cleaning"

If `.claude/specs/$ARGUMENTS.md` does not exist, stop
immediately and say:
> "Spec file not found at
> .claude/specs/$ARGUMENTS.md. Please check the spec
> name and try again."

---

## Step 1: Write Tests

Invoke the **portfolio-test-writer** subagent with the
following context:

- Spec file to base tests on:
  `.claude/specs/$ARGUMENTS.md`
- Source files to read for **structure only** (never
  for test logic):
  - The corresponding `src/` module (e.g.
    `src/data_cleaning.py` for `02-data-cleaning`)
  - `src/schemas.py` if it exists
  - `config.py`
- Output test file to create:
  `tests/unit/test_$ARGUMENTS.py`
- Instruction: Write tests based on what the spec
  says the module SHOULD do. Do NOT derive test
  logic from reading the implementation. Cover happy
  paths, edge cases, schema enforcement, idempotency,
  and error handling per the spec's acceptance
  criteria.

Wait for portfolio-test-writer to fully complete and
confirm the test file has been written before
proceeding to Step 2.

---

## Step 2: Run Tests

Once portfolio-test-writer has finished, invoke the
**portfolio-test-runner** subagent with the following
context:

- Test file to execute:
  `tests/unit/test_$ARGUMENTS.py`
- Spec file for context:
  `.claude/specs/$ARGUMENTS.md`
- Source files to analyze against when diagnosing
  failures:
  - The corresponding `src/` module
  - `src/schemas.py`
  - `config.py`
- Run command:
  `python -m pytest tests/unit/test_$ARGUMENTS.py -v`
- Instruction: Run ONLY the specified test file. Do
  NOT run the full test suite. Analyze any failures
  by cross-referencing the test code, the spec, and
  the source files. Classify each failure as a bug in
  source, a missing feature, or a test issue.

---

## Handoff Rules

- Do NOT start Step 2 until Step 1 is fully complete
- Do NOT attempt to fix any code regardless of what
  the test results show
- Do NOT run any tests beyond
  `tests/unit/test_$ARGUMENTS.py`
- If portfolio-test-writer reports it could not write
  the test file, stop and report the reason — do NOT
  proceed to Step 2

---

## Final Output

After both subagents complete, produce a combined
summary:

```
## Testing Pipeline Report — $ARGUMENTS

### Step 1 — Tests Written
[List each test written with a one-line description
of which spec requirement (FR<n> or acceptance
criterion) it validates]

### Step 2 — Test Results
[Mirror the portfolio-test-runner's structured
report]

### Verdict
One of:
- ✅ Ready for code review — all tests pass
- ❌ Needs fixes — list the failing tests and their
  root causes
```