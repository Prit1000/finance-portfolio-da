---
description: Runs parallel security and quality code review for a specific Finance Portfolio Pipeline module. Pass the spec name as argument e.g. /code-review-feature 02-data-cleaning
allowed-tools: Bash(git diff), Bash(git diff --staged)
---

Run the full code review pipeline for the module
specified in $ARGUMENTS.

If no argument is provided, stop immediately and say:
> "Please provide a spec name. Usage:
> /code-review-feature <spec-name> e.g.
> /code-review-feature 02-data-cleaning"

If `.claude/specs/$ARGUMENTS.md` does not exist, stop
immediately and say:
> "Spec file not found at
> .claude/specs/$ARGUMENTS.md. Please check the spec
> name and try again."

---

## Pre-flight Check

Before invoking any subagents, collect the diff:

- Run `git diff` for unstaged changes
- Run `git diff --staged` for staged changes
- Combine both into a single diff

If both are empty, stop immediately and say:
> "No changes detected. Implement the module before
> running code review."

---

## Step 1: Parallel Review

Invoke both subagents simultaneously with the same
context. **Both subagents must run in parallel.** Do
not wait for one to finish before starting the other.

**portfolio-security-reviewer** receives:

- The combined diff from the pre-flight check
- Spec file for context: `.claude/specs/$ARGUMENTS.md`
- Source files to reference:
  - The corresponding `src/` module
  - `config.py`
  - `requirements.txt`
- Instruction: Review only the changed code for
  applicable security concerns (secrets in code,
  unsafe deserialization, path handling, dependency
  hygiene). Do not comment on code quality or style.
  Be honest about scope — this is a local data
  pipeline, not a web app.

**portfolio-quality-reviewer** receives:

- The combined diff from the pre-flight check
- Spec file for context: `.claude/specs/$ARGUMENTS.md`
- Source files to reference:
  - The corresponding `src/` module
  - `config.py`
  - `src/schemas.py` if it exists
  - `main.py`
- Instruction: Review only the changed code for
  quality, pandas idioms, architectural rule
  compliance, and maintainability. Do not comment on
  security concerns.

---

## Step 2: Unified Report

Once both subagents have completed, combine their
findings into a single unified report. De-duplicate
any overlapping findings — if both agents flagged the
same line for different reasons, merge into one
finding with both perspectives noted.

Structure the combined report as:

```
## Code Review Report — $ARGUMENTS

### Security Findings
[portfolio-security-reviewer output]

### Quality Findings
[portfolio-quality-reviewer output]

### Combined Action Plan

Ordered checklist of everything that needs attention,
prioritised by severity:

1. [Security CHANGES REQUESTED items — real secret /
   deserialization risks]
2. [Quality CHANGES REQUESTED items — architectural
   rule violations]
3. [Security APPROVED WITH SUGGESTIONS items]
4. [Quality APPROVED WITH SUGGESTIONS items]
5. [Polish ideas from both]

### Overall Verdict

One of:
- **APPROVED** — ready to commit
- **APPROVED WITH SUGGESTIONS** — can commit;
  address suggestions in future steps
- **CHANGES REQUESTED** — must fix before
  committing; see action plan above
```

---

## Step 3: Ask for Approval

After presenting the unified report, ask:

> "Do you want me to implement the action plan now?"

Wait for explicit user confirmation before making any
changes. Do not touch any files until the user
approves.

---

## Rules

- Do NOT edit any files before user approval
- Do NOT start one reviewer before the other — both
  must run in parallel
- Do NOT skip the pre-flight diff check
- Do NOT proceed if the spec file at
  `.claude/specs/$ARGUMENTS.md` does not exist —
  report it and stop
- If either subagent fails or returns no output,
  report it and do not present a partial review as
  complete