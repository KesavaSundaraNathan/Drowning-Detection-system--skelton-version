---
name: lg-build-focus
description: "Use for any build, development or debugging task: ask every question up front in one batch, lock a plan, then execute end to end with minimal pauses."
---
# LG Build Focus: Plan First, Then Build Straight Through

Goal: spend the user's attention once, at the start. After approval, build, develop and debug in one continuous stretch without stopping for routine confirmation.

## Phase 1: Interrogate (before touching anything)
1. Read the context first: project_state.md if it exists, the repo tree, configs, relevant files, recent git log. Never ask something the code or docs already answer.
2. Ask EVERY question that could change the build, in ONE batch, grouped by theme (scope, behaviour, data, UI, performance, constraints, deployment, testing, what "done" means). Use AskUserQuestion with 2-4 options each, recommended option first. If there are more than 4 questions, send several grouped batches back to back, then stop asking.
3. Be exhaustive, not polite. Cover: edge cases, failure behaviour, naming, dependencies allowed, files that must not be touched, destructive actions allowed or not, performance targets, how to verify success, what to do when blocked.
4. Ask the "permission in advance" questions now so there are none later: may I delete files, install packages, run migrations, restart services, commit to git, run long jobs?

## Phase 2: Lock the plan
1. Write a short plan: ordered steps, files to change, how each step is verified, risks.
2. List assumptions made for anything unanswered, with the default chosen.
3. Create the task list (TaskCreate) with a final verification step.
4. Get one approval. This is the last routine pause.

## Phase 3: Build in one stretch
- Execute the whole plan without asking for confirmation between steps.
- Do not narrate each step. Update the task list and keep going.
- Debug in place: read the error, form a hypothesis, fix, re-run. Try at least 3 distinct approaches before treating something as a blocker.
- Run tests, linters and the app after each meaningful change. Fix failures before moving on.
- Decide small unknowns yourself using the locked assumptions and existing code conventions. Record each decision in a running log.
- Follow existing architecture, naming and formatting. Add no new dependencies unless pre-approved.

## When you MAY stop mid-build (only these)
- An action is irreversible or destructive and was not pre-approved.
- A required credential, file or access is missing and cannot be worked around.
- A discovery invalidates the approved plan (scope or architecture changes).
- The same failure persists after 3 genuinely different attempts.
When stopping, state the blocker, what you tried, and your recommended path. Park that item, and continue with every other step that does not depend on it.

## Phase 4: Finish
1. Run the full verification from the plan (tests, build, start the app, check the acceptance criteria).
2. Report briefly: what was built, how it was verified, decisions made on the user's behalf, anything parked.
3. If the project uses project_state.md, update it: phase, milestones, work in progress, blockers, decisions, next steps.

## Never
- Never ask a question one at a time across many turns.
- Never pause just to say "shall I continue?".
- Never silently skip a failing test or verification step.
- Never expand scope beyond the approved plan.
