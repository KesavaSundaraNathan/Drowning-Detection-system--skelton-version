---
description: Implement one approved option from docs/proposals/ in small verified steps
argument-hint: "[proposal number] [option letter]"
disable-model-invocation: true
---
Implement option $1 of docs/proposals/$0-*.md. Nothing else.
1. Read only that proposal and the files it names.
2. Work on branch exp/$0-$1 off lifeguard-rebuild.
3. Do the first experiment first. Run its validation command (dev split only) and report before and after
   in 8 lines or fewer. If the metric does not move the expected way, stop. Do not keep tuning.
4. Commit per logical step. Update project_state.md (ledger row, next actions, a decision line). Merge
   back only after validation passes and I say so.
5. Never run the test split or touch an invariant (see CLAUDE.md "Invariants"). Stop after the first
   experiment unless I say continue.
