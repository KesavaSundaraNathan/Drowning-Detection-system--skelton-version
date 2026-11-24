---
description: Rewrite project_state.md so a fresh session on any account can continue
disable-model-invocation: true
---
Rewrite project_state.md in place (max 120 lines, no narrative, no conversation summary).
1. Facts from git only: run `git status -sb` and `git log --oneline -8`.
2. Fill every section: phase and step, gate checklist, done (with shas), in progress (files and exact
   state), next actions (max 3, each with its verifying command), decisions, open questions, metrics
   ledger (update only rows measured this session, with command and date).
3. Drop anything finished more than one phase ago. Move long history to docs/archive/.
4. If there are uncommitted changes, ask me whether to commit them as `wip(PHASE.STEP): ...`. Do not
   commit without my yes.
5. Reply with only the line count of the file and the first next action.
