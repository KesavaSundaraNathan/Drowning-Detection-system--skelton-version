---
description: Resume the Life Guard rebuild from project_state.md with minimal context
argument-hint: "[optional override, e.g. skip to step 3]"
disable-model-invocation: true
---
Resume the Life Guard rebuild. Keep startup reads small.
1. Read project_state.md fully. Read only docs/LIFEGUARD_REBUILD_PLAN.md's section for the current phase
   (grep the phase heading, read that range - not the whole file).
2. Run `git status -sb` and `git log --oneline -8`. If branch or HEAD differ from project_state.md, trust
   git and say what differs.
3. Do not read anything else, run tests or edit files yet.
4. Reply in 12 lines or fewer: phase and step, done (shas), half-done files, the single next action with
   its verifying command, blockers or questions for me. Then wait for "go".
5. On "go": small commits. After each green commit, edit project_state.md in place (max 5 lines).
My override, if any: $ARGUMENTS
