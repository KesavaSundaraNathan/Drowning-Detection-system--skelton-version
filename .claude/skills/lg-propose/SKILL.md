---
description: Read-only proposal for one targeted issue, written to docs/proposals/
argument-hint: "[issue card text, or path to docs/issues/ID.md]"
disable-model-invocation: true
---
Propose, do not implement. Edit nothing except the proposal file. No installs, downloads, training runs
or test-split runs.

Issue card from me: $ARGUMENTS

Budget: read at most 8 files, starting with the file:line pointers in the card. Use grep and line ranges.
At most 3 web fetches, only to verify licences or versions. Command output over 40 lines goes to logs/.

Write docs/proposals/NNN-SLUG.md (next free number, max 70 lines):
1. Diagnosis: ranked causes, each with file:line or a command result. Label guesses as guesses.
2. Options (max 3): the change, expected effect on the target metric (a range, labelled estimate), cost
   (hours, data needed, fps or latency impact), risk, licence check for code, weights and training data
   (SPDX id and URL, or UNVERIFIED), rollback.
3. Recommendation, and what would change your mind.
4. Validation: metric, dev set, accept threshold, one command.
5. Smallest first experiment (under 1 hour) with a kill criterion.
6. Not touched: the invariants this stays clear of (see CLAUDE.md "Invariants").

Then reply in 5 lines: file path, recommendation, first experiment. Stop and wait for my approval.
