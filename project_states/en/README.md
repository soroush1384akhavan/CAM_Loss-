# Project States Log

This folder is a status ledger for the project. Every time a meaningful change occurs, a new `STATE_XXX_*.md` file is created. **Previous files are never edited.**

Each state file contains:

- The exact **commit** it corresponds to; `git checkout <commit>` restores that exact code
- Summary of architecture, loss function, data, and active config
- Latest results and known issues
- Test status
- **Diff from previous state** (from STATE_001 onward)

## Index

| State | Date | Commit | Summary |
|---|---|---|---|
| [STATE_000](STATE_000_baseline_snapshot.md) | 2026-10-21 | `5fa3e1d` | Lab-work snapshot before fixes: CAM sign-flip, early-stopping bug, test run on last-epoch weights |
| [STATE_001](STATE_001_run_analysis_fixes.md) | 2026-10-21 | `a1ac006` | Fixed 9 issues from RUNS_ANALYSIS: CAM centering, post-warmup best selection, signed metrics, augmentation. Full training run not yet done |

> Persian versions of these files are in the parent folder [../](../).

## How to Create a New State

1. Commit the code.
2. Copy the latest state file and increment its number.
3. Update the commit, config, results, and issues sections.
4. Fill in the "diff from previous state" section using:
   `git diff --stat <prev_commit> <new_commit>`
5. Add a row to the table above.
