# 2026-08-12: Git repo cleanup (history rewrite + artifact policy)

## What was done

1. **Policy**: generated artifacts are no longer tracked (`.gitignore` policy
   "nothing generated — regenerate from config"). Ignored:
   `data/experiment_results/`, `data/experiment_results_nothin/`,
   `data/salvage_*/`, `output/`, `plots/*.png`, `paper/figs/*.png`, `*.npz`,
   `paper/thesis.pdf`.
2. **History rewrite** via `git filter-repo` purged all generated binaries
   (`*.png`, `*.gif`, `*.jpg`, `*.jpeg`, `*.npz`, `*.pdf`) and the
   `data/experiment_results*` + `data/mdp_results` dirs from **all** history.
   `.git` shrank **62M → 11M**. Both `main` and `alt` force-pushed to origin.
3. **Deleted** scratch file `tmp_salvage_comparison.py`.
4. Committed pending WIP (code/docs/config) in 3 commits + 1 restore commit.

## What stays tracked (do NOT purge / re-ignore)

- Source data: `data/*.xls` (PMRC calibration + baseline), `data/example_stands.csv`
  (force-included via `!`).
- Scenario gold CSVs: `test_csvs/*.csv` + `desc.md`.
- Hand-authored diagram: `paper/figs/agent_env_loop_barto_rl.png`
  (NOT script-generated; kept via `!paper/figs/agent_env_loop_barto_rl.png`).
- Editor config: `.vscode/*.json`.
- Vendored OpenAI `baselines/` code (150 `.py`) exists only in history (removed
  from tree in an old commit); its **code** was preserved — only its generated
  result images were purged.

## Future-agent rules

- **Do not** `git add` PNGs/NPZs/CSVs/PDFs produced by `scripts/` or `examples/`.
  They are gitignored. If a plot must be version-controlled, argue for it
  explicitly (none qualify today).
- Regenerate results from config: `uv run python scripts/run_full_matrix_experiment.py`,
  `uv run python scripts/run_salvage_sensitivity.py`, etc. Outputs land in
  `data/experiment_results/` (gitignored).
- **Old clones are stale**: anyone with a pre-2026-08-12 clone must re-clone
  (history was rewritten; all commit hashes changed).

## Pre-existing test failures (NOT caused by this cleanup)

`tests/test_pmrc_alignment.py` — 12 failures
(`test_pmrc_initial_row_matches_gold_fixture`, `test_pmrc_matches_gold_csv_rows`,
`test_pmrc_duplicate_age_rows_assert_thinning_behavior`). Verified pre-existing:
the same 12 fail at the pre-WIP commit `d3776d4` (original tip) in a clean
worktree. Root cause is a code-vs-gold-CSV value drift (e.g.
`initial_volume_tvob` off ~77% on scenario_1_nothin); WIP also modified
`data/PMRC.xls` calibration. 196 other tests pass. Fixing this is a separate
task from the repo cleanup.

## Backups (ephemeral — in /tmp, cleared on reboot)

- `/tmp/sss-backup-20260812-022834.bundle` (41M, full pre-rewrite history incl. stash)
- `/tmp/sss-worktree-20260812-022834.tar.gz` (24M, pre-cleanup working tree)
- `/tmp/sss-cleanup-ts.txt` (timestamp)

These are the **only** copies of the pre-rewrite history. Copy to durable
storage if long-term recovery is needed.
