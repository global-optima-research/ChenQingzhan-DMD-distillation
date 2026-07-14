# Experiments

This directory is the one-line submission interface for the current mainline:
FastGen `WanT2V` (Wan2.1-T2V-1.3B) DMD2 progressive `50 -> 8 -> 4` distillation.
The canonical experiment record is `reports/experiment-report-wan21-t2v-dmd2-progressive.md`.

## Workflow

1. Pick or create one config in `experiments/configs/`.
2. Precheck server state (paths + GPU occupancy).
3. Dry-run to see the exact remote command.
4. Launch only after the dry-run output and GPU choice are clear.
5. Record the result under `experiments/results/` and update the canonical experiment report.

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
bash experiments/bin/run_remote_script.sh --dry-run experiments/configs/wan21_dmd2_step4_relay_eval10.env
bash experiments/bin/run_remote_script.sh experiments/configs/wan21_dmd2_step4_relay_eval10.env
```

## How the one-line layer works

`run_remote_script.sh` runs a parameterized remote script inside the FastGen repo
(`fastgen/configs/experiments/WanT2V/run_*.sh`) with env overrides from the `.env` file:

| Field | Meaning |
|---|---|
| `REMOTE_SCRIPT` | remote script path, relative to the FastGen repo |
| `CUDA_DEVICES` + `GPU_MAX_USED_MB` | GPU precheck before launch (aborts if any selected GPU is busy) |
| `REMOTE_ENV` | quoted `VAR=value` string exported on the server before the script runs |
| `DETACH` | 1 = nohup background + pid echo, 0 = stream in foreground |

New experiment variants (P1a/P1c LR/batch/R1 sweeps, P2 t_list shapes) change the python config
on the remote (`fastgen/configs/experiments/WanT2V/*.py`, one variable per experiment, new
`log_config.name`), then point `CONFIG=...` inside `REMOTE_ENV`.

## Current Configs

| Config | Purpose |
|---|---|
| `wan21_check.env` | Read-only precheck of paths, model, data, and GPUs |
| `wan21_dmd2_step4_relay_train.env` | Train the 4-step student relayed from 8-step `lr_original/0002500` |
| `wan21_dmd2_step4_relay_eval10.env` | Eval-10 inference sweep over relay checkpoints (skips complete ones) |

Legacy: the Wan2.2 line's `wan22_*.env` configs are archived at `archive/scripts/wan22-env-configs/`
and only run through the legacy `experiments/bin/run_remote.sh`.

## Remote Three-Zone Workspace (since 2026-07-13)

The remote now runs a conf-swapped layout (details in remote `FastGen/exp/README.md`):

- `FastGen/exp/` — thin resident launchers (`run.sh`, `eval_sweep.sh`) + one conf + one python config per experiment (one changed variable each)
- `FastGen/experiment/` — `INDEX.md` run index with key conclusions, per-experiment notes (`TEMPLATE.md`), auto-appended `LAUNCHES.log`
- `FastGen/FASTGEN_OUTPUT/` — checkpoints / logs / inference per run (unchanged)

Local one-line submission: `experiments/configs/wan21_sprint.env` drives `exp/run.sh`; switch experiments by editing only its `CONF=` line. Retired checkpoints and side runs were archived (mv, nothing deleted) to `/data/chenqingzhan/archive_pre_sprint_20260713/` on 2026-07-13 (~467G; kept runs retain `net_model`+`pth` for eval/relay).

## Tools

`experiments/tools/` holds server/data utilities salvaged from the early phase (server env setup, model/OpenVid download, WebDataset conversion). They are one-time setup helpers, not run entry points.

## Acceptance Log

`experiments/results/acceptance-log.md` — planner's acceptance ledger: one row per accepted node (research reports, execution checkpoints, freezes) with the independent spot-check performed, verdict, and open watch items. Updated at every acceptance.

## Result Notes

Use `experiments/results/README.md` as the template. A result note should be short enough to scan
and complete enough to resume:

- date and run name
- config path
- remote log path
- output/checkpoint path
- status
- key metrics or failure signature
- next action
