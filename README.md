# DMD Distillation Research Workspace

This repository is now organized as a lightweight workspace for FastGen-based video diffusion distillation experiments.

## Current Mainline

The active research line is:

- Framework: `FastGen`
- Server alias: `ust_ip`
- Server user/path: `chenqingzhan@111.17.197.107:/data/chenqingzhan/FastGen`
- Active target: `Wan2.2 TI2V 5B / WanI2V / DMD2`
- Default goal: fast experiment iteration, reliable run logging, and short result summaries

Use this repository for experiment planning, reproducible launch configs, result indexing, and handoff notes. Large training outputs and checkpoints should stay on the server unless explicitly curated.

## Quick Start

```bash
ssh ust_ip
bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

After reviewing the dry run, launch with:

```bash
bash experiments/bin/run_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

## Directory Map

| Path | Purpose |
|---|---|
| `experiments/` | Unified experiment configs, launch scripts, and result templates |
| `03-dmd-distillation/` | Active Task 3 notes, FastGen handoff, and reusable scripts |
| `agents/` | Agent roles and operating rules for research sessions |
| `artifacts/` | Curated local inference samples and visual evidence |
| `reports/` | Active report index only; historical reports are archived |
| `archive/` | Old surveys, weekly reports, meeting notes, historical scripts, and server patch records |

## Operating Rule

One experiment should have one config, one remote log path, one output path, and one short local result note. Avoid adding one-off root scripts; put new launchable workflows under `experiments/bin/` and new settings under `experiments/configs/`.

Read in this order for a new session:

1. `README.md`
2. `agents/README.md`
3. `experiments/README.md`
4. `03-dmd-distillation/HANDOFF.md`
5. `03-dmd-distillation/OVERVIEW.md`

