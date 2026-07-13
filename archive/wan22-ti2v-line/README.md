# Task 3 DMD Distillation

This directory is the active technical workspace for Task 3: FastGen-based DMD distillation and acceleration.

## Active Documents

| File | Use |
|---|---|
| `OVERVIEW.md` | Current state, pipeline, and next actions |
| `HANDOFF.md` | Detailed Wan2.2 TI2V 5B server handoff and verified facts |
| `Wan22_TI2V_5B_Execution_Plan.md` | Original Wan2.2 execution plan and known constraints |
| `FastGen_Guide.md` | FastGen-specific notes and usage details |
| `scripts/README.md` | Reusable lower-level FastGen helper scripts |

## Default Experiment Entry

Prefer the unified launcher in the repository root:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

The scripts in `03-dmd-distillation/scripts/` remain useful as lower-level references, but new experiment variants should be captured as `experiments/configs/*.env` first.

## Archived Material

Historical reports, meeting notes, personal boards, and older surveys were moved to `archive/`. Use them for context only; do not treat them as the current source of truth without re-checking the server.

