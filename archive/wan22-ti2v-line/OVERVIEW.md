# Task 3 Overview

Updated on `2026-06-05`.

> **Status correction (2026-07-06).** This file describes the `Wan2.2 TI2V 5B / WanI2V` line, which stopped on 2026-05-10 and is no longer the mainline. The active line since 2026-06-06 is `Wan2.1-T2V-1.3B (WanT2V) / DMD2` progressive `50 -> 8 -> 4` distillation on OpenVid-1M, with outputs under `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/`. See `research/T0_project_analysis.md` (section 0) for the verified current state. The content below is kept as historical context.

This file was previously the active source of truth for Task 3. Historical notes are preserved under `archive/`.

## Current Mainline

The current mainline is FastGen native `Wan2.2 TI2V 5B / WanI2V / DMD2` distillation on the UST server.

Active assumptions:

- Server alias: `ust_ip`
- Login target: `chenqingzhan@111.17.197.107`
- Hostname: `RTX-5090-32G-X8`
- FastGen path: `/data/chenqingzhan/FastGen`
- Model path: `/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers`
- Data path: `/data/datasets/OpenVid-1M/webdataset`
- Main FastGen config: `fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`

Use `03-dmd-distillation/HANDOFF.md` for detailed verified history and known failure modes.

## Default Pipeline

Use the unified experiment layer in the repository root:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

The lower-level scripts in `03-dmd-distillation/scripts/` remain available for setup, data conversion, and direct FastGen debugging. New experiment variants should be added as `experiments/configs/*.env`, not as standalone root scripts.

## Validated Route

The most stable recorded route is:

- `torchrun --standalone --nproc_per_node=2`
- two GPUs with less than about `100 MiB` already used
- `trainer.ddp=False`
- `trainer.fsdp=True`
- `trainer.fsdp_cpu_offload=True`
- `trainer.batch_size_global=2`
- `dataloader_train.batch_size=1`
- `dataloader_train.sequence_length=65`
- `model.input_shape=[48,17,44,80]`
- `model.student_sample_steps=2`
- `model.guidance_scale=null`
- `FASTGEN_DISABLE_MEDIA_LOGGING=true`

The exact native `guidance_scale=5.0` path has repeatedly OOMed on this server unless the experiment is carefully constrained. Treat CFG=5 as an explicit experiment, not the default smoke/stage route.

## Experiment Policy

Each run should have:

- one config: `experiments/configs/*.env`
- one remote log: `/data/chenqingzhan/logs/<run>.log`
- one remote output/checkpoint path
- one short result note: `experiments/results/YYYY-MM-DD-run-name.md`

Before launching:

1. run `check_remote.sh`
2. inspect GPU memory and utilization
3. run `run_remote.sh --dry-run`
4. verify `RUN_NAME`, log path, output root, checkpoint path, and data tag

After launching:

1. record PID and log path
2. check the first metrics and first checkpoint
3. write a short result note

## Active Files

| File | Purpose |
|---|---|
| `HANDOFF.md` | Detailed current handoff and verified server facts |
| `Wan22_TI2V_5B_Execution_Plan.md` | Original execution plan and blockers |
| `FastGen_Guide.md` | FastGen technical notes |
| `scripts/README.md` | Reusable helper script inventory |

## Archived Context

Moved out of the active path:

- weekly reports: `archive/reports/weekly/`
- midterm package: `archive/reports/midterm-2026-05/`
- old Phase 0 reports: `archive/reports/phase0-dmd-distillation/`
- old meeting/personal notes: `archive/notes/`
- old one-off Wan2.2 scripts: `archive/scripts/wan22-i2v-2026-05/`
- old FastGen patch snapshots: `archive/server-patches/`
- long surveys: `archive/surveys/`

Use archived files for background only. Re-check the server before treating any archived status as current.

