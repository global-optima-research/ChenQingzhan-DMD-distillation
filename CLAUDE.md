# CLAUDE.md - AI Assistant Context Configuration

## Identity

- Developer: Chen Qingzhan / Chen Hing Chin
- Project: PVTT Task 3, DMD distillation and acceleration
- Active role: research pipeline agent
- Active workspace: `03-dmd-distillation/` plus the unified `experiments/` layer

## Current Mainline

The active technical target is FastGen native `Wan2.2 TI2V 5B / WanI2V / DMD2` distillation.

Default server access:

```bash
ssh ust_ip
```

Known server paths:

- FastGen: `/data/chenqingzhan/FastGen`
- Logs: `/data/chenqingzhan/logs`
- Output root: `/data/chenqingzhan/fastgen_output`
- HuggingFace cache: `/data/chenqingzhan/.cache/huggingface`
- Data: `/data/datasets/OpenVid-1M/webdataset`

## Read Order

Start every session with:

1. `README.md`
2. `agents/README.md`
3. `experiments/README.md`
4. `03-dmd-distillation/HANDOFF.md`
5. `03-dmd-distillation/OVERVIEW.md`

## Experiment Workflow

Use config-driven launches:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

Rules:

- One experiment equals one `experiments/configs/*.env` file.
- Always check GPU state before launching.
- Always dry-run before launching.
- Record remote log path, output path, checkpoint path, and status.
- Add a short result note under `experiments/results/`.
- Do not create new root-level one-off scripts.
- Archived reports and scripts are context only, not current state.

## Language Rules

- Communicate with the user in Chinese unless they initiate in English.
- Keep code, shell scripts, config files, and commit messages in English.
- Use absolute dates for experiment status and handoffs.

## Repo Structure

| Path | Purpose |
|---|---|
| `experiments/` | Active config-driven experiment pipeline |
| `03-dmd-distillation/` | Active Task 3 technical notes and helper scripts |
| `agents/` | Agent roles and operating rules |
| `artifacts/` | Curated local inference evidence |
| `reports/` | Active short reports only |
| `archive/` | Historical surveys, reports, notes, scripts, and patch snapshots |

