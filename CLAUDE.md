# CLAUDE.md - AI Assistant Context Configuration

## Identity

- Developer: Chen Qingzhan / Chen Hing Chin
- Project: PVTT Task 3, DMD distillation and acceleration
- Active role: research pipeline agent
- Active workspace: `03-dmd-distillation/` plus the unified `experiments/` layer

## Current Mainline

The active technical target is FastGen native `Wan2.1-T2V-1.3B (WanT2V) / DMD2` progressive distillation (`50 -> 8 -> 4` steps) on OpenVid-1M. Corrected on 2026-07-06: the earlier `Wan2.2 TI2V 5B / WanI2V / DMD2` line stopped on 2026-05-10 and is historical context. Verified current state: `research/T0_project_analysis.md` (section 0).

Default server access:

```bash
ssh ust_ip
```

Known server paths:

- FastGen: `/data/chenqingzhan/FastGen`
- Output root (current line, since 2026-06-08): `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT`
- Output root (legacy, stale since 2026-06-06): `/data/chenqingzhan/fastgen_output`
- Logs (legacy central dir, stale since 2026-06-06): `/data/chenqingzhan/logs`; current runs log into `<run_dir>/logs/` under `FASTGEN_OUTPUT`
- HuggingFace cache: `/data/chenqingzhan/.cache/huggingface`
- Data: `/data/datasets/OpenVid-1M/webdataset`

## Read Order

Start every session with:

1. `README.md`
2. `agents/README.md`
3. `experiments/README.md`
4. `03-dmd-distillation/HANDOFF.md`
5. `03-dmd-distillation/OVERVIEW.md`

For a research-planning/director session, use `research/planner_startprompt.md` as the startup prompt. That role reads `research/README.md`, `research/workflow.md`, and `research/task_brief_template.md` first, then reconciles this local research repo with the actual remote FastGen state at `ust_ip:/data/chenqingzhan/FastGen`.

## Experiment Workflow

Use config-driven launches:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
bash experiments/bin/run_remote_script.sh --dry-run experiments/configs/wan21_dmd2_step4_relay_eval10.env
bash experiments/bin/run_remote_script.sh experiments/configs/wan21_dmd2_step4_relay_eval10.env
```

The canonical experiment record is `reports/experiment-report-wan21-t2v-dmd2-progressive.md`; update it after every run. The legacy wan22 runner and configs are archived (`experiments/bin/run_remote.sh`, `archive/scripts/wan22-env-configs/`).

Rules:

- One experiment equals one `experiments/configs/*.env` file.
- Always check GPU state before launching.
- Always dry-run before launching.
- Record remote log path, output path, checkpoint path, and status.
- Add a short result note under `experiments/results/`.
- Do not create new root-level one-off scripts.
- Archived reports and scripts are context only, not current state.

Research workflow rule:

- The local repository contains stage reports and curated evidence.
- Actual code, logs, outputs, and checkpoints live on `ust_ip:/data/chenqingzhan/FastGen`.
- A planner agent must do a read-only remote status check before treating any local handoff as current.
- The planner role should not launch training unless the user explicitly changes the task from research planning to experiment execution.

## Language Rules

- Communicate with the user in Chinese unless they initiate in English.
- Keep code, shell scripts, config files, and commit messages in English.
- Use absolute dates for experiment status and handoffs.

## Repo Structure

| Path | Purpose |
|---|---|
| `experiments/` | Active config-driven experiment pipeline |
| `research/` | Planner/content-agent workflow for literature, novelty, T0/T1-T4 tasks, and submission planning |
| `03-dmd-distillation/` | Active Task 3 technical notes and helper scripts |
| `agents/` | Agent roles and operating rules |
| `artifacts/` | Curated local inference evidence |
| `reports/` | Active short reports only |
| `archive/` | Historical surveys, reports, notes, scripts, and patch snapshots |
