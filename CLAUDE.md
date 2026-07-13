# CLAUDE.md - AI Assistant Context Configuration

## Identity

- Developer: Chen Qingzhan / Chen Hing Chin
- Project: PVTT Task 3, DMD distillation and acceleration
- Active role: research pipeline agent

## Core Goal

Distill `Wan2.1-T2V-1.3B` from 50-step generation into a high-quality 4-step student with DMD2, via a staged step-count relay: `50 -> 8-step intermediate student -> 4-step student` (the 4-step generator is initialized from the best 8-step checkpoint; optimizer, fake score, and discriminator are reset).

Wording rules: call the method "step-count relay" or "progressive step reduction" — never "phased DMD" / "progressive distribution matching" (taken by arXiv 2510.27684). The discriminator is trainable multiscale MLP heads on frozen-teacher features (code-verified 2026-07-06).

## Server

Default access:

```bash
ssh ust_ip
```

- FastGen repo: `/data/chenqingzhan/FastGen` (entry scripts + runbook: `fastgen/configs/experiments/WanT2V/README.md`)
- Output root: `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT` (each run dir's `config.yaml` is the hyperparameter ground truth)
- HuggingFace cache: `/data/chenqingzhan/.cache/huggingface`
- Data: `/data/datasets/OpenVid-1M/webdataset`
- Legacy, stale since 2026-06-06: `/data/chenqingzhan/fastgen_output`; `/data/chenqingzhan/logs` now only receives one-line-launcher wrapper logs
- sshd throttles rapid repeated connections: batch remote commands into few sessions

## Read Order

Start every session with:

1. `README.md`
2. `reports/experiment-report-wan21-t2v-dmd2-progressive.md` (canonical experiment record)
3. `research/T0_project_analysis.md` (verified facts, evidence grading, novelty axes)
4. `experiments/README.md` (how to submit runs)

For a research-planning/director session, use `research/planner_startprompt.md` as the startup prompt. That role reads `research/README.md`, `research/workflow.md`, and `research/task_brief_template.md` first, then reconciles this local repo with the actual remote FastGen state (read-only) before trusting any local doc.

## Experiment Workflow

Config-driven one-line submission:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
bash experiments/bin/run_remote_script.sh --dry-run experiments/configs/<config>.env
bash experiments/bin/run_remote_script.sh experiments/configs/<config>.env
```

Rules:

- One experiment = one `experiments/configs/*.env` + one remote run dir + one `experiments/results/YYYY-MM-DD-*.md` note; then update the canonical experiment report.
- Change ONE major factor per experiment; new variants are new remote python configs with a new `log_config.name` (configs are snapshots, not templates).
- Always GPU-precheck (selected GPUs < 100 MiB used) and dry-run before launching.
- Quality conclusions must cite the quantitative protocol (VBench 6-dim subset + CD-FVD + Dynamic Degree + cross-seed diversity; full VBench + T2VHE-style human eval for main tables). Training-health metrics are not quality evidence.
- Do not create new root-level one-off scripts. Archived reports and scripts are context only, not current state.

## Research Workflow Rule

- The local repository contains stage reports and curated evidence. Actual code, logs, outputs, and checkpoints live on `ust_ip:/data/chenqingzhan/FastGen`.
- A planner agent must do a read-only remote status check before treating any local doc as current.
- The planner role should not launch training unless the user explicitly changes the task from research planning to experiment execution.

## Language Rules

- Communicate with the user in Chinese unless they initiate in English.
- Keep code, shell scripts, config files, and commit messages in English.
- Use absolute dates for experiment status and handoffs.

## Repo Structure

| Path | Purpose |
|---|---|
| `experiments/` | One-line submission layer: `bin/`, `configs/`, `results/`, `tools/` |
| `reports/` | Canonical experiment record + frozen June artifact index |
| `research/` | Research pipeline: T0 analysis, T1-T4 briefs/reports, experiment plan, `paper/` PDFs |
| `docs/` | FastGen framework manual |
| `agents/` | Agent roles and operating rules |
| `artifacts/` | Curated local inference evidence |
| `archive/` | Historical material only, incl. `archive/wan22-ti2v-line/` (2026-04/05 Wan2.2 line) |
