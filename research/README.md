# Research Workflow Entry

Initialized on `2026-07-03`.

This directory embeds the research workflow kit for the FastGen DMD distillation project. It is for research planning, literature positioning, novelty checks, report planning, and task delegation to content agents. It is not the default place for launching training.

## Workspace Split

There are two active workspaces:

| Workspace | Path | Role |
|---|---|---|
| Local research repo | `/Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation` | Stage reports, handoff docs, experiment configs, curated evidence, planner outputs |
| Remote code/results repo | `ust_ip:/data/chenqingzhan/FastGen` | Actual FastGen code, modified experiment configs/scripts, logs, outputs, checkpoints |

Important remote paths:

- FastGen code: `/data/chenqingzhan/FastGen`
- Logs: `/data/chenqingzhan/logs`
- Output root: `/data/chenqingzhan/fastgen_output`
- Wan DMD2 output root observed on `2026-07-03`: `/data/chenqingzhan/fastgen_output/fastgen/wan_dmd2`

## How To Start A New Research Planner Agent

1. Open a new agent in this repository root:
   `/Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation`
2. Paste the full content of `research/planner_startprompt.md`.
3. The new agent should read local docs, do a read-only remote status check, then summarize its understanding and wait for user confirmation.
4. Only after confirmation should it write `research/T0_project_analysis.md` or generate T1-T4 research task prompts.

## Method Files

- `research/workflow.md`: workflow, roles, quality mechanisms, common failure modes.
- `research/task_brief_template.md`: content-agent task prompt template and planner-agent report acceptance checklist.
- `research/planner_startprompt.md`: project-specific startup prompt for the next planner agent.

## Local Context To Read First

For research planning, use this local read order:

1. `research/README.md`
2. `research/workflow.md`
3. `research/task_brief_template.md`
4. `README.md`
5. `CLAUDE.md`
6. `agents/README.md`
7. `experiments/README.md`
8. `03-dmd-distillation/OVERVIEW.md`
9. `03-dmd-distillation/HANDOFF.md`
10. Recent active reports, especially:
    - `reports/2026-06-29-lightning-progressive-adversarial-distillation-report.md`
    - `reports/2026-06-29-progressive-distillation-core-path-report.md`
    - `reports/2026-06-29-progressive-distillation-literature-support.md`
    - `reports/2026-06-29-progressive-distillation-paper-reading.md`
    - `reports/2026-06-25-progressive-distillation-paper-list.md`
    - `reports/2026-06-17-wan-dmd2-openvid-progress.md`

Use `archive/` only for background. Do not treat archived status as current without re-checking the server.

## Remote Snapshot From 2026-07-03

This snapshot is only a starting point. Re-check it at the beginning of a new session.

Read-only command used:

```bash
ssh ust_ip "cd /data/chenqingzhan/FastGen && git rev-parse --short HEAD && git status --short | head -40"
```

Observed:

- Hostname: `RTX-5090-32G-X8`
- Remote branch: `main`
- Remote GitHub remote: `git@github.com:Tonkmy/FastGen.git`
- FastGen commit: `e66f6c6`
- Remote worktree is dirty.
- Recently modified tracked/untracked files include:
  - `experiment/2026-06-17-wan-dmd2-openvid-progress/2026-06-17-wan-dmd2-openvid-progress.md`
  - `fastgen/configs/experiments/WanT2V/config_dmd2.py`
  - `fastgen/configs/experiments/WanT2V/config_dmd2_smoke.py`
  - `fastgen/configs/experiments/WanT2V/config_dmd2_step8_2k.py`
  - `fastgen/configs/experiments/WanT2V/config_dmd2_step4_from_step8_2p5k.py`
  - `fastgen/configs/experiments/WanT2V/run_infer_dmd2_step8_freq_eval5.sh`
  - `fastgen/configs/experiments/WanT2V/run_train_dmd2_step4_from_step8_2p5k.sh`
  - `fastgen/configs/experiments/WanT2V/run_infer_dmd2_step4_from_step8_8node_eval10.sh`
  - `scripts/reports/build_wan_dmd2_multi_experiment_report.py`

This matters because older local handoff docs mention `FastGen` commit `34f30e8` and a Wan2.2 TI2V / WanI2V route. The newer remote state and recent reports point strongly to a WanT2V / progressive DMD2 line. A new planner must reconcile that difference with the user before making research claims.

## First Planner Deliverable

Before writing T0, the planner agent should give the user a short confirmation summary:

1. Current method in paper language.
2. Evidence split: proved with numbers, plausible but unproved, stale or conflicting.
3. Current mainline ambiguity: Wan2.2 TI2V/WanI2V handoff versus WanT2V progressive 50 -> 8 -> 4 reports.
4. Proposed T0 acceptance target and what will not be touched.

Then stop and wait for confirmation.
