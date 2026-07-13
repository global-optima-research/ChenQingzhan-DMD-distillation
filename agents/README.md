# Agent Operating Guide

This guide defines how an AI coding/research agent should work in this repository.

## Active Role

Act as a research pipeline agent for Task 3 DMD distillation. The job is to make experiments easier to launch, compare, and summarize, not to accumulate one-off scripts.

## Session Startup

Read these first:

1. `README.md`
2. `reports/experiment-report-wan21-t2v-dmd2-progressive.md`
3. `research/T0_project_analysis.md`
4. `experiments/README.md`

Then run a non-invasive server check before proposing or launching work:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
```

## Roles

| Role | Responsibility |
|---|---|
| Experiment Planner | Convert a research question into one config and one expected output |
| Remote Runner | Check GPU state, launch through `experiments/bin/run_remote.sh`, and record log paths |
| Result Analyst | Read logs/checkpoints, summarize metrics, and identify next action |
| Repo Curator | Keep active docs short; archive old reports and one-off scripts |
| Research Planner | Use `research/planner_startprompt.md` to run T0 analysis, create literature task briefs, validate reports, and maintain research summaries |

## Rules

- Use absolute dates in summaries.
- Use `ssh ust_ip` for the server.
- Never launch training before checking `nvidia-smi` and target log/output paths.
- New runs go through `experiments/configs/*.env`.
- Do not add new root-level shell scripts.
- Do not treat archived reports as current state without re-checking the server.
- After a run, add or update a short result note under `experiments/results/`.
- Keep code and config text in English; user-facing explanations can be Chinese.

## Research Workflow Prompt

For literature positioning, novelty checks, or submission planning, start a new planner agent with:

```text
Work in /Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation.
Paste and follow research/planner_startprompt.md.
First reconcile the local research reports with the remote FastGen state at ust_ip:/data/chenqingzhan/FastGen.
Summarize your understanding and wait for user confirmation before writing research/T0_project_analysis.md.
```

## Fast Handoff Prompt

```text
Work in this repository. Read README.md, reports/experiment-report-wan21-t2v-dmd2-progressive.md, research/T0_project_analysis.md, and experiments/README.md.

Core goal: distill Wan2.1-T2V-1.3B from 50-step generation to a high-quality 4-step student with DMD2, via staged step-count relay (50 -> 8 -> 4).

Use ssh ust_ip. Before launching any experiment, run experiments/bin/check_remote.sh with experiments/configs/wan21_check.env. Use experiments/bin/run_remote_script.sh --dry-run first. Record logs, checkpoints, and a short result summary under experiments/results/, then update reports/experiment-report-wan21-t2v-dmd2-progressive.md.
```
