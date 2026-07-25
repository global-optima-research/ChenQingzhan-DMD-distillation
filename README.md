# Wan2.1-T2V DMD2 Progressive Distillation Workspace

Core goal: distill the 50-step `Wan2.1-T2V-1.3B` teacher into a **high-quality 4-step student** with DMD2, using a staged step-count relay (`50 -> 8 -> 4`): the best 8-step intermediate checkpoint initializes the 4-step stage (generator weights only; optimizer / fake score / discriminator reset).

## Current State (2026-07-13)

- Milestones: progress report 2026-07-28; final thesis submission 2026-07-31 (updated 2026-07-21; the earlier 07-21 defense date is obsolete).
- Canonical experiment record: `reports/experiment-report-wan21-t2v-dmd2-progressive.md` — runs W1-W7; all quality conclusions are still visual-only, quantitative evaluation (E0) is the next step.
- Verified facts and evidence grading: `research/T0_project_analysis.md`.
- Research pipeline: T0-T3 accepted (T3 novelty adjudication governs claim wording); T4 pivoted to the thesis itself; paper-mainline candidate gated on E5 + GAN-0 results (`research/idea_mainline_candidate.md`).
- Active plan (`research/experiment_plan.md`, 2026-07-11 sprint edition; supersedes the earlier P0-P2 ladder): E0 quantification of existing checkpoints, E1a/E1b 50->4 direct baselines vs W5+W7 relay, E2a-c discriminator audit (+E2d gated), E5 offline probe. Not started as of 2026-07-13 (remote idle since 2026-06-25).
- Remote: `ust_ip:/data/chenqingzhan/FastGen` (branch `main`, `94a4517`); outputs under `FASTGEN_OUTPUT/fastgen/wan_dmd2/`; entry-script runbook at `fastgen/configs/experiments/WanT2V/README.md`.

## Quick Start

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
bash experiments/bin/run_remote_script.sh --dry-run experiments/configs/wan21_dmd2_step4_relay_eval10.env
bash experiments/bin/run_remote_script.sh experiments/configs/wan21_dmd2_step4_relay_eval10.env
```

## Read Order (new session)

1. `README.md`
2. `reports/experiment-report-wan21-t2v-dmd2-progressive.md`
3. `research/T0_project_analysis.md`
4. `experiments/README.md`

Research-planning sessions start from `research/planner_startprompt.md` instead.

## Directory Map

| Path | Purpose |
|---|---|
| `experiments/` | One-line submission layer: `bin/`, `configs/`, `results/`, `tools/` |
| `reports/` | Canonical experiment record + frozen June artifact index |
| `research/` | Research pipeline: T0 analysis, T1-T4 briefs and reports, experiment plan, `paper/` PDFs |
| `docs/` | FastGen framework manual |
| `agents/` | Agent roles and operating rules |
| `artifacts/` | Curated local evidence (small, git-friendly) |
| `archive/` | Historical material only, incl. `archive/wan22-ti2v-line/` (the 2026-04/05 Wan2.2 TI2V 5B line) |

## Operating Rules

- One experiment = one `experiments/configs/*.env` + one remote run dir + one `experiments/results/` note; always GPU-precheck and dry-run before launching.
- Change one major factor per experiment; new variants get a new remote python config with a new `log_config.name`.
- Quality claims only via the quantitative protocol (VBench 6-dim subset + CD-FVD + Dynamic Degree + cross-seed diversity; full VBench + human eval for main tables).
- Method wording: "step-count relay" / "progressive step reduction" — never "phased DMD" or "progressive distribution matching" (names taken).
- Use absolute dates. Archived material is context only.
