# Wan2.1-T2V DMD2 Few-Step Distillation Workspace

Core goal: distill the 50-step `Wan2.1-T2V-1.3B` teacher into a **high-quality 4-step student** with DMD2, using a staged step-count relay (`50 -> 8 -> 4`): the selected 8-step intermediate checkpoint initializes the 4-step stage (generator weights only; optimizer / fake score / discriminator reset).

## Final Results (report submitted 2026-07-31, ARIN6900)

- **25x acceleration**: 165.24 s -> 6.59-6.66 s per 480p/81-frame video (50-step + CFG teacher vs 4-step guidance-free student).
- **Diagnosis**: the principal degradation axis is cross-seed diversity collapse (teacher 0.732 -> students 0.59-0.64, mean pairwise LPIPS), not the motion collapse the literature warns about.
- **Attribution**: quality's early peak-then-decline and the elevated motion amplitude both track the adversarial branch (candidate mechanism); the shared (t, eps) pairing convention shows no detectable effect; the diversity collapse survives every tested component and, within them, is attributable to the distillation objective itself (open problem).
- **Controlled comparison**: at matched budget the step-count relay brings no quality gain over one-stage direct distillation and retains less diversity; the deliverable model is the direct student at iteration 1000, selected under the preregistered protocol.

Final report sources: `latex/` (`main.tex` -> `main.pdf`, 25 pages). Paper figures are regenerable from frozen local records via `figures/` (`uv run python3 figN_*.py`). Presentation decks are built from `slides/` (CN) and `slides_en/` (EN).

## Quick Start (remote experiment submission)

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
bash experiments/bin/run_remote_script.sh --dry-run experiments/configs/wan21_dmd2_step4_relay_eval10.env
bash experiments/bin/run_remote_script.sh experiments/configs/wan21_dmd2_step4_relay_eval10.env
```

## Reading Order

1. `README.md`
2. `reports/experiment-report-wan21-t2v-dmd2-progressive.md` (canonical experiment record)
3. `research/T0_project_analysis.md` (verified facts and evidence grading)
4. `experiments/README.md` (how runs are submitted and recorded)

## Directory Map

| Path | Purpose |
|---|---|
| `latex/` | Final report sources (LaTeX, bibliography, compiled PDF) |
| `figures/` | Paper figures: matplotlib scripts, rendered PDFs/PNGs, per-number data provenance (`DATA_SOURCES.md`) |
| `experiments/` | One-line submission layer: `bin/`, `configs/`, `results/`, `tools/` |
| `reports/` | Canonical experiment record + frozen June artifact index |
| `research/` | Research reports: T0 analysis, literature/novelty studies (T1-T3), probe results, chapter drafts, storyline, `paper/` PDFs |
| `slides/`, `slides_en/` | Presentation deck build scripts (CN / EN) |
| `docs/` | FastGen framework manual |
| `artifacts/` | Curated local evidence (small, git-friendly) |
| `archive/` | Historical material only, incl. `archive/wan22-ti2v-line/` (the 2026-04/05 Wan2.2 TI2V 5B line) |

## Operating Rules

- One experiment = one `experiments/configs/*.env` + one remote run dir + one `experiments/results/` note; always GPU-precheck and dry-run before launching.
- Change one major factor per experiment; new variants get a new remote python config with a new `log_config.name`.
- Quality claims only via the quantitative protocol (VBench 6-dim subset + Dynamic Degree on motion prompts + cross-seed diversity + RAFT optical flow; full VBench for main tables).
- Method wording: "step-count relay" / "progressive step reduction" — never "phased DMD" or "progressive distribution matching" (names taken).
- Use absolute dates. Archived material is context only.
