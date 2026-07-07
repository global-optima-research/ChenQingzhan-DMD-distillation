# Research-Driven Experiment Plan

- Created: 2026-07-06 by research planner (adoption decisions from T1 acceptance); updated 2026-07-06 after T2 acceptance + planner code check.
- Rule: every adopted item cites its source task. Execution scheduling and GPU allocation are decided by the user; the planner does not launch training. All runs go through the `experiments/` config discipline (one config, one log, one output, one result note).
- Quality metrics protocol (T1 §2.4 + T2 §2.2): daily ablations use VBench 6 quality dimensions + CD-FVD (not I3D-FVD) + optional JEDi; main table uses full VBench (Total/Quality/Semantic); human eval side-by-side vs teacher 50-step per T2VHE. Always report Dynamic Degree and cross-seed diversity (same prompt, N seeds, LPIPS/DINO feature distance) — TMD showed VBench totals cannot detect mode collapse; FVD is strictly monotone in anchor-position sweeps (Lip Forcing) and serves as the primary ranking metric for shape scans.

## Adopted (priority order)

| P | Experiment | Purpose / axis | Source | Status |
|---|---|---|---|---|
| P0 | Quantitative eval of existing checkpoints: 4-step baseline `0001000`, 8-step lr_original `0002500`, second-round 4-from-8 `0000500`-`0002500` (all 5), plus `_step8_normalize` / `_step8_freq` best ckpts. VBench 6-dim subset + CD-FVD; JEDi optional. No new training. | Convert all visual-only conclusions to numbers; prerequisite for every claim | T1 §6.3-4, §2.4 | Adopted, awaiting user scheduling |
| P1a | LR × batch de-confound: fix two of {LR, effective batch, GPU count}, sweep the third; minimum 2-3 runs around the second-round 4-from-8 recipe (`5e-6`/16 vs `1e-5`/16 vs `5e-6`/12) | Axis C hard prerequisite; fixes the T0 §0.3 confound | T0 §0.3 + T1 §6.3-3 | Adopted, awaiting user scheduling |
| P1b | 50→4 direct distillation vs 50→8→4 relay, same budget / data / t_list, only the path differs | Axis B life-or-death ablation (neither GPD nor CoDMD reports it) | T1 §6.3-1 | Adopted, awaiting user scheduling |
| P1c | 4-step relay stage: GAN weight {0, 0.03} × approximate-R1 {on, off}; primary metrics Dynamic Degree + cross-seed diversity. R1 is already implemented in FastGen (`gan_r1_reg_weight`, default alpha 0.1 ≈ APT's video σ) — **config-only, zero code** | Discriminator keep/drop decision (Data-Forcing same-base warning: adding GAN dropped Dynamic Degree 0.500→0.375); supplies the GAN on/off evidence One-Forcing lacks; prices T2 weakness 1 | T2 §6.1 + planner code check 2026-07-06 | Adopted, awaiting user scheduling |
| P2 | t_list shape matrix (4-step, t=σu/(1+(σ−1)u)): uniform (negative control, TMD predicts collapse) / σ=3 / σ=5 (baseline) / σ=8 (=SGMD's silent config) / σ=12 (cliff check, FlashMol warning) / **non-nested σ=5 control** (relay-exclusive ablation no prior work has). Second factor per TMD: training-t shift {3,5,10} × anchor σ=5. `_step8_normalize` (uniform, 8-step) is an existing data point. Careful: lightx2v nominal `[1000,750,500,250]` must be converted through shift-5 warp before comparison | Axis A | T1 §6.3-2 + T2 §2.2 | Adopted, awaiting user scheduling |

## Candidates (decide after T3 / after P0 numbers)

- **IDA for the 4-step relay stage** (SenseFlow: fake ← 0.97·fake + 0.03·student after each student update; +0.6~4%/iter): natural fit for the fake-score-reset window; small code change; can be framed as a relay-protocol component. Source: T2 §6.1-2.
- **Data-Forcing style post-training** on the best 4-step ckpt (100-300 iter, teacher score input swapped to real latents w.p. 0.5; same-base evidence: camera trajectory diversity +349%). Source: T2 §6.1-3.
- **Discriminator head spatio-temporal upgrade** (learnable-query cross-attn to video-wise logit, or per-layer 1D temporal branch; POSE +16.3 / AAD-1 frame-wise→static / SF-V FVD 180.9 vs 514.7). Medium cost. Source: T2 §6.1-4.
- **Same-ε vs independent-ε ablation** (`gan_use_same_t_noise` flag; method default is False, our config sets True): config-only; unique ablation point no paper or known repo reports (T3 to confirm repo scan). Source: T2 §4.2 + planner code check.
- **ISG intra-segment relay** (SenseFlow) if high-noise anchors (0.999/0.937/0.833 all in the volatile t∈[0.8,1.0] band) prove under-supervised. Source: T2 §6.1-5.
- Training-t high-noise coverage check (Phased DMD nested-interval finding). Source: T1 §6.3-5.
- Teacher CFG annealing / reduction in the 4-step stage (GPD annealed 6.0→1.5; Few-Step SiD Zero-CFG line; CoDMD uses 3.5). Source: T1 §6.3-5 + T2 §3.3.
- CoDMD relational regularizer as add-on (zero extra networks). Source: T1 §6.3-5.
- Conditional triggers: Salt SC regularizer if "more steps get worse" reappears in 8-step; Flash-DMD high/low-noise loss split if GAN/DM gradients conflict; fake-EMA + TTUR downshift if training cost becomes binding. Source: T2 §6.1.

## Explicitly not planned now

- 1-step / 2-step students; new base models or datasets; ADM-style adversarial main objective or full VFM discriminator replacement (story drift, T2 §6.1); anything claiming "8-step stage is necessary" before P1b runs.
