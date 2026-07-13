# Experiment Plan — 11-Day Thesis Sprint

- Written: 2026-07-11 (D0). Defense: ~2026-07-22 (D11). Numbers freeze: D8 night. Draft freeze: D9-D10.
- Resource: one node, 8x RTX 5090 32G (`ust_ip`). All runs serial on this node; standing split where possible = train 6-8 GPUs / eval bursts between runs.
- Governing decisions (user-approved 2026-07-11): thesis plan A+B merged (Ch1 recipe+quantification, Ch2 relay-vs-direct, Ch3 discriminator audit), planner-corrected version; **2-arm direct baseline approved**; claims governed by T3 verdicts (`research/T3_novelty_adjudication.md`).
- Supersedes: the P0-P2 ladder of 2026-07-07 (axis-A t_list matrix cancelled per T3 §6.3-6; three-factor runs cut — per-anchor comparison becomes a free observation from W5 vs W7 data).

## Claim map

| Thesis part | Experiments | Surviving claim it supports (T3 wording) |
|---|---|---|
| Ch1 recipe + quantification | E0 | Relay recipe statement (orchestration-layer contribution only; upstream statement T3 §6.2 verbatim) + first quantified numbers for W1/W5/W7 |
| Ch2 relay necessity | E1a/E1b vs existing W5+W7 | "First controlled 50->4 direct vs 50->8->4 relay comparison on this base" (GPD/CoDMD/FastWan all lack it); neutral outcome pre-registered as reportable |
| Ch3 discriminator audit | E2a-c (+E2d gated), E5 probe | Claim D (paired vs independent (t,ε): no ablation in papers or public implementations — T3 upgraded); GAN on/off answers Data-Forcing's negative-GAN evidence; R1 prices T2 weakness 1 |

## Runs

| ID | What (one variable vs its reference) | Base / iter | GPUs | Est. wall | Queue slot |
|---|---|---|---|---|---|
| E0 | Eval tooling + quantification of existing ckpts: W7 x5, W1 best 2, W5 best, W3/W4 best, teacher subset | no training | 2 (gaps) | tooling 0.5-1 d; gen ~12 GPU·h | D0-D1 + bursts |
| E1b | Direct 50->4, recipe B = LR 1e-5 / batch 12 (matches W5-stage recipe), 5000 iter, init = teacher | `config_dmd2_smoke` variant | 6 | ~1.9 d | **launch D0 tonight** |
| E1a | Direct 50->4, recipe A = LR 5e-6 / batch 16 (matches W7-stage recipe), 5000 iter, init = teacher | same, variant | 8 | ~2.2 d | D2 -> D4 |
| E2a | GAN weight 0 (else exact W7 recipe: relay from W5 `0002500`, LR 5e-6/16, 2500 iter) | relay-stage variant | 8 | ~1.1 d | D4 -> D5 |
| E2b | `gan_use_same_t_noise=False` (fully independent t AND ε; flag flips both — wording must say "joint pairing") | relay-stage variant | 8 | ~1.1 d | D5 -> D6 |
| E2c | Approximate R1 on (`gan_r1_reg_weight` > 0, `alpha=0.1` default; calibrate weight once so `gan_loss_ar1` ~ same order as `gan_loss_disc`, then fix) | relay-stage variant | 8 | ~1.1 d | D6 -> D7 |
| E2d | GATED: same-t-independent-ε middle arm (~3-line patch in `dmd2.py::_compute_real_feat`) | relay-stage variant | 8 | ~1.1 d | only if G3 green |
| E5 | Offline probe on existing teacher/8-step/4-step ckpts: layer separability (15/22/29 vs alternates, matched t) + off-manifold measurement | no training | 1-2 (gaps) | ~0.5 d code + few GPU·h | D2-D3 gaps |

Fixed references: relay arm for Ch2 = existing W5 (2500 iter) + W7 (2500 iter) — total budget 5000 iter, hence E1 budget 5000. W1 (LR 1.25e-6/batch 8/6000) is a free third direct data point (weak recipe) — table row, no new run. Rationale for 2 direct arms: the relay path is recipe-heterogeneous (stage 1 = 1e-5/12, stage 2 = 5e-6/16); E1a/E1b bracket both, immunizing Ch2 against "untuned direct baseline".

## Day-by-day

| Day | Node (train queue) | Gaps / 2-GPU lane | Writing lane |
|---|---|---|---|
| D0 07-11 | prep all remote config variants + local .env + dry-runs; **launch E1b (6 GPUs) tonight** | E0 tooling install; start E0 generation backlog | thesis skeleton; Ch1 method + upstream statement (material exists in T0/T3) |
| D1 07-12 | E1b running | E0: existing-ckpt numbers done -> Ch1 tables; **G1** | related work from T1/T2/T3 |
| D2 07-13 | E1b ends ~midday; launch E1a (8 GPUs) | E1b ckpt sweep eval burst before E1a starts; E5 coding | Ch1 draft |
| D3 07-14 | E1a running | E5 probe compute deferred if no GPUs; CD-FVD/JEDi pipeline ready-check (else cut with limitation note) | Ch3 setup text |
| D4 07-15 | E1a ends; eval burst both E1 arms (8 GPUs, ~2 h); **launch E2a** | **G2: Ch2 verdict -> framing decision** | Ch2 results draft |
| D5 07-16 | E2a ends; launch E2b | eval burst E2a | Ch2 finalize |
| D6 07-17 | E2b ends; launch E2c | eval burst E2b; **G3: E2d go/no-go** | Ch3 results start |
| D7 07-18 | E2c ends (or E2d if swapped) | eval burst E2c | Ch3 draft |
| D8 07-19 | full VBench on final selected student (once, ~2 h gen on 8 GPUs + eval); E5 compute if pending | **numbers freeze D8 night** | results integration |
| D9-D10 | node idle (buffer for one rerun) | — | full draft + slides |
| D11 07-22 | — | — | defense |

Node budget: E1b 1.9 + E1a 2.2 + E2a-c 3.3 ≈ 7.4 node-days in a 8.5-day window — ~1 day slack for exactly ONE crash/rerun. Nightly discipline: check first-iteration metrics after every launch; checkpoint every 500 iter; a run that fails overnight costs the slack.

## Evaluation protocol (ablation-level)

- 6 VBench quality dims (subject/background consistency, motion smoothness, **dynamic degree**, aesthetic, imaging) on 150-200 fixed prompts; DD is a first-class metric (Data-Forcing/Phased DMD attack line).
- Cross-seed diversity: 40 prompts x 8 seeds, LPIPS + DINO feature distance (TMD: VBench totals cannot detect mode collapse).
- Ch2 head-to-head only: add CD-FVD or JEDi (~2k clips/arm) **if pipeline ready by D3**; otherwise cut with a limitation note.
- Full VBench (946 prompts x 5): final selected student ONLY, once, D8. Comparators cited from literature, not rerun: CoDMD 84.46 / teacher 83.69 / AnyFlow 83.54@4 / Causal-rCM 84.37 / Self-Forcing 83.76.
- Checkpoint comparison is best-of-sweep vs best-of-sweep (early stopping allowed on all arms), sweep grid = every 500 iter.
- Teacher generation only on the diversity subset + ~100-prompt 6-dim subset (165 s/video — full teacher sweeps unaffordable and unnecessary).

## Gates and fallbacks

- **G1 (D1)**: E0 sanity — if quantified W7 does not beat W1/W5 as the visual record claims, re-anchor Ch1/Ch2 wording before launching E1a (honest reporting; story survives either way).
- **G2 (D4)**: Ch2 verdict. Relay wins -> "recipe + necessity" framing. Tie/loss -> pre-registered neutral framing: "relay unnecessary at 1.3B/OpenVid scale under matched budget" + Ch3 carries novelty (claim D unique regardless).
- **G3 (D6)**: E2d go only if E2b moved DD/diversity/6-dim beyond noise AND queue is on schedule; otherwise E2d -> limitation/future work.
- Crash fallback order: drop E2c first, then E2d, then shrink E1 eval grid to {1000, 2500, 5000}; audit arms may stop at 2000 iter if behind (4 sweep points remain).

## Writing red lines (from T3, binding)

1. Upstream relation: use T3 §6.2 paragraph verbatim — ALL single-stage hyperparameters (t_list, GAN 0.03, layers 15/22/29, `gan_use_same_t_noise=True`, TTUR 1:5, CFG=5, lr) are NVIDIA FastGen public factory config; our contribution is orchestration only (relay schedule, 8-step intermediate stage, cross-stage reset, OpenVid instantiation).
2. `same_t_noise=True` is the upstream factory value — never "our deliberate choice" (T3 backfill item 2 corrects the earlier planner note).
3. Never "phased DMD" / "progressive distribution matching" (lightx2v productization makes this a hard collision); use "step-count relay".
4. Ch3 framing: "controlled audit of community-standard defaults", not "our design choices".
5. E5 claims: "to our knowledge" + documented search coverage; not a headline contribution.
6. t_list: recipe statement citing TMD's criterion; nested-anchor question is future work only.
7. Claim D wording: "first controlled ablation of the paired-(t,ε) discriminator input convention (papers and 12 public implementations scanned)"; do not imply the design is ours.
8. Free observation to include: per-anchor effective generator update count (W5 vs W7 at equal iter) — observation/discussion only, no causal claim.

## D0 checklist (today)

1. Remote: create 5 python config variants (E1a/E1b/E2a/E2b/E2c; one variable each, new `log_config.name`; E2d patch drafted but not applied), commit to FastGen.
2. Local: 5 matching `experiments/configs/*.env`; dry-run each through `run_remote_script.sh`.
3. Eval tooling on server: VBench custom-video 6-dim mode + diversity script (+ CD-FVD/JEDi if quick); smoke-test on one existing W7 checkpoint folder.
4. Apply T3 §6.1 backfill correction notes to T0/T1/T2 (writing-facing).
5. Launch E1b tonight after GPU precheck + dry-run; start E0 generation backlog on remaining 2 GPUs.

## Explicitly cut (11-day edition)

Layer-set training arm (E5 probe covers it observationally); CD-FVD beyond the Ch2 head-to-head; human eval (limitation note); full-VBench sweeps; three-factor attribution runs; homogeneous-recipe relay rerun; external VFM discriminator; perceptual regression term; any new infrastructure.
