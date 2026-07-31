# DATA_SOURCES — final-report figures (2026-07-28)

Every number drawn in `figN_*.pdf/.png` was transcribed verbatim from the frozen
local source files below (no remote access, no recomputation, no from-memory
values). Line numbers refer to the files as of 2026-07-28. Regenerate any figure
with `uv run python3 figN_<slug>.py` from inside `figures/`.

Abbreviations: `07-14` = `experiments/results/2026-07-14-e0-full-table-g1.md`,
`07-20` = `experiments/results/2026-07-20-g2-relay-vs-direct-final.md`,
`07-24` = `experiments/results/2026-07-24-e2a-fulltable-ch3.md`,
`07-25` = `experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md`,
`ch2` = `research/thesis_ch2_draft.md`,
`acc` = `experiments/results/acceptance-log.md`.

---

## F1 `fig1_relay_design` (schematic, no data axes)

| Drawn element | Value / wording | Source |
|---|---|---|
| Teacher node | 50-step, CFG | ch2 L12 / 07-20 L12 ("teacher 50-step CFG5") |
| 8-step intermediate student, `2500 iters` | W5, 2500 iter | ch2 L12 |
| Re-init marker ("reset all but generator", revision 2026-07-31 after reader review: old "full re-init" contradicted the note) + note | "generator weights only; optimizer / fake score / discriminator reset" | ch2 L12 ("仅继承 W5@2500 生成器权重;优化器/fake score/判别器重置") |
| 4-step relay student, `2500 iters` | W7, 2500 iter | ch2 L12 |
| 4-step direct student ×2 arms, `5000 iters each; low / high LR` | E1a LR 5e-6 (low) / E1b LR 1e-5 (high), 5000 iter each | ch2 L13 |
| Invariants bar: data · budget · 4-step timestep list · discriminator · evaluation protocol | matched-invariants list | ch2 L14 (data / 4-step t_list / discriminator structure + hyperparams / eval protocol) + ch2 L12-13 (budget 5000 vs 5000) |
| Badge "pre-registered: parity expected → observed: direct slightly ahead" | pre-registration outcome | 07-20 L25 ("预注册…原预期'相当',实测直蒸略优") + ch2 L3/§2.1 framing |

## F2 `fig2_gan_bifurcation` (aesthetic vs iteration, 3 arms, q150 seed0)

| Series | Values @500/1000/1500/2000/2500 | Source |
|---|---|---|
| GAN off (E2a) | 0.5908 / 0.5921 / 0.5984 / 0.6109 / 0.6074 | 07-24 L9-L13, `aes` col |
| GAN on · independent (t,ε) (E2b) | 0.5774 / 0.5670 / 0.5477 / 0.5471 / 0.5487 | 07-25 L9-L13, `aes` col |
| GAN on · shared (t,ε) (W7, relay reference) | 0.5768 / 0.5592 / 0.5433 / 0.5477 / 0.5379 | 07-14 L11-L15, `aes` col (see sourcing note below) |
| Labeled points | peak 0.611 (=0.6109 @2000, E2a); ends 0.607 / 0.549 / 0.538 (3-decimal renderings of 0.6074 / 0.5487 / 0.5379) | same rows as above |
| "same init (8-step intermediate)" | all three arms start from W5 relay init, single-variable configs | acc L23 (row #11: E2a/E2b config.yaml differ from W7 by exactly one key each, relay init shared); ch2 L12 |

**Sourcing note (deviation):** the task designates 07-25 as the source for the
shared-pairing (relay reference) series, but 07-25 contains only @500 (0.577)
and @1000 (0.559) in prose (L19). The full 5-point W7 series was therefore
transcribed from the 07-14 E0 table (same q150 seed0 protocol) and
cross-checked against 07-25 L19 and 07-20 L14-15 (0.577/0.559 match after
rounding). No value comes from memory.

## F3 `fig3_dd_diversity_panels`

Final arbiter per task spec: 07-20 champion table (values also present in 07-14; cross-checked).

| Drawn element | Value | Source |
|---|---|---|
| (a) teacher dynamic degree bar | 0.625 | 07-20 L12 (`DD_clean`) |
| (a) student band endpoints | 0.750 – 1.000 | 07-20 L13-L17: E1a@1000 0.750, W7@500 0.825, W7@1000 1.000, E1b@500 0.975, W1@1000 0.950; + W5@2500 0.950 (07-14 L75, coincides with W1 tick) — revision 2026-07-28 to match report §6.1 ("the 8-step intermediate alike") |
| (b) teacher diversity bar + ▼ | 0.732 | 07-20 L12 (`diversity`) |
| (b) student band endpoints (ORANGE) | 0.586 – 0.649 | 07-20 L13-L17: W7@500 0.598, W7@1000 0.613, E1b@500 0.628, E1a@1000 0.635, W1@1000 0.649; + E2a@2000 0.5860 (07-24 L12), E2b@500 0.590 (07-25 L9), W5@2500 0.5949 (07-14 L19) — revision 2026-07-28: band extended to match report §6.1 range 0.586–0.649 (all arms incl. ablations and the 8-step intermediate) |
| (a) footnote "motion smoothness 0.97+ …" | 0.97+ verified | 07-14 L79 ("motion_smoothness 全员 0.970-0.987"); dm40 table L64-L75 motion_smooth col 0.9705-0.9868 |
| (a) header "not degraded" / note "higher, not lower" | all students ≥ teacher | 07-20 L23 ("全部臂 DD_clean 0.75-1.0 ≥ teacher 0.625,无一坍缩") |
| (b) header "consistently lower" | both direct arms and relay below teacher | 07-20 L22 |

## F4 `fig4_sweep_curves`

| Series / element | Values | Source |
|---|---|---|
| relay student (W7), 5 pts @500-2500 | 0.5768 / 0.5592 / 0.5433 / 0.5477 / 0.5379 | 07-14 L11-L15 `aes` col (same sourcing note as F2) |
| direct student (E1a), 10 pts @500-5000 | 0.5243 / 0.5665 / 0.5523 / 0.5609 / 0.5385 / 0.5219 / 0.5327 / 0.5281 / 0.5226 / 0.5196 | acc L25 (row #12, parenthesized E1a aes full trajectory, verbatim) |
| labels 0.577 (relay start/peak), 0.538 (relay end), peak 0.567 (=0.5665 @1000), 0.524 (direct start), 0.520 (direct end) | 3-decimal renderings of the above | same rows |
| light-orange band 500-1000 "quantitative optimum" | quality optimum @500-@1000 | 07-14 L28 ("质量最优在 @500-@1000") |
| orange ring @2500 "subjective pick (end of decline)" | eyeball-selected ckpt = @2500; aes monotonically declining by then | 07-14 L15 ("@2500(肉眼最佳)") + L28 (aesthetic 0.577→0.538 单调下降) |

**Caveat (recorded per task):** the direct-arm 10-point trajectory is a
single-seed (seed0) sweep; per acc L25 it may be cited only as a *pattern*
("peaks around @1000 then broadly declines, not strictly monotonic"), not as
attribution evidence — over-training may contribute; the isolated-variable
evidence for the GAN mechanism is E2a only (F2). Figure caption in the report
should carry this framing.

## F5 `fig5_attribution`

| Drawn element | Value | Source |
|---|---|---|
| teacher bar + ▼ | 0.732 | 07-20 L12 |
| student band endpoints (ORANGE) | 0.586 – 0.649 | low end: 07-25 L20 ("三臂全部 0.586-0.613" — three-arm interval low, = E2a@2000 0.5860 per 07-24 L12); high end: 07-20 L17 (W1@1000 0.649) |
| band member ticks | 0.586, 0.590, 0.598, 0.613, 0.628, 0.635, 0.649 | 0.586 = three-arm interval low (07-25 L20); 0.590 = E2b@500 (07-25 L9, its best-of-sweep arm champion per 07-25 L25); 0.598 / 0.613 = W7@500/@1000 (07-20 L14-15); 0.628 = E1b@500 (07-20 L16); 0.635 = E1a@1000 (07-20 L13); 0.649 = W1@1000 (07-20 L17) |
| box 1 "Step-count relay — ruled out" ("relay vs. direct: collapse in both") | both arms below teacher | 07-20 L22 (direct 0.635/0.628 and relay 0.598/0.613, all < 0.732) |
| box 2 "GAN discriminator branch — ruled out" ("on vs. off: collapse unchanged") | diversity insensitive to GAN switch | 07-25 L20 (ruling 4); 07-24 L21 |
| box 3 "Shared (t,ε) pairing — ruled out" ("shared vs. independent: unchanged") | diversity insensitive to pairing | 07-25 L19-L20 (rulings 3-4) |

**Band composition:** min-max over all student arms at their best-of-sweep
checkpoints, cross-checked over both designated files; global endpoints happen
to be champion checkpoints themselves (E2a@2000 = 0.586 low, W1@1000 = 0.649
high). The three-arm sweep-wide interval (0.586-0.613) is fully contained.
Label fixed 2026-07-31 (reviewer catch): "all ablation arms" → "all arms"
(the band includes non-ablation arms W1/W5/E1a/E1b).

## F6 `fig6_seed_grid` (image grid, no numeric data)

| Element | Source |
|---|---|
| Top row, seeds s0-s7 in order | `slides/covers/p3v2_teacher_s0.png` … `p3v2_teacher_s7.png` (16 files verified present 2026-07-28) |
| Bottom row, seeds s0,s2,s3,s4,s6,s8,s9,s11 in order | `slides/covers/p3v2_e1a_s{0,2,3,4,6,8,9,11}.png` |
| Per-cell seed tags | real seed number from each filename |
| Row labels | wording per task spec; teacher = 50-step CFG, student = 4-step (E1a) |

## MISSING list

None. All task-specified annotations were sourced. One sourcing substitution
(not a missing value) is recorded under F2: the full 5-point shared-pairing
series comes from 07-14 rather than 07-25 (which holds only 2 of the 5 points),
with cross-checks against 07-25 L19 and 07-20 L14-15.
