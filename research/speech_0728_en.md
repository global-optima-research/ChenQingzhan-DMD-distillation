# 2026-07-28 Presentation Script (English, 510-second execution version)

> Companion deck: `slides/DMD2_report_0728_V2.pptx` (compact-session timing, commit 88e0a5f).
> Pace baseline: ~140 words/min; cumulative time marks per page.
> Rules: no numbers beyond what is on the slides (spoken approximations allowed — exact values are on screen); banned wording — "significant", any claim of "surpassing the teacher", all internal project codenames.
> Emergency rule: if past 7:00 when P5 ends, deliver only the first and last sentences of P7.

---

## P0 | Cover (10s) [cum. 0:10]

Good afternoon, professors. I am Chen Hing Chin, student ID 21205180. My topic today: few-step distillation of Wan2.1-T2V — diagnosing and attributing the quality degradation behind a 25× speed-up.

[Advance; P1 videos autoplay]

## P1 | Task and Question (60s) [cum. 1:10]

On screen are two videos from the same prompt. Left: the 50-step teacher — 165 seconds per clip. Right: our distilled 4-step student — about six and a half seconds. Roughly 25× faster, consistent with the reduction in compute.

[Speak over the video after ~5 seconds]

The training framework and all hyperparameters come from NVIDIA FastGen's public configuration. Our own work is twofold: first, a "step-count relay" training schedule — train an 8-step intermediate model, then hand over to a 4-step stage; second, a systematic degradation audit, which is today's main thread.

At first glance the two videos look close. But "looks fine" is not the same as "is fine". Today I answer one question: after a 25× speed-up, what exactly did we lose — and which component is responsible?

[Advance]

## P2 | Why Not Trust the Eye (50s) [cum. 2:00]

First, why subjective judgement is not enough.

[Point to left chart] This is the relay student's aesthetic quality over training: it declines monotonically. Yet the checkpoint we had picked by visual inspection was step 2500 — the end of the decline; the quantitative optimum lies between steps 500 and 1000. On the other training line, visual inspection picked too early. Two lines, wrong in opposite directions — the unreliable thing is subjective selection itself.

[Point to right table] And an extreme counterexample: a collapsed model, almost static output — yet it scores highest of all on consistency metrics. Judged by those alone, a broken model would rank first.

Hence our four-component protocol, and one rule: checkpoint every 500 steps, evaluate all of them, take the best, then re-test it with three random seeds.

[Advance]

## P3 | Core Finding (75s) [cum. 3:15; hard checkpoint ≤3:30]

Measured with this instrument, here is the central finding of the project.

Start with what the field worries about — motion. [Point to lower-left panel] On 40 motion-oriented prompts, the students' dynamic degree is not below the teacher's, and motion smoothness above 0.97 rules out jitter artifacts. The feared "few-step models stop moving" did not appear in our setting.

What is consistently lost is diversity. [Point to the wall] Same prompt, only the random seed changes. Top row — the teacher: eight seeds, eight different compositions. Bottom row — the student: eight seeds converging to nearly one template. [Point to lower-right panel] The numbers agree: teacher 0.732; every student drops to 0.59–0.64 — roughly a 13 to 19 percent loss. Weak or strong recipe, direct distillation or relay — no exception. This is the most-replicated finding of the project.

So who is responsible? Two controlled ablations.

[Advance]

## P4 | Ablation 1: the Step-Count Relay (75s) [cum. 4:30]

The first ablation targets our own relay design.

[Point to design box] The comparison is fully matched: 5,000 iterations total for the relay path, 5,000 for direct distillation; same data, same recipe, same evaluation; and two direct-distillation runs at high and low learning rates, so "the baseline wasn't tuned" does not apply. Our pre-registered expectation was a tie.

[Point to main table] The result: direct distillation is slightly ahead. Quality is essentially tied — on the imaging metric the direct student is actually the highest among all students. On diversity, both direct runs are above both relay checkpoints, same direction. So the relay does not mitigate the diversity collapse; it deepens it.

[Point to bar chart] The relay's one robust, real difference is motion amplitude: about 1.9 times the direct student's, consistent across all four seeds, and above the teacher reference line. Whether that is a merit depends on your criterion — we take no side. But it is a lead: what maintains this motion?

[Advance]

## P5 | Ablation 2: the GAN Branch (105s) [cum. 6:15; hard checkpoint ≤6:30]

The answer lies in the second ablation: the discriminator branch — the GAN term in the recipe.

The design is clean: same 8-step intermediate model as the starting point, same recipe, one field changed at a time. Arm one: switch the GAN off entirely. Arm two: change only the discriminator's pairing convention.

[Point to center chart] This is the most important figure of the talk. X-axis, training iterations; y-axis, aesthetic quality. With the GAN off, quality climbs throughout training. With the GAN on — both arms — quality declines. The "early peak, then decline" left open on page 3 gets its candidate explanation here: switch the GAN off, and the phenomenon disappears and reverses; the best checkpoints re-tested on three seeds agree in direction. I say "candidate" because we tested a single GAN weight, on the relay path only.

Second conclusion: motion amplitude is mainly maintained by the GAN. Off — motion falls back to the teacher's level. On — it rebuilds from a low point up to 4.7. Yet the direct student, with the GAN on, still moves little — the effect interacts with the relay initialization, so we do not claim "GAN always increases motion".

Third: the widely copied default pairing — the discriminator sees real and fake corrupted with the same noise and timestep. To our knowledge it had never been tested in a controlled way. We tested it: across five checkpoint pairs the differences are all within 0.011 — no detected effect.

[Point to bottom band] And keep this line in mind: diversity across all three arms sits within 0.586–0.613 — insensitive to the GAN switch and to the pairing.

[Advance]

## P6 | Closing the Attribution (45s) [cum. 7:00]

Put the three checks together. Replace the relay route — the collapse remains, even deeper. Switch off the GAN — it remains. Change the pairing — it remains.

Among every component we tested, none is the source: the diversity collapse comes from the distribution-matching distillation itself.

We have not solved it. But we have turned "the model seems worse" into a precisely measured problem whose suspected sources have been excluded under control. That is the open problem we leave with this direction.

[Advance]

## P7 | External Benchmark (45s) [cum. 7:45; if over time, first and last sentences only]

On the standard VBench benchmark — 946 prompts, 5 seeds each, 12 dimensions — the conclusion: no dominant model.

[Point to right column] Our primary direct student leads on consistency, smoothness and flicker; the relay student leads on dynamic degree and action semantics; the GAN-off ablation model has the best static image quality with motion intact — consistent, across datasets, with the ablation findings.

And the diversity champion is still the teacher — a dimension VBench does not measure. Which is exactly why we built our own protocol.

[Advance]

## P8 | Conclusions (45–50s) [cum. 8:30–8:35]

Four take-aways. Diagnosis: the main degradation is cross-seed diversity, not motion. Attribution: the quality decline and the motion amplitude relate mainly to the GAN branch; the pairing convention shows no detected effect; the diversity collapse comes from distillation itself — an open problem. Method: a lightweight, seed-controlled degradation audit protocol. Engineering: a 25× speed-up, and — to our knowledge — the first controlled relay-versus-direct comparison on this base model.

Limitations, stated plainly: no human evaluation; four benchmark dimensions missing; the R1-regularization arm untested due to 32-gigabyte memory limits; one training run per configuration.

[Point to bottom band] One line to remember: 165 seconds down to six point six; diversity 0.732 down to around 0.6; three components excluded under control.

Thank you. The thesis will be submitted on July 31st.

---

### Rehearsal notes
- Pass = two consecutive timed runs within 510s. Checkpoints: P3 done ≤3:30, P5 done ≤6:30, P7 done ≤7:45.
- P3 through P6 is one continuous argument — no pauses between pages. P5 is the longest page; take it calmly and reclaim time on P7.
- Videos: play only on P1 (~5s); walk through P3/P4 on static thumbnails.
- Post-session questions: fallback answers live in backup pages B1–B6 and the speaker notes (seed replacement, composite score, the three practical recommendations).
