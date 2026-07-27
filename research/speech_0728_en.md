# 2026-07-28 Presentation Script (English, simple spoken version, 510s)

> Companion deck: `slides/DMD2_report_0728_V2.pptx` (compact-session timing, commit 88e0a5f).
> Style: short sentences, common words, subject-verb-object. Made for a non-native speaker. Pace ~120–125 words/min — do not rush; pause briefly at every [bracket].
> Rules: no numbers beyond what is on the slides; banned wording — "significant", any claim of "surpassing the teacher", all internal project codenames.
> Emergency rule: if past 7:00 when P5 ends, say only the first and last sentences of P7.

---

## P0 | Cover (10s) [cum. 0:10]

Good afternoon, professors. I am Chen Hing Chin, student ID 21205180. My topic: few-step distillation of Wan2.1 text-to-video — we make it 25 times faster, and we study what quality is lost, and which part causes it.

[Advance; P1 videos autoplay]

## P1 | Task and Question (60s) [cum. 1:10]

Here are two videos from the same prompt. Left: the teacher model, 50 steps, 165 seconds per video. Right: our student model, only 4 steps, about 6.5 seconds. That is about 25 times faster.

[Speak over the video after ~5 seconds]

The training framework and all hyper-parameters come from NVIDIA FastGen, a public repository. Our own work has two parts. Part one: a training schedule called "step-count relay" — first train an 8-step model, then use it to start a 4-step model. Part two: a careful study of the quality loss. That is today's main story.

The two videos look similar. But "looks fine" does not mean "is fine". So, one question today: after the 25-times speed-up, what did we lose — and which part of the method caused it?

[Advance]

## P2 | Why Not Trust the Eye (50s) [cum. 2:00]

First: why we cannot just trust our eyes.

[Point to left chart] This curve is the quality of the relay student during training. It keeps going down. But by eye, we had picked step 2500 — the end of the curve. The true best point is between step 500 and 1000. On another training run, we picked too early. Two runs, two mistakes, opposite directions. So the problem is "picking by eye" itself.

[Point to right table] And an extreme example: this model is broken — its output is almost a still image. But its consistency scores are the highest of all. If we only look at consistency, a broken model wins.

So we built a four-part evaluation protocol, with one rule: save a checkpoint every 500 steps, test all of them, pick the best, then re-test it with three random seeds.

[Advance]

## P3 | Core Finding (75s) [cum. 3:15; hard checkpoint ≤3:30]

With this protocol, here is our main finding.

First, motion. [Point to lower-left panel] Many papers worry that few-step models stop moving. Not in our case. On 40 motion prompts, the students move as much as the teacher, or more. And the motion is smooth — real motion, not shaking.

What we really lose is diversity. [Point to the wall] Same prompt, only the random seed changes. Top row, the teacher: eight seeds, eight different scenes. Bottom row, the student: eight seeds, almost the same scene every time. [Point to lower-right panel] The numbers agree: the teacher scores 0.732; every student drops to 0.59 to 0.64 — about 13 to 19 percent lower. Weak recipe, strong recipe, direct, relay — all of them drop. This is the most repeated finding in our project.

So, which part causes it? We run two controlled ablations.

[Advance]

## P4 | Ablation One: the Step-Count Relay (75s) [cum. 4:30]

Ablation one: our own relay design.

[Point to design box] The comparison is fair. The relay path gets 5000 iterations in total; direct distillation also gets 5000. Same data, same recipe, same evaluation. And we run direct distillation twice — high and low learning rate. Before the experiment, we registered our expectation: a tie.

[Point to main table] The result: direct is slightly better. Quality is basically a tie — on the imaging score, the direct student is even the highest of all students. On diversity, both direct runs beat both relay checkpoints. So the relay does not reduce the diversity problem. It makes it worse.

[Point to bar chart] The relay keeps only one real difference: the amount of motion. It moves about 1.9 times as much as the direct student, on all four seeds, and more than the teacher. Good or bad? It depends on what you want; we take no side. But it is a clue: what keeps this motion so high?

[Advance]

## P5 | Ablation Two: the GAN Branch (105s) [cum. 6:15; hard checkpoint ≤6:30]

The answer comes from ablation two: the GAN part — the discriminator.

The design is clean. Same starting point — the same 8-step model. Same recipe. We change one thing at a time. Arm one: turn the GAN off. Arm two: change only the pairing rule of the discriminator.

[Point to center chart] This is the most important figure today. X-axis: training iterations. Y-axis: aesthetic quality. With the GAN off, quality goes up, all the way. With the GAN on — both arms — quality goes down. Remember page 3: quality peaks early, then drops. Here is the candidate reason. Turn off the GAN, and the problem disappears — it even reverses. We re-tested the best checkpoints with three seeds; same direction. I say "candidate", because we tested only one GAN weight, and only on the relay path.

Second: motion. The GAN keeps the motion high. Turn it off — motion falls back to the teacher's level. Turn it on — motion climbs to 4.7. But the direct student also has the GAN on, and it still moves little. So the effect also depends on the relay initialization. We do not claim "GAN always increases motion".

Third: the pairing rule. By default, the discriminator sees the real and the fake sample with the same noise and the same timestep. Everyone copies this setting; as far as we know, nobody has tested it. We tested it. Across five checkpoint pairs, all differences are within 0.011. No effect detected.

[Point to bottom band] One more line: diversity in all three arms stays between 0.586 and 0.613. GAN on, GAN off, pairing changed — diversity does not move.

[Advance]

## P6 | Closing the Attribution (45s) [cum. 7:00]

Now put the three checks together. Change the training route — the collapse is still there, even worse. Turn off the GAN — still there. Change the pairing — still there.

None of the parts we tested is the cause. The diversity collapse comes from the distillation itself.

We did not solve it. But we turned a vague feeling — "the model seems worse" — into a precise, measured problem, with the main suspects ruled out. We leave it as an open problem.

[Advance]

## P7 | External Benchmark (45s) [cum. 7:45; if over time, first and last sentences only]

On the standard benchmark, VBench — 946 prompts, 5 seeds, 12 dimensions — no model wins everywhere.

[Point to right column] The direct student wins on consistency, smoothness and flicker. The relay student wins on motion and action. The GAN-off model has the best static image quality, and its motion is still fine — this matches our ablation results.

And the diversity champion is still the teacher. VBench does not measure diversity at all — that is exactly why we built our own protocol.

[Advance]

## P8 | Conclusions (45–50s) [cum. 8:30–8:35]

To conclude, four points. Diagnosis: the main loss is cross-seed diversity, not motion. Attribution: the quality drop and the amount of motion are mainly related to the GAN part; the pairing rule shows no detected effect; the diversity collapse comes from distillation itself — an open problem. Method: a light, seed-controlled evaluation protocol. Engineering: a 25-times speed-up, and — as far as we know — the first controlled relay-versus-direct comparison on this model.

Our limitations, honestly: no human evaluation; four benchmark dimensions are missing; one ablation arm was blocked by GPU memory; each configuration was trained only once.

[Point to bottom band] One line to remember: 165 seconds down to 6.6; diversity 0.732 down to about 0.6; three parts ruled out, under control.

Thank you. The thesis will be submitted on July 31st.

---

### Rehearsal notes
- Pass = two timed runs in a row within 510s. Checkpoints: P3 done ≤3:30, P5 done ≤6:30, P7 done ≤7:45.
- Speak slowly and clearly (~120 words/min). Short pause at every [bracket]. Do not add sentences that are not in the script.
- Hard words check (practice these): distillation / discriminator / diversity / iterations / checkpoint. If "attribution" feels hard, say "which part causes it" instead — same meaning.
- Videos: play only on P1 (~5s). Walk through P3/P4 on the still images.
- If a professor asks questions after the session, fallback answers are in backup pages B1–B6 and the speaker notes.
