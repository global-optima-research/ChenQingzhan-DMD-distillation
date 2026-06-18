# Wan2.2 I2V DMD2 65-Frame CFG Smoke

Date: 2026-04-29 CST

## Goal

Test whether native CFG training can fit on 2x RTX 5090 32GB after reducing temporal length from the original 81 frames.

## Baseline Problem

Original FastGen Wan2.2 I2V DMD2 config:

- `model.input_shape = [48, 21, 44, 80]`
- `dataloader_train.sequence_length = 81`
- `model.guidance_scale = 5.0`
- `dataloader_train.batch_size = 1` in our reduced-memory test
- 2 GPUs with FSDP CPU offload

Result: OOM at iteration 5, during DMD2 student update CFG negative teacher forward:

- failure site: `_apply_classifier_free_guidance -> teacher_x0_neg = self.teacher(...)`
- requested allocation: about `1.27 GiB`
- PyTorch allocated: about `30.21 GiB`
- process memory: about `31.18 GiB`

Log:

`/data/chenqingzhan/logs/wan22_dmd2_smoke_20260429_g14_cfg5_bs1_10iter_v2.log`

## Failed 49-Frame Test

Attempted:

- `model.input_shape = [48, 13, 44, 80]`
- `dataloader_train.sequence_length = 49`
- `model.guidance_scale = 5.0`

Result: invalid for the native DMD2 discriminator.

Error:

```text
RuntimeError: input image (T: 13 H: 44 W: 80) smaller than kernel size (kT: 16 kH: 16 kW: 16)
```

Reason: the Wan2.2 DMD2 discriminator uses temporal pooling with kernel size 16, so latent temporal length must be at least 16. Because frame count is `(latent_T - 1) * 4 + 1`, the smallest practical odd frame count is 65 frames with `latent_T=17`.

## Successful 65-Frame Test

Run:

- run name: `wan22_dmd2_65f_cfg5_bs1_10iter_20260429_g34`
- GPUs: `3,4`
- `model.input_shape = [48, 17, 44, 80]`
- `dataloader_train.sequence_length = 65`
- `dataloader_train.batch_size = 1`
- `trainer.batch_size_global = 2`
- `model.guidance_scale = 5.0`
- `trainer.fsdp = True`
- `trainer.fsdp_cpu_offload = True`
- `trainer.max_iter = 10`
- `trainer.save_ckpt_iter = 10`

Result: completed 10 iterations and saved checkpoint `0000010`.

Important observations:

- Successfully crossed iteration 5, the first real DMD2 student update.
- Peak GPU memory: about `28.08 GiB`
- Peak reserved GPU memory: about `29.79 GiB`
- Iteration 5 time: about `60.46s`
- Later iterations: about `24-25s/iter`

Log:

`/data/chenqingzhan/logs/wan22_dmd2_65f_cfg5_bs1_10iter_20260429_g34.log`

Temporary checkpoint:

`/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_dmd2_65f_cfg5_bs1_10iter_20260429_g34/checkpoints/0000010`

## Recommended Formal Run

Use 65 frames with native CFG:

```text
model.input_shape=[48,17,44,80]
dataloader_train.sequence_length=65
model.guidance_scale=5.0
dataloader_train.batch_size=1
trainer.batch_size_global=2
trainer.fsdp=True
trainer.fsdp_cpu_offload=True
trainer.max_iter=5000
trainer.save_ckpt_iter=1000
```

This is not fully native because original temporal length is 81 frames, but it preserves native CFG and the DMD2 student/teacher/fake-score/discriminator structure.

## Formal 5000-Iter Run Started

Started on 2026-04-30 CST:

- run name: `wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34`
- GPUs: `3,4`
- `model.input_shape = [48, 17, 44, 80]`
- `dataloader_train.sequence_length = 65`
- `dataloader_train.batch_size = 1`
- `trainer.batch_size_global = 2`
- `model.guidance_scale = 5.0`
- `trainer.max_iter = 5000`
- `trainer.save_ckpt_iter = 1000`

Log:

`/data/chenqingzhan/logs/wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34.log`

Output:

`/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34`

Status checked at 2026-04-30 11:34 CST:

- training still running
- latest logged profiler iteration: `1300`
- checkpoint `0001000` saved successfully at 2026-04-30 09:06 CST
- peak GPU memory remained about `28.08 GiB`
- peak reserved GPU memory about `30.45 GiB`
- typical iteration time around `24-26s`
