# Wan2.2 TI2V 5B Handoff

> **Status correction (2026-07-06).** This handoff covers the `Wan2.2 TI2V 5B / WanI2V / DMD2` line only through 2026-04-27. That line continued in May 2026 (65-frame CFG=5 runs up to a `0013000` checkpoint, not recorded here) and stopped on 2026-05-10. The active mainline since 2026-06-06 is `Wan2.1-T2V-1.3B (WanT2V) / DMD2` progressive `50 -> 8 -> 4` on OpenVid-1M. Do not resume from the procedures below without re-checking the server. Verified current state: `research/T0_project_analysis.md` (section 0).

This file is the fast resume note for a new Codex session.

Use absolute dates. Do not rely on "today", "yesterday", or "latest" without re-checking the server.

## 1. Current Mainline

Current mainline task:

- Validate and continue `FastGen` native `Wan2.2 TI2V 5B / WanI2V / DMD2` distillation on the server.
- Priority path is native `2-step DMD2`, not the older `Wan2.1` placeholder experiments.
- Before any new training launch, always re-check current GPU usage and current training logs on the server.

## 2. Read These First

Start every new session by reading:

1. `README.md`
2. `agents/README.md`
3. `experiments/README.md`
4. `03-dmd-distillation/HANDOFF.md`
5. `03-dmd-distillation/OVERVIEW.md`

## 3. Stable Facts Already Verified

As of `2026-04-23`, the following were verified on the server:

- Server host: `111.17.197.107`
- User: `chenqingzhan`
- Local SSH alias: `ust_ip`
- Hostname: `RTX-5090-32G-X8`
- `FastGen` path: `/data/chenqingzhan/FastGen`
- `FastGen` commit: `34f30e8`
- Config exists: `fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
- Config exists: `fastgen/configs/experiments/WanI2V/config_sft_wan22_5b.py`
- Network exists: `fastgen/networks/WanI2V/network.py`
- Inference entry exists: `scripts/inference/video_model_inference.py`
- Model path exists: `/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers`
- Data path exists: `/data/datasets/OpenVid-1M/webdataset`

## 4. Last Fully Validated Training Result

Last fully validated milestone was on `2026-04-23`.

Validated working configuration:

- GPUs: `5,6`
- Launch mode: `torchrun --standalone --nproc_per_node=2`
- `trainer.ddp=False`
- `trainer.fsdp=True`
- `trainer.fsdp_cpu_offload=True`
- `trainer.batch_size_global=2`
- `dataloader_train.batch_size=1`
- `dataloader_train.datatags=["WDS:/data/datasets/OpenVid-1M/webdataset"]`

Validated smoke run:

- Run name: `wan22_5b_i2v_dmd2_smoke_20260423_g56_cpuoffload`
- Log: `/data/chenqingzhan/logs/wan22_dmd2_smoke_20260423_g56_cpuoffload.log`
- Output dir: `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_smoke_20260423_g56_cpuoffload`

Validated outcomes:

- FSDP wrapping completed
- Dataloader instantiated
- Real training loop entered
- Iteration `1` metrics printed
- Iteration `2` checkpoint save started and completed
- Training exited cleanly

Observed metrics at iteration `1`:

- `avg_total_loss = 1.3984`
- `avg_fake_score_loss = 0.0182`
- `avg_gan_loss_disc = 1.3789`

## 5. Stage Run Status

A longer run was launched on `2026-04-23`. Its status was re-checked on `2026-04-26`.

Stage run that was launched:

- Run name: `wan22_5b_i2v_dmd2_stage1_20260423_g56_cpuoffload`
- Target: `max_iter=1000`
- Log: `/data/chenqingzhan/logs/wan22_dmd2_stage1_20260423_g56_cpuoffload.log`
- Output dir: `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_stage1_20260423_g56_cpuoffload`

Important:

- The `2026-04-23` stage run is not active.
- It failed with CUDA OOM after entering training and printing iteration `1` metrics.
- No checkpoint exists for this stage run.
- The `2026-04-23` smoke checkpoint still exists at iteration `2`.

Observed stage metrics before failure:

- `avg_total_loss = 1.4219`
- `avg_fake_score_loss = 0.0346`
- `avg_gan_loss_disc = 1.3906`

The OOM signature was:

- `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.27 GiB`
- Per-rank allocated memory was about `30.21 GiB`
- The failure happened in `fastgen/networks/Wan/network.py`, `block_forward`

Additional `2026-04-26` attempts:

- `wan22_5b_i2v_dmd2_smoke_20260426_g0123_cpuoffload_v2`
  - Failed before training due to a bad teacher override.
  - Do not pass `+model.teacher.model_id_or_local_path=...` for this config.
  - Leave `model.teacher` as `null`; the teacher uses `model.net`.
- `wan22_5b_i2v_dmd2_smoke_20260426_g0123_cpuoffload_v3`
  - GPUs: `0,1,2,3`
  - `trainer.fsdp_cpu_offload=True`
  - `trainer.batch_size_global=4`
  - `dataloader_train.batch_size=1`
  - Printed iteration `1` metrics:
    - `avg_total_loss = 1.4102`
    - `avg_fake_score_loss = 0.0183`
    - `avg_gan_loss_disc = 1.3906`
  - Failed with the same `1.27 GiB` OOM at the next forward.
- `wan22_5b_i2v_dmd2_smoke_20260426_g0123_cpuoffload_v4_nowandb`
  - Same 4-GPU route, with `~trainer.callbacks.wandb`
  - Failed with the same `1.27 GiB` OOM after FSDP wrapping / training start.
- `wan22_5b_i2v_dmd2_smoke_20260426_g01_cpuoffload_nowandb`
  - Was started on `0,1`, but another user had taken GPU `0,1,2,3` between checks.
  - It was manually stopped on `2026-04-26` to avoid interfering with that job.
- `wan22_5b_i2v_dmd2_smoke_20260426_g01_cpuoffload_nowandb_v2`
  - GPUs: `0,1`
  - Removed `wandb` callback with `~trainer.callbacks.wandb`
  - Kept native `model.guidance_scale=5.0`
  - Failed at the first real student update (`iteration 5`) with the same `1.27 GiB` OOM.
- `wan22_5b_i2v_dmd2_smoke_20260426_g01_cpuoffload_no_cfg_nowandb`
  - GPUs: `0,1`
  - Removed `wandb` callback with `~trainer.callbacks.wandb`
  - Set `model.guidance_scale=null`
  - Target: `max_iter=10`, `save_ckpt_iter=10`
  - Result: completed successfully on `2026-04-26 23:22:06 CST`
  - Checkpoint saved at:
    `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_smoke_20260426_g01_cpuoffload_no_cfg_nowandb/checkpoints/0000010.pth`
  - This run crossed the real student-update point at `iteration 5`.
  - Observed profiler at `iteration 5`:
    - data loading time: `2.27s`
    - avg forward pass time: `5.99s`
    - backward pass time: `38.65s`
    - optimizer step time: `14.96s`

Additional `2026-04-27` status:

- Official/native FastGen config inspected on server commit `34f30e8`:
  `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
  - Native target already uses `trainer.max_iter = 5000`, `trainer.logging_iter = 100`, `trainer.save_ckpt_iter = 500`.
  - Student route:
    - `config.model.net = Wan22_I2V_5B_Config`
    - `config.model.student_sample_steps = 2`
    - `config.model.student_sample_type = "ode"`
    - `config.model.sample_t_cfg.t_list = [0.999, 0.833, 0.0]`
  - Native DMD2 components:
    - student/net: `Wan22_I2V_5B_Config`
    - teacher: built from the same net config when `model.teacher` is left `null`
    - fake_score: another Wan I2V 5B network
    - discriminator: `Discriminator_Wan22_5B_Config`, `disc_type = "multiscale_down_mlp_large"`, `feature_indices = [15, 22, 29]`
  - Native optimization/loss:
    - `net_optimizer.lr = 1e-5`
    - `discriminator_optimizer.lr = 1e-5`
    - `fake_score_optimizer.lr = 1e-5`
    - `gan_loss_weight_gen = 0.03`
    - `gan_use_same_t_noise = True`
    - `fake_score_pred_type = "x0"`
    - `student_update_freq = 5` inherited from the DMD2 default method config
  - Native memory-sensitive defaults:
    - `config.model.guidance_scale = 5.0`
    - `config.dataloader_train.batch_size = 2`
  - On this server, the exact native `guidance_scale=5.0` path OOMs at the first real student update. The validated hardware route keeps the native 2-step student/training target but overrides `model.guidance_scale=null`, `dataloader_train.batch_size=1`, `trainer.batch_size_global=2`, `trainer.fsdp=True`, and `trainer.fsdp_cpu_offload=True`.
- Server-side patch applied to `/data/chenqingzhan/FastGen/fastgen/callbacks/wandb.py`
  - Backup: `/data/chenqingzhan/FastGen/fastgen/callbacks/wandb.py.bak_20260426`
  - New env var: `FASTGEN_DISABLE_MEDIA_LOGGING=true`
  - Purpose: keep scalar loss logging while skipping sample/video media logging.
  - `python -m py_compile fastgen/callbacks/wandb.py` passed on the server.
- `wan22_5b_i2v_dmd2_stage1_20260426_g01_cpuoffload_no_cfg`
  - GPUs: `0,1`
  - Kept `wandb` callback for scalar metrics and used `FASTGEN_DISABLE_MEDIA_LOGGING=true`
  - Set `model.guidance_scale=null`
  - Failed after iteration `1` with CUDA OOM because GPU `0` already had about `712 MiB` used by another user's process.
  - Conclusion: the precheck threshold must be stricter than `1 GiB`; prefer selected GPUs with less than about `100 MiB` used and low utilization.
- `wan22_5b_i2v_dmd2_stage1_20260427_g23_cpuoffload_no_cfg`
  - GPUs: `2,3`
  - Launched on `2026-04-27 02:26 CST`
  - PID: `2971098`
  - Log: `/data/chenqingzhan/logs/wan22_dmd2_stage1_20260427_g23_cpuoffload_no_cfg.log`
  - Output dir: `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_stage1_20260427_g23_cpuoffload_no_cfg`
  - Config route:
    - `trainer.ddp=False`
    - `trainer.fsdp=True`
    - `trainer.fsdp_cpu_offload=True`
    - `trainer.batch_size_global=2`
    - `dataloader_train.batch_size=1`
    - `model.student_sample_steps=2`
    - `model.guidance_scale=null`
    - `FASTGEN_DISABLE_MEDIA_LOGGING=true`
    - `trainer.callbacks.wandb.sample_logging_iter=1000000`
  - As of `2026-04-27 02:37 CST`, the run was active and had passed iteration `10`.
  - Iteration `1` metrics:
    - `avg_total_loss = 1.4492`
    - `avg_fake_score_loss = 0.0424`
    - `avg_gan_loss_disc = 1.4062`
  - Iteration `10` metrics:
    - `avg_total_loss = 1.1008`
    - `avg_fake_score_loss = 0.0157`
    - `avg_gan_loss_disc = 1.3845`
    - `avg_vsd_loss = 0.0307`
    - `avg_gan_loss_gen = 0.6865`
  - Iteration `10` profiler:
    - data loading time: `2.27s`
    - avg forward pass time: `5.73s`
    - backward pass time: `32.27s`
    - optimizer step time: `7.60s`
  - First checkpoint is expected at iteration `100`; no checkpoint existed yet at `2026-04-27 02:37 CST`.
- A continuation queue for the native 5000-iteration target was installed on `2026-04-27 02:47 CST`.
  - Queue script: `/data/chenqingzhan/scripts/queue_wan22_dmd2_5000_after_g23.sh`
  - Queue log: `/data/chenqingzhan/logs/queue_wan22_dmd2_5000_after_g23.log`
  - Queue PID at launch: `3067168`
  - It waits for PID `2971098` to exit, sleeps `60s`, checks GPUs `2,3` are clean (`<=100 MiB` and utilization `<=20%`), then relaunches the same run name with:
    - `trainer.max_iter=5000`
    - `trainer.logging_iter=100`
    - `trainer.save_ckpt_iter=500`
    - `trainer.resume=True` inherited from the base config
  - The continuation log will be:
    `/data/chenqingzhan/logs/wan22_dmd2_formal_20260427_g23_cpuoffload_no_cfg_5000_resume.log`
  - It uses the same validated hardware-safe overrides:
    - `trainer.ddp=False`
    - `trainer.fsdp=True`
    - `trainer.fsdp_cpu_offload=True`
    - `trainer.batch_size_global=2`
    - `dataloader_train.batch_size=1`
    - `model.student_sample_steps=2`
    - `model.guidance_scale=null`
    - `FASTGEN_DISABLE_MEDIA_LOGGING=true`

Current `2026-04-27` conclusion:

- Do not treat 4-GPU `0,1,2,3` as a fix; it still OOMs.
- The first smoke that crosses DMD2 student update on this hardware is the 2-GPU no-CFG route:
  - `trainer.ddp=False`
  - `trainer.fsdp=True`
  - `trainer.fsdp_cpu_offload=True`
  - `trainer.batch_size_global=2`
  - `dataloader_train.batch_size=1`
  - `~trainer.callbacks.wandb`
  - `model.guidance_scale=null`
- A 1000-iter stage based on that no-CFG route is active as
  `wan22_5b_i2v_dmd2_stage1_20260427_g23_cpuoffload_no_cfg`.
- Before any future launch, enforce a hard precheck that aborts if selected GPUs have more than about `100 MiB` used or show non-idle utilization.
- Avoid GPU `4` while it shows `100%` utilization with no visible process.
- If loss metrics are needed while avoiding media logging, use the server-side `FASTGEN_DISABLE_MEDIA_LOGGING=true` patch or remove the callback with `~trainer.callbacks.wandb`; simply setting `log_config.wandb_mode=disabled` does not prevent iteration-1 sample/video decoding in the unpatched callback.

## 6. Known Failure Modes

These were already hit and diagnosed. Do not repeat them blindly.

1. `DDP + FSDP` enabled together
   - Symptom: `AssertionError: Model cannot be wrapped into both DDP and FSDP`
   - Fix: use `trainer.ddp=False` with `trainer.fsdp=True`

2. Wrong WebDataset tag pattern
   - Symptom: `ValueError: Invalid prefix: /data/datasets/OpenVid-1M/webdataset/shard-{000000..000003}.tar`
   - Fix: use the directory tag form
   - Valid form: `["WDS:/data/datasets/OpenVid-1M/webdataset"]`

3. OOM without CPU offload
   - Symptom: OOM during FSDP init or first forward
   - Fix: `trainer.fsdp_cpu_offload=True`

4. GPU availability changes while debugging
   - Symptom: a partially free GPU becomes occupied by another user's process between checks
   - Fix: re-check GPU occupancy immediately before launch

5. Bad teacher override for `WanI2V/config_dmd2_wan22_5b.py`
   - Symptom: `omegaconf.errors.ConfigAttributeError: Key 'eval' is not in struct`
   - Cause: adding only `model.teacher.model_id_or_local_path` creates an incomplete teacher config
   - Fix: do not override `model.teacher`; override only `model.net.model_id_or_local_path`

6. `wandb_mode=disabled` still performs media/sample decoding
   - Symptom: iteration `1` logs video/sample output and then later OOMs
   - Cause: `WandbCallback.on_training_step_end()` always calls `log_sample_map()` at `iteration == 1`
   - Workaround: remove the callback with `~trainer.callbacks.wandb`, or patch the callback to log scalar losses without media

7. Native CFG teacher pass does not fit at the real DMD2 student update
   - Symptom: OOM at `iteration 5`, inside `_student_update_step -> _apply_classifier_free_guidance`
   - Cause: the second teacher forward for negative-condition CFG tries to allocate another `1.27 GiB`
   - Workaround validated on `2026-04-26`: set `model.guidance_scale=null`

## 7. Recommended Resume Procedure

When resuming in a new Codex session:

1. Read the docs listed above.
2. Run the non-invasive precheck:
   `bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env`
3. If launching, run dry-run first:
   `bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env`
4. Check the latest status of:
   - `/data/chenqingzhan/logs/wan22_dmd2_smoke_20260423_g56_cpuoffload.log`
   - `/data/chenqingzhan/logs/wan22_dmd2_stage1_20260423_g56_cpuoffload.log`
5. Check whether checkpoints exist under:
   - `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_smoke_20260423_g56_cpuoffload/checkpoints`
   - `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_stage1_20260423_g56_cpuoffload/checkpoints`
6. If the stage run is dead or incomplete, do not blindly relaunch it. First find two actually free GPUs and use the validated no-CFG 2-GPU route.
7. Only after `1000 iter` is stable should the run be expanded to a larger formal run.

## 8. Minimal Resume Prompt

Paste this into a new Codex chat if needed:

```text
Work in /Users/x3y/Desktop/ChenQingzhan-DMD-distillation.

First read:
1. README.md
2. agents/README.md
3. experiments/README.md
4. 03-dmd-distillation/HANDOFF.md
5. 03-dmd-distillation/OVERVIEW.md

Main task: continue FastGen native Wan2.2 TI2V 5B / WanI2V / DMD2 distillation.

Last fully validated milestone was on 2026-04-23:
- 2-GPU run on GPUs 5,6
- trainer.ddp=False
- trainer.fsdp=True
- trainer.fsdp_cpu_offload=True
- batch_size_global=2
- dataloader_train.batch_size=1
- datatags=["WDS:/data/datasets/OpenVid-1M/webdataset"]
- 2-iter smoke run completed and saved checkpoint

Known failure modes:
- DDP + FSDP together
- invalid WDS shard brace-pattern tag
- OOM without fsdp_cpu_offload

After reading, use ssh ust_ip. Run experiments/bin/check_remote.sh with the target config, dry-run experiments/bin/run_remote.sh, then continue from the validated g56 + CPU offload route.
```
