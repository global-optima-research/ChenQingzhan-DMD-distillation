# Wan2.2 I2V DMD2 Experiment Configs

Date: 2026-05-04

## Stopped Baseline Continuation

- Run name: `wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34`
- Resume log: `/data/chenqingzhan/logs/wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34_resume_from5000_20260501_g56.log`
- Status: stopped by SIGTERM on 2026-05-04 02:59 CST
- Start point: resumed from `0005000`
- Latest stable checkpoint before stop: `0013000`
- Latest logged iteration before stop: about `13200`
- Base learning rate:
  - `model.net_optimizer.lr=1e-5`
  - `model.fake_score_optimizer.lr=1e-5`
  - `model.discriminator_optimizer.lr=1e-5`
- Shared training config:
  - Wan2.2 TI2V 5B diffusers model
  - DMD2, WanI2V config
  - 65 frames: `model.input_shape=[48,17,44,80]`, `sequence_length=65`
  - CFG: `model.guidance_scale=5.0`
  - Student sampling: `student_sample_steps=2`, `student_sample_type=ode`
  - GPUs: 2, FSDP + CPU offload
  - Per-GPU batch size: 1
  - Global batch size: 2
  - Dataset: `WDS:/data/datasets/OpenVid-1M/webdataset`
  - Save interval: 1000 iter

## LR x10 From-Scratch Run

- Run name: `wan22_dmd2_65f_cfg5_bs1_lr10x_fromscratch_20260504_g56`
- Log: `/data/chenqingzhan/logs/wan22_dmd2_65f_cfg5_bs1_lr10x_fromscratch_20260504_g56.log`
- Launch script: `/data/chenqingzhan/scripts/start_wan22_dmd2_65f_cfg5_lr10x_fromscratch_g56.sh`
- Local script copy: `scripts/start_wan22_dmd2_65f_cfg5_lr10x_fromscratch_g56.sh`
- Status at launch check: running, `iter_start: 0`
- Progress check: `0001000` checkpoint saved at 2026-05-04 09:59 CST; latest checked log reached `1200 iter`
- Resume behavior: `trainer.resume=False`
- LR x10:
  - `model.net_optimizer.lr=1e-4`
  - `model.fake_score_optimizer.lr=1e-4`
  - `model.discriminator_optimizer.lr=1e-4`
- Shared training config:
  - Wan2.2 TI2V 5B diffusers model
  - DMD2, WanI2V config
  - 65 frames: `model.input_shape=[48,17,44,80]`, `sequence_length=65`
  - CFG: `model.guidance_scale=5.0`
  - Student sampling: `student_sample_steps=2`
  - GPUs: 5,6
  - FSDP + CPU offload
  - Per-GPU batch size: 1
  - Global batch size: 2
  - Dataset: `WDS:/data/datasets/OpenVid-1M/webdataset`
  - Save interval: 1000 iter
  - Max iter set high (`1000000`) so it continues until manually stopped

## 0013000 Inference Evaluation

- Source checkpoint: `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34/checkpoints/0013000`
- Run name: `wan22_dmd2_65f_cfg5_0013000_10distinct_20260504`
- Server output: `/data/chenqingzhan/inference_outputs/wan22_dmd2_65f_cfg5_0013000_10distinct_20260504`
- Local output: `artifacts/inference_videos/2026-05-04/wan22_dmd2_65f_cfg5_0013000_10distinct_20260504`
- Prompt/image inputs: `/data/chenqingzhan/inference_inputs/wan22_13000_10distinct_20260504`
- GPU: 0
- Status: completed 10/10 samples on 2026-05-04 03:20 CST and downloaded locally.
- 10 prompts and 10 first-frame images are all distinct.
