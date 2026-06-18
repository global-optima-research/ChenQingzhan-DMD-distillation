# Wan2.2 TI2V 5B DMD2 Inference Sample

- Date: 2026-04-27
- Server checkpoint prefix:
  `/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_stage1_20260427_g23_cpuoffload_no_cfg/checkpoints/0001000`
- Config: `fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
- Sampling: student only, `student_sample_steps=2`, `model.guidance_scale=null`
- Seed: `42`
- Prompt:
  `A golden retriever puppy playing joyfully in a sunny garden with colorful flowers blooming around it`
- Source image:
  `/data/chenqingzhan/FastGen/scripts/inference/examples/00_child_swings_rusty_swing_set.png`
- Output:
  `student_step2_i2v_0000_seed42.mp4`

Note: the first inference attempt completed latent sampling but OOMed during VAE decode.
The successful run used a temporary inference-only script on the server:
`/data/chenqingzhan/scripts/video_model_inference_decode_offload.py`.
It moves the student transformer to CPU before VAE decode, then saves the video.

