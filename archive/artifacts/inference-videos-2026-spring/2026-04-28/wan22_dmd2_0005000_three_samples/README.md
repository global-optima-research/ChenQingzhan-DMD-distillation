# Wan2.2 DMD2 0005000 Inference Samples

Generated on server `chenqingzhan@111.17.197.107` from checkpoint:

`/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_5b_i2v_dmd2_stage1_20260427_g23_cpuoffload_no_cfg/checkpoints/0005000`

Inference settings:

- Config: `fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
- Student sampling only
- `student_sample_steps=2`
- `guidance_scale=null`
- Seed: `0`
- FPS: `16`
- Decode path: `/data/chenqingzhan/scripts/video_model_inference_decode_offload.py`

Files:

- `sample_0_golden_retriever.mp4`
  - Prompt: `A golden retriever puppy playing joyfully in a sunny garden with colorful flowers blooming around it`
  - Input image: `scripts/inference/examples/00_child_swings_rusty_swing_set.png`
- `sample_1_rainy_cat.mp4`
  - Prompt: `A white cat sitting on a windowsill watching raindrops slide down the glass on a rainy afternoon`
  - Input image: `scripts/inference/examples/01_impressionist_rubber_duck_sunset.png`
- `sample_2_surfer.mp4`
  - Prompt: `A surfer riding a massive ocean wave under a bright blue sky with seagulls flying overhead`
  - Input image: `scripts/inference/examples/02_two_cyclists_stop_sign.png`
