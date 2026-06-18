#!/usr/bin/env bash
set -euo pipefail

cd /data/chenqingzhan/FastGen

RUN=wan22_dmd2_65f_cfg5_0005000_10samples_20260501
OUT=/data/chenqingzhan/inference_outputs/$RUN
LOG=/data/chenqingzhan/logs/${RUN}.log
CKPT=/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34/checkpoints/0005000
PROMPTS=/data/chenqingzhan/inference_inputs/wan22_5000_10samples_20260501/prompts.txt
IMAGES=/data/chenqingzhan/inference_inputs/wan22_5000_10samples_20260501/images.txt

mkdir -p "$OUT"
: > "$LOG"

{
  echo "__START__ $(date +%F_%T_%Z)"
  echo "ckpt=$CKPT"
  echo "prompts=$PROMPTS"
  echo "images=$IMAGES"
  echo "out=$OUT"
} | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=5 \
FASTGEN_OUTPUT_ROOT=/data/chenqingzhan/fastgen_output \
HF_HOME=/data/chenqingzhan/.cache/huggingface \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH=/data/chenqingzhan/FastGen \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
/data/chenqingzhan/miniconda3/envs/fastgen/bin/python \
  /data/chenqingzhan/scripts/video_model_inference_decode_offload.py \
  --ckpt_path "$CKPT" \
  --do_student_sampling True \
  --do_teacher_sampling False \
  --save_as_gif False \
  --fps 16 \
  --prompt_file "$PROMPTS" \
  --input_image_file "$IMAGES" \
  --video_save_dir "$OUT" \
  --config fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py \
  - \
  model.net.model_id_or_local_path=/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers \
  model.input_shape=[48,17,44,80] \
  dataloader_train.sequence_length=65 \
  model.student_sample_steps=2 \
  model.student_sample_type=ode \
  model.sample_t_cfg.t_list=[0.999,0.833,0.0] \
  model.guidance_scale=5.0 \
  log_config.name=${RUN} \
  log_config.wandb_mode=disabled \
  ~trainer.callbacks.wandb >> "$LOG" 2>&1

{
  echo "__DONE__ $(date +%F_%T_%Z)"
  find "$OUT" -type f -name "*.mp4" -printf "%s %p\n" | sort
} | tee -a "$LOG"
