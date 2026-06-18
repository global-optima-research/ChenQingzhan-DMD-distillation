#!/usr/bin/env bash
set -euo pipefail

cd /data/chenqingzhan/FastGen

GPU_ID=0
RUN=wan22_dmd2_65f_cfg5_0013000_0501_first5_20260504
OUT=/data/chenqingzhan/inference_outputs/$RUN
LOG=/data/chenqingzhan/logs/${RUN}.log
CKPT=/data/chenqingzhan/fastgen_output/fastgen/wan22_5b_i2v_dmd2/wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34/checkpoints/0013000
SRC_PROMPTS=/data/chenqingzhan/inference_inputs/wan22_5000_10samples_20260501/prompts.txt
SRC_IMAGES=/data/chenqingzhan/inference_inputs/wan22_5000_10samples_20260501/images.txt
INPUT_DIR=/data/chenqingzhan/inference_inputs/${RUN}
PROMPTS=$INPUT_DIR/prompts.txt
IMAGES=$INPUT_DIR/images.txt

mkdir -p "$OUT" "$INPUT_DIR" "$OUT/first_frames"
sed -n '1,5p' "$SRC_PROMPTS" > "$PROMPTS"
sed -n '1,5p' "$SRC_IMAGES" > "$IMAGES"
: > "$LOG"

{
  echo "__START__ $(date +%F_%T_%Z)"
  echo "gpu=$GPU_ID"
  echo "ckpt=$CKPT"
  echo "src_prompts=$SRC_PROMPTS"
  echo "src_images=$SRC_IMAGES"
  echo "prompts=$PROMPTS"
  echo "images=$IMAGES"
  echo "out=$OUT"
} | tee -a "$LOG"

for idx in 0 1 2 3 4; do
  line_no=$((idx + 1))
  prompt="$(sed -n "${line_no}p" "$PROMPTS")"
  image="$(sed -n "${line_no}p" "$IMAGES")"
  sample_dir="$OUT/sample_${idx}"
  pfile="$OUT/prompt_${idx}.txt"
  ifile="$OUT/image_${idx}.txt"

  mkdir -p "$sample_dir"
  printf "%s\n" "$prompt" > "$pfile"
  printf "%s\n" "$image" > "$ifile"
  cp "$image" "$OUT/first_frames/$(printf '%02d' "$idx")_$(basename "$image")"

  echo "__SAMPLE_${idx}_START__ $(date +%F_%T_%Z)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU_ID \
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
    --prompt_file "$pfile" \
    --input_image_file "$ifile" \
    --video_save_dir "$sample_dir" \
    --config fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py \
    - \
    model.net.model_id_or_local_path=/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers \
    model.input_shape=[48,17,44,80] \
    dataloader_train.sequence_length=65 \
    model.student_sample_steps=2 \
    model.student_sample_type=ode \
    model.sample_t_cfg.t_list=[0.999,0.833,0.0] \
    model.guidance_scale=5.0 \
    log_config.name=${RUN}_sample_${idx} \
    log_config.wandb_mode=disabled \
    ~trainer.callbacks.wandb >> "$LOG" 2>&1
  echo "__SAMPLE_${idx}_DONE__ $(date +%F_%T_%Z)" | tee -a "$LOG"
done

{
  echo "__DONE__ $(date +%F_%T_%Z)"
  find "$OUT" -type f -name "*.mp4" -printf "%s %p\n" | sort
} | tee -a "$LOG"
