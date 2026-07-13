#!/bin/bash
# run_ect_cd_4step_inference.sh — Run 4-step inference on ECT/CD checkpoints
#
# Consistency models are trained step-agnostic. During inference, multi-step
# sampling iteratively denoises: denoise → re-noise at lower level → denoise again.
# 4-step should produce much better quality than 1-step.
#
# Usage: bash run_ect_cd_4step_inference.sh [GPU_ID]
# Default: GPU 0

set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
PROMPT_FILE="/data/chenqingzhan/FastGen/scripts/inference/prompts/eval_prompts.txt"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "=== 4-Step Inference for ECT & CD ==="
echo "GPU: $GPU"
echo ""

# --- ECT 4-step inference ---
ECT_CKPT="$FASTGEN_OUTPUT_ROOT/fastgen/wan_cm_ct/ect_1000iter_single_gpu/checkpoints"
ECT_LATEST=$(ls -t "$ECT_CKPT"/*.pth 2>/dev/null | head -1)

if [ -n "$ECT_LATEST" ]; then
    echo ">>> [1/2] ECT 4-step inference (ckpt: $(basename $ECT_LATEST))"
    python scripts/inference/video_model_inference.py \
        --config=fastgen/configs/experiments/WanT2V/config_cm_ct.py \
        --do_student_sampling True \
        --student_steps 4 \
        --ckpt_path="$ECT_LATEST" \
        --prompt_file="$PROMPT_FILE" \
        - trainer.ddp=False \
          trainer.seed=42 \
          model.guidance_scale=5.0 \
          model.net.model_id_or_local_path=$MODEL_PATH \
          log_config.name=ect_4step_inference
    echo ">>> ECT 4-step inference done"
else
    echo ">>> ECT checkpoint not found at $ECT_CKPT, skipping"
fi

echo ""

# --- CD 4-step inference ---
CD_CKPT="$FASTGEN_OUTPUT_ROOT/fastgen/wan_cm_cd/cd_1000iter_single_gpu/checkpoints"
CD_LATEST=$(ls -t "$CD_CKPT"/*.pth 2>/dev/null | head -1)

if [ -n "$CD_LATEST" ]; then
    echo ">>> [2/2] CD 4-step inference (ckpt: $(basename $CD_LATEST))"
    python scripts/inference/video_model_inference.py \
        --config=fastgen/configs/experiments/WanT2V/config_cm_cd.py \
        --do_student_sampling True \
        --student_steps 4 \
        --ckpt_path="$CD_LATEST" \
        --prompt_file="$PROMPT_FILE" \
        - trainer.ddp=False \
          trainer.seed=42 \
          model.guidance_scale=5.0 \
          model.net.model_id_or_local_path=$MODEL_PATH \
          log_config.name=cd_4step_inference
    echo ">>> CD 4-step inference done"
else
    echo ">>> CD checkpoint not found at $CD_CKPT, skipping"
fi

echo ""
echo "=== All inference complete ==="
