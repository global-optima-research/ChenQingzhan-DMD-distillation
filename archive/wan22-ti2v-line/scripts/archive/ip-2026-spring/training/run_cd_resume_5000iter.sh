#!/bin/bash
# run_cd_resume_5000iter.sh — Resume CD from 1000iter checkpoint, train to 5000
#
# Single GPU + 8-bit AdamW, kimg_per_stage=5 (accelerated curriculum)
# Resumes from: cd_1000iter_single_gpu/checkpoints/0001000.pth
# Saves checkpoint every 1000 iter (2000, 3000, 4000, 5000)
#
# Usage: bash run_cd_resume_5000iter.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
echo "[CD-Resume] Using GPU: ${CUDA_VISIBLE_DEVICES}"

export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_lowmem.py"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

# Resume checkpoint path
RESUME_CKPT="$FASTGEN_OUTPUT_ROOT/fastgen/wan_cm_cd/cd_1000iter_single_gpu/checkpoints/0001000.pth"

echo "[CD-Resume] Resuming from: $RESUME_CKPT"
echo "[CD-Resume] Target: 5000 iter, kimg_per_stage=5, single GPU, 8-bit AdamW"
echo ""

python "$TRAIN_SCRIPT" \
    --config=fastgen/configs/experiments/WanT2V/config_cm_cd.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=1 \
      trainer.max_iter=5000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=1000 \
      trainer.validation_iter=1000 \
      trainer.resume=True \
      trainer.checkpointer.pretrained_ckpt_path=$RESUME_CKPT \
      trainer.callbacks.ct_schedule.kimg_per_stage=5 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=cd_resume_5000iter_kimg5

echo "[CD-Resume] Training complete!"
