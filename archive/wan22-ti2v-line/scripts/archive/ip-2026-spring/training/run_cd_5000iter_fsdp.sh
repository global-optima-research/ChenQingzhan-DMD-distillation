#!/bin/bash
# run_cd_5000iter_fsdp.sh — CD training with accelerated curriculum, multi-GPU FSDP
#
# Changes from 1000-iter run:
#   - kimg_per_stage: 50 → 5 (10x faster curriculum progression)
#   - max_iter: 5000
#   - Multi-GPU FSDP (3 GPUs)
#   - batch_size_global=3 (1 per GPU, no grad accumulation)
#
# Usage: bash run_cd_5000iter_fsdp.sh

set -euo pipefail

# === GPU Assignment (3 GPUs) ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,5}"
echo "[CD-5K] Using GPUs: ${CUDA_VISIBLE_DEVICES}"

# === Environment ===
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[CD-5K] CD 5000 iter, 2-GPU FSDP, kimg_per_stage=5"
echo "[CD-5K] Model: Wan2.1-T2V-1.3B (Student + Teacher)"
echo "[CD-5K] Dataset: OpenVid-1M (21K samples)"
echo ""

# 3-GPU FSDP training
torchrun --nproc_per_node=2 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_cm_cd.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=2 \
      trainer.max_iter=5000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=1000 \
      trainer.validation_iter=1000 \
      trainer.callbacks.ct_schedule.kimg_per_stage=5 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=cd_5000iter_fsdp_kimg5

echo "[CD-5K] Training complete!"
