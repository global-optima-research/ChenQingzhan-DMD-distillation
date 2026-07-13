#!/bin/bash
# run_cd_1000iter.sh — CD lightweight training (1000 iter, single GPU)
#
# CD: Consistency Distillation with teacher (use_cd=True)
# Single GPU, batch_size=1, no FSDP/DDP
#
# Usage: bash run_cd_1000iter.sh
# Assign GPU via CUDA_VISIBLE_DEVICES before launch, or use default GPU 1.

set -euo pipefail

# === GPU Assignment ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "[CD] Using GPU: ${CUDA_VISIBLE_DEVICES}"

# === Environment ===
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_lowmem.py"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[CD] Starting lightweight training: 1000 iter, single GPU, batch=1"
echo "[CD] Model: Wan2.1-T2V-1.3B (Student + Teacher)"
echo "[CD] Config: config_cm_cd.py (custom CD)"
echo "[CD] Dataset: OpenVid-1M (21K samples)"
echo "[CD] Optimizer: 8-bit AdamW (via train_lowmem.py)"
echo "[CD] Note: Teacher model is frozen (no gradients), used for ODE solving"
echo ""

# Single GPU: use python directly, disable FSDP/DDP
# Use train_lowmem.py wrapper for 8-bit AdamW (~5 GB VRAM savings)
python "$TRAIN_SCRIPT" \
    --config=fastgen/configs/experiments/WanT2V/config_cm_cd.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=1 \
      trainer.max_iter=1000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=200 \
      trainer.validation_iter=500 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=cd_1000iter_single_gpu

echo "[CD] Training complete!"
