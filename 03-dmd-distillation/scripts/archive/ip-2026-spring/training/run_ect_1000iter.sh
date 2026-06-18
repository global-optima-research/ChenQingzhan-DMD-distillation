#!/bin/bash
# run_ect_1000iter.sh — ECT lightweight training (1000 iter, single GPU)
#
# ECT: Consistency Training without teacher (use_cd=False)
# Single GPU, batch_size=1, no FSDP/DDP
#
# Usage: bash run_ect_1000iter.sh
# Assign GPU via CUDA_VISIBLE_DEVICES before launch, or use default GPU 0.

set -euo pipefail

# === GPU Assignment ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[ECT] Using GPU: ${CUDA_VISIBLE_DEVICES}"

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

echo "[ECT] Starting lightweight training: 1000 iter, single GPU, batch=1"
echo "[ECT] Model: Wan2.1-T2V-1.3B"
echo "[ECT] Config: config_cm_ct.py (custom ECT)"
echo "[ECT] Dataset: OpenVid-1M (21K samples)"
echo "[ECT] Optimizer: 8-bit AdamW (via train_lowmem.py)"
echo ""

# Single GPU: use python directly (not torchrun), disable FSDP/DDP
# Use train_lowmem.py wrapper for 8-bit AdamW (~5 GB VRAM savings)
python "$TRAIN_SCRIPT" \
    --config=fastgen/configs/experiments/WanT2V/config_cm_ct.py \
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
      log_config.name=ect_1000iter_single_gpu

echo "[ECT] Training complete!"
