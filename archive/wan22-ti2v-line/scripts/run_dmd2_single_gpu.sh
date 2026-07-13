#!/bin/bash
# run_dmd2_single_gpu.sh — Run DMD2 distillation on Wan2.1-1.3B with single GPU
# Usage: bash run_dmd2_single_gpu.sh [data_dir_or_wds_tag]
#
# Note:
# - This script now uses the current OpenVid WebDataset path by default.
# - On a single 32 GB GPU, DMD2 still has a high OOM risk. Prefer this as a
#   smoke-test or debugging entrypoint rather than the default formal run.

set -euo pipefail

# Environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
export HF_HOME="${HF_HOME:-/data/chenqingzhan/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CONDA_DIR="/data/chenqingzhan/miniconda3"
FASTGEN_DIR="${FASTGEN_DIR:-/data/chenqingzhan/FastGen}"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"
MODEL_PATH="${MODEL_PATH:-/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$MODEL_PATH}"
DATA_DIR="${1:-/data/datasets/OpenVid-1M/webdataset}"
RUN_NAME="${RUN_NAME:-dmd2_wan1.3b_single_gpu}"
MAX_ITER="${MAX_ITER:-6000}"
LOGGING_ITER="${LOGGING_ITER:-50}"
SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
BATCH_SIZE_GLOBAL="${BATCH_SIZE_GLOBAL:-8}"
LOG_PATH="${LOG_PATH:-$FASTGEN_OUTPUT_ROOT/dmd2_train.log}"

make_wds_tag() {
    case "$1" in
        WDS:*) printf '%s' "$1" ;;
        *) printf 'WDS:%s' "$1" ;;
    esac
}

DATA_TAG="$(make_wds_tag "$DATA_DIR")"

if [ ! -d "$FASTGEN_DIR" ]; then
    echo "[DMD2] FASTGEN_DIR not found: $FASTGEN_DIR"
    exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
    echo "[DMD2] MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

# Single GPU training: use python directly (not torchrun with nproc>1)
# Key adjustments for single GPU:
#   - trainer.ddp=False (no distributed data parallel)
#   - trainer.fsdp=False (no model sharding)
#   - trainer.batch_size_global defaults to 8 (grad accumulation path)
#   - DMD2 still has a high VRAM requirement on single 32 GB GPUs
$PYTHON train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=$BATCH_SIZE_GLOBAL \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOGGING_ITER \
      trainer.save_ckpt_iter=$SAVE_CKPT_ITER \
      model.net.model_id_or_local_path=$MODEL_PATH \
      model.teacher.model_id_or_local_path=$TEACHER_MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_TAG\"]" \
      log_config.name=$RUN_NAME \
      2>&1 | tee "$LOG_PATH"
