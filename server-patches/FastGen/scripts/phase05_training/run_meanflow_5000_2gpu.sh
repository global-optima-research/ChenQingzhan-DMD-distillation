#!/bin/bash
# MeanFlow stage run on 2 GPUs with raw-video WebDataset input.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
export HF_HOME="${HF_HOME:-/data/chenqingzhan/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FASTGEN_DIR="${FASTGEN_DIR:-/data/chenqingzhan/FastGen}"
MODEL_PATH="${MODEL_PATH:-/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers}"
DATA_SHARDS="${DATA_SHARDS:-WDS:/data/datasets/OpenVid-1M/webdataset}"
RUN_NAME="${RUN_NAME:-meanflow_5000iter_2gpu}"
MAX_ITER="${MAX_ITER:-5000}"
SAVE_ITER="${SAVE_ITER:-500}"
VAL_ITER="${VAL_ITER:-500}"
LOG_ITER="${LOG_ITER:-50}"
NPROC="${NPROC:-2}"

if [ ! -d "$FASTGEN_DIR" ]; then
    echo "[MeanFlow] FASTGEN_DIR not found: $FASTGEN_DIR"
    exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
    echo "[MeanFlow] MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[MeanFlow] GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[MeanFlow] Model: ${MODEL_PATH}"
echo "[MeanFlow] Data shards: ${DATA_SHARDS}"
echo "[MeanFlow] Iter: ${MAX_ITER}, Save every: ${SAVE_ITER}"

torchrun --nproc_per_node=$NPROC --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/our/exp11_meanflow_video_5000_2gpu.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=2 \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOG_ITER \
      trainer.save_ckpt_iter=$SAVE_ITER \
      trainer.validation_iter=$VAL_ITER \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=$RUN_NAME
