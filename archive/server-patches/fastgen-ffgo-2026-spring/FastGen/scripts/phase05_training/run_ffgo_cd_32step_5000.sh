#!/bin/bash
# FFGO-compatible CD stage run on 2 GPUs.
# Uses a backbone-compatible placeholder teacher until merged FFGO weights are ready.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
export HF_HOME="${HF_HOME:-/data/chenqingzhan/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FASTGEN_DIR="${FASTGEN_DIR:-/data/chenqingzhan/FastGen}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers}"
FFGO_TEACHER_PATH="${FFGO_TEACHER_PATH:-$STUDENT_MODEL_PATH}"
DATA_SHARDS="${DATA_SHARDS:-WDS:/data/datasets/OpenVid-1M/webdataset}"
RUN_NAME="${RUN_NAME:-ffgo_cd_32step_5000iter_2gpu}"
MAX_ITER="${MAX_ITER:-5000}"
SAVE_ITER="${SAVE_ITER:-500}"
VAL_ITER="${VAL_ITER:-500}"
LOG_ITER="${LOG_ITER:-50}"
NPROC="${NPROC:-2}"

if [ ! -d "$FASTGEN_DIR" ]; then
    echo "[FFGO-CD] FASTGEN_DIR not found: $FASTGEN_DIR"
    exit 1
fi

if [ ! -e "$STUDENT_MODEL_PATH" ]; then
    echo "[FFGO-CD] STUDENT_MODEL_PATH not found: $STUDENT_MODEL_PATH"
    exit 1
fi

if [ ! -e "$FFGO_TEACHER_PATH" ]; then
    echo "[FFGO-CD] FFGO_TEACHER_PATH not found: $FFGO_TEACHER_PATH"
    exit 1
fi

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[FFGO-CD] GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[FFGO-CD] Student base: ${STUDENT_MODEL_PATH}"
echo "[FFGO-CD] Teacher path: ${FFGO_TEACHER_PATH}"
echo "[FFGO-CD] Data shards: ${DATA_SHARDS}"
echo "[FFGO-CD] Steps: 32, Iter: ${MAX_ITER}, Save every: ${SAVE_ITER}"

if [ "$FFGO_TEACHER_PATH" = "$STUDENT_MODEL_PATH" ]; then
    echo "[FFGO-CD] NOTE: using the backbone-compatible placeholder teacher path."
    echo "[FFGO-CD] Replace FFGO_TEACHER_PATH with merged FFGO weights once available."
fi

torchrun --nproc_per_node=$NPROC --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/our/exp10_ffgo_cd_32step_5000.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=2 \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOG_ITER \
      trainer.save_ckpt_iter=$SAVE_ITER \
      trainer.validation_iter=$VAL_ITER \
      model.net.model_id_or_local_path=$STUDENT_MODEL_PATH \
      model.teacher.model_id_or_local_path=$FFGO_TEACHER_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=$RUN_NAME
