#!/bin/bash
# FFGO-compatible DMD2 feasibility run.
# Uses a backbone-compatible placeholder teacher until merged FFGO weights are ready.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
export HF_HOME="${HF_HOME:-/data/chenqingzhan/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FASTGEN_DIR="${FASTGEN_DIR:-/data/chenqingzhan/FastGen}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers}"
FFGO_TEACHER_PATH="${FFGO_TEACHER_PATH:-$STUDENT_MODEL_PATH}"
DATA_SHARDS="${DATA_SHARDS:-WDS:/data/datasets/OpenVid-1M/webdataset}"
RUN_NAME="${RUN_NAME:-ffgo_dmd2_32step_5000iter}"
MAX_ITER="${MAX_ITER:-5000}"
SAVE_ITER="${SAVE_ITER:-500}"
VAL_ITER="${VAL_ITER:-500}"
LOG_ITER="${LOG_ITER:-50}"
NPROC="${NPROC:-4}"

if [ ! -d "$FASTGEN_DIR" ]; then
    echo "[FFGO-DMD2] FASTGEN_DIR not found: $FASTGEN_DIR"
    exit 1
fi

if [ ! -e "$STUDENT_MODEL_PATH" ]; then
    echo "[FFGO-DMD2] STUDENT_MODEL_PATH not found: $STUDENT_MODEL_PATH"
    exit 1
fi

if [ ! -e "$FFGO_TEACHER_PATH" ]; then
    echo "[FFGO-DMD2] FFGO_TEACHER_PATH not found: $FFGO_TEACHER_PATH"
    exit 1
fi

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[FFGO-DMD2] GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[FFGO-DMD2] Student base: ${STUDENT_MODEL_PATH}"
echo "[FFGO-DMD2] Teacher path: ${FFGO_TEACHER_PATH}"
echo "[FFGO-DMD2] Data shards: ${DATA_SHARDS}"
echo "[FFGO-DMD2] Steps: 32, Iter: ${MAX_ITER}, Save every: ${SAVE_ITER}"

if [ "$FFGO_TEACHER_PATH" = "$STUDENT_MODEL_PATH" ]; then
    echo "[FFGO-DMD2] NOTE: using the backbone-compatible placeholder teacher path."
    echo "[FFGO-DMD2] Replace FFGO_TEACHER_PATH with merged FFGO weights once available."
fi

torchrun --nproc_per_node=$NPROC --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/our/exp09_ffgo_dmd2_32step_5000.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=4 \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOG_ITER \
      trainer.save_ckpt_iter=$SAVE_ITER \
      trainer.validation_iter=$VAL_ITER \
      model.net.model_id_or_local_path=$STUDENT_MODEL_PATH \
      model.teacher.model_id_or_local_path=$FFGO_TEACHER_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=$RUN_NAME
