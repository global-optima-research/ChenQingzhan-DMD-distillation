#!/bin/bash
# run_fdistill_fsdp.sh — f-distill on Wan2.1-1.3B with 3-GPU FSDP
#
# f-distill: f-divergence weighted DMD2 (4 networks: Student+Teacher+FakeScore+Disc)
# Reference: FastGen config_fdistill.py
#
# Usage:
#   bash run_fdistill_fsdp.sh          # Full 6000-iter training
#   SMOKE=1 bash run_fdistill_fsdp.sh  # 10-iter smoke test

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,4,7}"
SMOKE="${SMOKE:-0}"

export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"
NUM_GPUS=3

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

if [ "$SMOKE" = "1" ]; then
    MAX_ITER=10
    SAVE_ITER=10
    VAL_ITER=10
    LOG_ITER=1
    RUN_NAME="fdistill_smoke"
    echo "[f-distill] === SMOKE TEST (10 iter) ==="
else
    MAX_ITER=6000
    SAVE_ITER=1000
    VAL_ITER=1000
    LOG_ITER=50
    RUN_NAME="fdistill_6000iter"
    echo "[f-distill] === FULL TRAINING (6000 iter) ==="
fi

echo "[f-distill] GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "[f-distill] batch_global=${NUM_GPUS}, max_iter=${MAX_ITER}"
echo ""

torchrun --nproc_per_node=$NUM_GPUS --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_fdistill.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=$NUM_GPUS \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOG_ITER \
      trainer.save_ckpt_iter=$SAVE_ITER \
      trainer.validation_iter=$VAL_ITER \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=$RUN_NAME

echo ""
echo "[f-distill] Training complete! (${MAX_ITER} iterations)"
echo "[f-distill] Checkpoints: ${FASTGEN_OUTPUT_ROOT}/fastgen/wan_fdistill/${RUN_NAME}/checkpoints/"
