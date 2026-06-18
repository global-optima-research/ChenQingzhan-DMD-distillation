#!/bin/bash
# run_causvid_fsdp.sh — CausVid on Wan2.1-1.3B with FSDP
#
# CausVid: Causal DMD (4 networks: CausalStudent+Teacher+FakeScore+Disc)
# Requires pretrained causal student init (ode_init.pt)
#
# Usage:
#   bash run_causvid_fsdp.sh           # Full 6000-iter
#   SMOKE=1 bash run_causvid_fsdp.sh   # 10-iter smoke test

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
SMOKE="${SMOKE:-0}"
NUM_GPUS="${NUM_GPUS:-4}"

export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

if [ "$SMOKE" = "1" ]; then
    MAX_ITER=10; SAVE_ITER=10; VAL_ITER=10; LOG_ITER=1
    RUN_NAME="causvid_smoke"
    echo "[CausVid] === SMOKE TEST ==="
else
    MAX_ITER=6000; SAVE_ITER=1000; VAL_ITER=1000; LOG_ITER=50
    RUN_NAME="causvid_6000iter"
    echo "[CausVid] === FULL TRAINING (6000 iter) ==="
fi

echo "[CausVid] GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"

torchrun --nproc_per_node=$NUM_GPUS --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_causvid.py \
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

echo "[CausVid] Complete! Checkpoints: ${FASTGEN_OUTPUT_ROOT}/fastgen/wan_causvid/${RUN_NAME}/checkpoints/"
