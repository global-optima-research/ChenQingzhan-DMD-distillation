#!/bin/bash
# run_dmd2_fsdp.sh — DMD2 distillation on Wan2.1-1.3B with 4-GPU FSDP
#
# DMD2: VSD loss + GAN adversarial training (4 networks: Student+Teacher+FakeScore+Disc)
# Memory: 4-GPU FSDP + standard AdamW (~25 GB/GPU)
# Note: 8-bit AdamW incompatible with FSDP2 DTensor; 2-GPU OOMs (30.41 GB/GPU)
# Reference: FastGen config_dmd2.py (VBench 83.24, 4-step)
#
# Usage:
#   bash run_dmd2_fsdp.sh          # Full 2000-iter training
#   SMOKE=1 bash run_dmd2_fsdp.sh  # 10-iter smoke test

set -euo pipefail

# === Configuration ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SMOKE="${SMOKE:-0}"

export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

# === Smoke test or full run ===
if [ "$SMOKE" = "1" ]; then
    MAX_ITER=10
    SAVE_ITER=10
    VAL_ITER=10
    LOG_ITER=1
    RUN_NAME="dmd2_smoke_test"
    echo "[DMD2] === SMOKE TEST (10 iter) ==="
else
    MAX_ITER=2000
    SAVE_ITER=200
    VAL_ITER=500
    LOG_ITER=50
    RUN_NAME="dmd2_4gpu_fsdp"
    echo "[DMD2] === FULL TRAINING (2000 iter) ==="
fi

echo "[DMD2] GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[DMD2] 4-GPU FSDP + standard AdamW"
echo "[DMD2] batch_global=4 (1 per GPU), max_iter=${MAX_ITER}"
echo ""

torchrun --nproc_per_node=4 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=4 \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOG_ITER \
      trainer.save_ckpt_iter=$SAVE_ITER \
      trainer.validation_iter=$VAL_ITER \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=$RUN_NAME

echo ""
echo "[DMD2] Training complete! (${MAX_ITER} iterations)"
echo "[DMD2] Checkpoints: ${FASTGEN_OUTPUT_ROOT}/fastgen/wan_dmd2/${RUN_NAME}/checkpoints/"
