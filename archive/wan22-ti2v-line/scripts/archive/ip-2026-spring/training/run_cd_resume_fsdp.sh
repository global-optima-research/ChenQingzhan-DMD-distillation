#!/bin/bash
# run_cd_resume_fsdp.sh — Resume CD from 2000iter FSDP checkpoint, train to 5000
#
# 2-GPU FSDP, kimg_per_stage=5, checkpoint every 1000 iter
# Resumes from: cd_5000iter_fsdp_kimg5/checkpoints/0002000.pth
#
# Usage: bash run_cd_resume_fsdp.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,5}"
echo "[CD-Resume] Using GPUs: ${CUDA_VISIBLE_DEVICES}"

export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[CD-Resume] Resuming from 2000 iter, target 5000 iter"
echo "[CD-Resume] 2-GPU FSDP, kimg_per_stage=5, batch=2"
echo ""

torchrun --nproc_per_node=2 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_cm_cd.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=2 \
      trainer.max_iter=5000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=1000 \
      trainer.validation_iter=1000 \
      trainer.resume=True \
      trainer.callbacks.ct_schedule.kimg_per_stage=5 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=cd_5000iter_fsdp_kimg5

echo "[CD-Resume] Training complete!"
