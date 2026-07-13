#!/bin/bash
# run_meanflow_single_gpu.sh — Run MeanFlow on Wan2.1-1.3B with single GPU
# Usage: bash run_meanflow_single_gpu.sh [latent_data_dir_or_wds_tag]
#
# MeanFlow: Consistency model family, learns mean velocity between trajectory points
# Data format: VideoLatentLoaderConfig (pre-computed latent.pth + txt_emb.pth)
# NOTE: Different data format from DMD2/f-distill/LADD — needs pre-computed VAE latents
#       See convert_parquet_to_wds.py for one latent-to-WDS conversion path.
# Reference: Geng et al., 2025 (https://arxiv.org/abs/2505.13447)

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
LATENT_DATA_DIR="${1:-/data/chenqingzhan/training_data/latent_shards}"
RUN_NAME="${RUN_NAME:-meanflow_wan1.3b_single_gpu}"
MAX_ITER="${MAX_ITER:-6000}"
LOGGING_ITER="${LOGGING_ITER:-50}"
SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
BATCH_SIZE_GLOBAL="${BATCH_SIZE_GLOBAL:-8}"
LOG_PATH="${LOG_PATH:-$FASTGEN_OUTPUT_ROOT/meanflow_train.log}"

make_wds_tag() {
    case "$1" in
        WDS:*) printf '%s' "$1" ;;
        *) printf 'WDS:%s' "$1" ;;
    esac
}

LATENT_DATA_TAG="$(make_wds_tag "$LATENT_DATA_DIR")"

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

$PYTHON train.py \
    --config=fastgen/configs/experiments/WanT2V/config_mf.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=$BATCH_SIZE_GLOBAL \
      trainer.max_iter=$MAX_ITER \
      trainer.logging_iter=$LOGGING_ITER \
      trainer.save_ckpt_iter=$SAVE_CKPT_ITER \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$LATENT_DATA_TAG\"]" \
      log_config.name=$RUN_NAME \
      2>&1 | tee "$LOG_PATH"
