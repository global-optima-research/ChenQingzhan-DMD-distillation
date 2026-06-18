#!/bin/bash
# run_cd_single_gpu.sh — Run CD (Consistency Distillation) on Wan2.1-1.3B with single GPU
# Usage: bash run_cd_single_gpu.sh [data_dir_or_wds_tag]
#
# CD: Consistency Distillation with teacher (use_cd=True)
# Same data format as DMD2 (VideoLoaderConfig: mp4+txt WebDataset)
# Config: custom config_cm_cd.py (adapted from EDM2 CM + WanT2V DMD2)
# Reference: Song et al., 2023 (https://arxiv.org/abs/2303.01469)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
export HF_HOME="${HF_HOME:-/data/chenqingzhan/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

FASTGEN_DIR="${FASTGEN_DIR:-/data/chenqingzhan/FastGen}"
PYTHON="/data/chenqingzhan/miniconda3/envs/fastgen/bin/python"
MODEL_PATH="${MODEL_PATH:-/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$MODEL_PATH}"
CONFIG_PATH="${CONFIG_PATH:-fastgen/configs/experiments/WanT2V/config_cm_cd.py}"
DATA_DIR="${1:-/data/datasets/OpenVid-1M/webdataset}"
RUN_NAME="${RUN_NAME:-cd_wan1.3b_single_gpu}"
LOG_PATH="${LOG_PATH:-$FASTGEN_OUTPUT_ROOT/cd_train.log}"

make_wds_tag() {
    case "$1" in
        WDS:*) printf '%s' "$1" ;;
        *) printf 'WDS:%s' "$1" ;;
    esac
}

DATA_TAG="$(make_wds_tag "$DATA_DIR")"

if [ ! -d "$FASTGEN_DIR" ]; then
    echo "[CD] FASTGEN_DIR not found: $FASTGEN_DIR"
    exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
    echo "[CD] MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

$PYTHON train.py \
    --config=$CONFIG_PATH \
    - trainer.ddp=False \
      trainer.fsdp=False \
      model.net.model_id_or_local_path=$MODEL_PATH \
      model.teacher.model_id_or_local_path=$TEACHER_MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_TAG\"]" \
      log_config.name=$RUN_NAME \
      2>&1 | tee "$LOG_PATH"
