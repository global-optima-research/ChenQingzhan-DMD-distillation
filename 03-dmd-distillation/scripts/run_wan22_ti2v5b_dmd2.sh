#!/bin/bash
# run_wan22_ti2v5b_dmd2.sh — DMD2 launcher skeleton for FastGen Wan2.2 TI2V 5B
# Usage:
#   bash run_wan22_ti2v5b_dmd2.sh smoke  [data_dir_or_wds_tag]
#   bash run_wan22_ti2v5b_dmd2.sh stage1 [data_dir_or_wds_tag]
#   bash run_wan22_ti2v5b_dmd2.sh formal [data_dir_or_wds_tag]
#
# Notes:
# - This script assumes the server-side FastGen tree already contains:
#   fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py
# - The server-side native config currently uses 2-step training by default.
# - The validated route uses torchrun + FSDP + CPU offload, with one sample per GPU.

set -euo pipefail

MODE="${1:-smoke}"
DATA_DIR="${2:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export FASTGEN_OUTPUT_ROOT="${FASTGEN_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
export HF_HOME="${HF_HOME:-/data/chenqingzhan/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Honored by the server-side WandbCallback patch. Harmless in unpatched trees.
export FASTGEN_DISABLE_MEDIA_LOGGING="${FASTGEN_DISABLE_MEDIA_LOGGING:-true}"

CONDA_DIR="/data/chenqingzhan/miniconda3"
FASTGEN_DIR="${FASTGEN_DIR:-/data/chenqingzhan/FastGen}"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"
TORCHRUN="$CONDA_DIR/envs/fastgen/bin/torchrun"
CONFIG_PATH="${CONFIG_PATH:-fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py}"
MODEL_PATH="${MODEL_PATH:-/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers}"

RUN_NAME_BASE="${RUN_NAME_BASE:-wan22_ti2v5b_dmd2}"
LOGGING_ITER="${LOGGING_ITER:-10}"
STUDENT_SAMPLE_STEPS="${STUDENT_SAMPLE_STEPS:-2}"
# Native guidance_scale=5.0 OOMs at the first real DMD2 student update on
# 2x RTX 5090 32GB. Use GUIDANCE_SCALE=5.0 only when re-testing that path.
GUIDANCE_SCALE="${GUIDANCE_SCALE:-null}"
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-True}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DISABLE_WANDB_CALLBACK="${DISABLE_WANDB_CALLBACK:-1}"
WANDB_SAMPLE_LOGGING_ITER="${WANDB_SAMPLE_LOGGING_ITER:-1000000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}"

make_wds_tag() {
    case "$1" in
        WDS:*) printf '%s' "$1" ;;
        *) printf 'WDS:%s' "$1" ;;
    esac
}

case "$MODE" in
    smoke)
        MAX_ITER="${MAX_ITER:-10}"
        SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-10}"
        BATCH_SIZE_GLOBAL="${BATCH_SIZE_GLOBAL:-$NPROC_PER_NODE}"
        RUN_NAME="${RUN_NAME:-${RUN_NAME_BASE}_smoke}"
        ;;
    stage1)
        MAX_ITER="${MAX_ITER:-1000}"
        SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-100}"
        BATCH_SIZE_GLOBAL="${BATCH_SIZE_GLOBAL:-$NPROC_PER_NODE}"
        RUN_NAME="${RUN_NAME:-${RUN_NAME_BASE}_1000iter}"
        ;;
    formal)
        MAX_ITER="${MAX_ITER:-5000}"
        SAVE_CKPT_ITER="${SAVE_CKPT_ITER:-500}"
        BATCH_SIZE_GLOBAL="${BATCH_SIZE_GLOBAL:-$NPROC_PER_NODE}"
        RUN_NAME="${RUN_NAME:-${RUN_NAME_BASE}_5000iter}"
        ;;
    *)
        echo "Usage: bash run_wan22_ti2v5b_dmd2.sh {smoke|stage1|formal} [data_dir_or_wds_tag]"
        exit 1
        ;;
esac

if [ -z "$DATA_DIR" ]; then
    echo "[Wan22-DMD2] DATA_DIR is required."
    echo "Pass a dataset directory or a full WDS tag as the second argument."
    exit 1
fi

DATA_TAG="$(make_wds_tag "$DATA_DIR")"
LOG_PATH="${LOG_PATH:-$FASTGEN_OUTPUT_ROOT/${RUN_NAME}.log}"

if [ ! -d "$FASTGEN_DIR" ]; then
    echo "[Wan22-DMD2] FASTGEN_DIR not found: $FASTGEN_DIR"
    exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
    echo "[Wan22-DMD2] MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

echo "[Wan22-DMD2] mode=$MODE"
echo "[Wan22-DMD2] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[Wan22-DMD2] nproc_per_node=$NPROC_PER_NODE"
echo "[Wan22-DMD2] config=$CONFIG_PATH"
echo "[Wan22-DMD2] model=$MODEL_PATH"
echo "[Wan22-DMD2] datatag=$DATA_TAG"
echo "[Wan22-DMD2] run_name=$RUN_NAME"
echo "[Wan22-DMD2] student_sample_steps=$STUDENT_SAMPLE_STEPS"
echo "[Wan22-DMD2] guidance_scale=$GUIDANCE_SCALE"
echo "[Wan22-DMD2] fastgen_disable_media_logging=$FASTGEN_DISABLE_MEDIA_LOGGING"

OPTS=(
    -
    trainer.ddp=False
    trainer.fsdp=True
    trainer.fsdp_cpu_offload="$FSDP_CPU_OFFLOAD"
    trainer.batch_size_global="$BATCH_SIZE_GLOBAL"
    trainer.max_iter="$MAX_ITER"
    trainer.logging_iter="$LOGGING_ITER"
    trainer.save_ckpt_iter="$SAVE_CKPT_ITER"
    dataloader_train.batch_size=1
    "dataloader_train.datatags=[\"$DATA_TAG\"]"
    model.net.model_id_or_local_path="$MODEL_PATH"
    model.student_sample_steps="$STUDENT_SAMPLE_STEPS"
    model.guidance_scale="$GUIDANCE_SCALE"
    log_config.name="$RUN_NAME"
    log_config.wandb_mode="$WANDB_MODE"
)

if [ "$DISABLE_WANDB_CALLBACK" = "1" ]; then
    OPTS+=(~trainer.callbacks.wandb)
else
    OPTS+=(trainer.callbacks.wandb.sample_logging_iter="$WANDB_SAMPLE_LOGGING_ITER")
fi

"$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
    --config="$CONFIG_PATH" \
    "${OPTS[@]}" \
    2>&1 | tee "$LOG_PATH"
