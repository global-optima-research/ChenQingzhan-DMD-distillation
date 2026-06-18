#!/bin/bash
# run_dmd2_inference_all.sh — Consolidate FSDP checkpoints & run 4-step inference
#
# Consolidates 5 DMD2 FSDP checkpoints (400,800,1200,1600,2000 iter)
# then runs 4-step student inference on each with a shared prompt.
#
# Usage: CUDA_VISIBLE_DEVICES=2 bash run_dmd2_inference_all.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
CKPT_DIR="$FASTGEN_OUTPUT_ROOT/fastgen/wan_dmd2/dmd2_4gpu_fsdp/checkpoints"
OUT_DIR="$FASTGEN_OUTPUT_ROOT/dmd2_consolidated"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

# Create prompt file
PROMPT_FILE="/tmp/dmd2_eval_prompt.txt"
echo "A cat walking on the grass in the sunshine" > "$PROMPT_FILE"

ITERS="400 800 1200 1600 2000"

# === Step 1: Consolidate FSDP checkpoints ===
echo "[DMD2-Infer] Consolidating FSDP checkpoints..."
mkdir -p "$OUT_DIR"

for ITER in $ITERS; do
    ITER_PAD=$(printf "%07d" $ITER)
    CONSOLIDATED="$OUT_DIR/dmd2_${ITER}iter.pth"

    if [ -f "$CONSOLIDATED" ]; then
        echo "  [${ITER}iter] Already consolidated, skip."
        continue
    fi

    echo "  [${ITER}iter] Consolidating net_model..."
    python3 -c "
import torch, torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

base = '${CKPT_DIR}'
iter_pad = '${ITER_PAD}'

# Load FSDP sharded net_model
reader = FileSystemReader(f'{base}/{iter_pad}.net_model')
md = reader.read_metadata()
net = {k: torch.empty(m.size) for k, m in md.state_dict_metadata.items() if hasattr(m, 'size')}
dcp.load(net, storage_reader=reader)

# Load metadata
meta = torch.load(f'{base}/{iter_pad}.pth', map_location='cpu', weights_only=False)

# Save consolidated
torch.save({
    'model': {'net': net},
    'scheduler': meta.get('scheduler', {}),
    'grad_scaler': meta.get('grad_scaler', {}),
    'callbacks': meta.get('callbacks', {}),
    'iteration': meta.get('iteration', ${ITER}),
}, '${CONSOLIDATED}')
print(f'  Saved ${CONSOLIDATED}')
"
done

echo ""
echo "[DMD2-Infer] All checkpoints consolidated."
echo ""

# === Step 2: Run 4-step inference on each ===
for ITER in $ITERS; do
    CONSOLIDATED="$OUT_DIR/dmd2_${ITER}iter.pth"
    RUN_NAME="dmd2_${ITER}iter_4step"

    echo "[DMD2-Infer] Running 4-step inference: ${ITER} iter..."
    python scripts/inference/video_model_inference.py \
        --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
        --do_student_sampling True \
        --do_teacher_sampling False \
        --ckpt_path="$CONSOLIDATED" \
        --prompt_file="$PROMPT_FILE" \
        - trainer.ddp=False \
          trainer.seed=42 \
          model.student_sample_steps=4 \
          model.guidance_scale=5.0 \
          model.net.model_id_or_local_path="$MODEL_PATH" \
          log_config.name="$RUN_NAME"

    echo "  [${ITER}iter] Inference complete."
    echo ""
done

echo "[DMD2-Infer] All done! Videos saved under $FASTGEN_OUTPUT_ROOT"
