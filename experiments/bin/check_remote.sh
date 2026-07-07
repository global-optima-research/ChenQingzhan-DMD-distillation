#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-experiments/configs/wan21_check.env}"

if [ ! -f "$CONFIG" ]; then
    echo "[check_remote] config not found: $CONFIG" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG"

SSH_HOST="${SSH_HOST:-ust_ip}"
REMOTE_FASTGEN="${REMOTE_FASTGEN:-/data/chenqingzhan/FastGen}"
REMOTE_CONDA="${REMOTE_CONDA:-/data/chenqingzhan/miniconda3}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/data/chenqingzhan/logs}"
HF_HOME_REMOTE="${HF_HOME_REMOTE:-/data/chenqingzhan/.cache/huggingface}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_TAG="${DATA_TAG:-}"
CUDA_DEVICES="${CUDA_DEVICES:-${GPU_ID:-0}}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-100}"

DATA_PATH="${DATA_TAG#WDS:}"

quote() {
    printf "%q" "$1"
}

ssh "$SSH_HOST" "bash -s" <<REMOTE
set -euo pipefail

REMOTE_FASTGEN=$(quote "$REMOTE_FASTGEN")
REMOTE_CONDA=$(quote "$REMOTE_CONDA")
REMOTE_OUTPUT_ROOT=$(quote "$REMOTE_OUTPUT_ROOT")
REMOTE_LOG_DIR=$(quote "$REMOTE_LOG_DIR")
HF_HOME_REMOTE=$(quote "$HF_HOME_REMOTE")
MODEL_PATH=$(quote "$MODEL_PATH")
DATA_PATH=$(quote "$DATA_PATH")
CUDA_DEVICES=$(quote "$CUDA_DEVICES")
GPU_MAX_USED_MB=$(quote "$GPU_MAX_USED_MB")

echo "__REMOTE__ \$(hostname) \$(whoami) \$(date +%F_%T_%Z)"
echo "__PATHS__"
test -d "\$REMOTE_FASTGEN" && echo "ok fastgen=\$REMOTE_FASTGEN"
test -d "\$REMOTE_CONDA" && echo "ok conda=\$REMOTE_CONDA"
test -d "\$REMOTE_OUTPUT_ROOT" && echo "ok output_root=\$REMOTE_OUTPUT_ROOT"
test -d "\$REMOTE_LOG_DIR" && echo "ok log_dir=\$REMOTE_LOG_DIR"
test -d "\$HF_HOME_REMOTE" && echo "ok hf_home=\$HF_HOME_REMOTE"

if [ -n "\$MODEL_PATH" ]; then
    test -e "\$MODEL_PATH" && echo "ok model=\$MODEL_PATH"
fi

if [ -n "\$DATA_PATH" ]; then
    test -e "\$DATA_PATH" && echo "ok data=\$DATA_PATH"
fi

echo "__GPU__"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "__SELECTED_GPU_CHECK__ selected=\$CUDA_DEVICES max_used_mb=\$GPU_MAX_USED_MB"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v ids="\$CUDA_DEVICES" -v max="\$GPU_MAX_USED_MB" '
  BEGIN {
    n=split(ids,a,",");
    for (i=1;i<=n;i++) want[a[i]]=1;
  }
  {
    gsub(/ /,"",\$1);
    gsub(/ /,"",\$2);
    if ((\$1 in want) && ((\$2 + 0) > (max + 0))) {
      print "busy gpu=" \$1 " used_mb=" \$2;
      bad=1;
    }
  }
  END {
    if (bad) exit 2;
    print "selected GPUs pass memory threshold";
  }
'
REMOTE
