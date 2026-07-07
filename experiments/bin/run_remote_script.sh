#!/usr/bin/env bash
set -euo pipefail

# Generic one-line launcher for the current WanT2V line.
# It runs a parameterized remote script that lives inside the FastGen repo
# (e.g. fastgen/configs/experiments/WanT2V/run_*.sh) with env overrides.
#
# Usage:
#   bash experiments/bin/run_remote_script.sh [--dry-run] experiments/configs/<config>.env
#
# Required in the .env:
#   REMOTE_SCRIPT   path of the remote script, relative to REMOTE_REPO or absolute
# Optional in the .env:
#   SSH_HOST        default ust_ip
#   REMOTE_REPO     default /data/chenqingzhan/FastGen
#   RUN_NAME        default = config basename; names the wrapper log
#   CUDA_DEVICES    if set, GPU precheck runs on these ids before launch
#   GPU_MAX_USED_MB precheck threshold, default 100
#   REMOTE_ENV      single-quoted "VAR=value VAR2=\"a b\"" string, eval-exported remotely
#   DETACH          1 (default) = nohup background + pid; 0 = stream in foreground
#   WRAPPER_LOG_DIR default /data/chenqingzhan/logs

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

CONFIG="${1:?usage: run_remote_script.sh [--dry-run] <config.env>}"
if [ ! -f "$CONFIG" ]; then
    echo "[run_remote_script] config not found: $CONFIG" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG"

SSH_HOST="${SSH_HOST:-ust_ip}"
REMOTE_REPO="${REMOTE_REPO:-/data/chenqingzhan/FastGen}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:?REMOTE_SCRIPT is required in the config}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG" .env)}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-100}"
REMOTE_ENV="${REMOTE_ENV:-}"
DETACH="${DETACH:-1}"
WRAPPER_LOG_DIR="${WRAPPER_LOG_DIR:-/data/chenqingzhan/logs}"

quote() {
    printf "%q" "$1"
}

ssh "$SSH_HOST" "bash -s" <<REMOTE
set -euo pipefail

DRY_RUN=$(quote "$DRY_RUN")
REMOTE_REPO=$(quote "$REMOTE_REPO")
REMOTE_SCRIPT=$(quote "$REMOTE_SCRIPT")
RUN_NAME=$(quote "$RUN_NAME")
CUDA_DEVICES=$(quote "$CUDA_DEVICES")
GPU_MAX_USED_MB=$(quote "$GPU_MAX_USED_MB")
REMOTE_ENV=$(quote "$REMOTE_ENV")
DETACH=$(quote "$DETACH")
WRAPPER_LOG_DIR=$(quote "$WRAPPER_LOG_DIR")

cd "\$REMOTE_REPO"

case "\$REMOTE_SCRIPT" in
    /*) SCRIPT_PATH="\$REMOTE_SCRIPT" ;;
    *)  SCRIPT_PATH="\$REMOTE_REPO/\$REMOTE_SCRIPT" ;;
esac

if [ ! -f "\$SCRIPT_PATH" ]; then
    echo "__ERROR__ remote script not found: \$SCRIPT_PATH" >&2
    exit 1
fi
bash -n "\$SCRIPT_PATH"

echo "__REMOTE__ \$(hostname) \$(date +%F_%T_%Z)"
echo "__SCRIPT__ \$SCRIPT_PATH"
echo "__REMOTE_ENV__ \$REMOTE_ENV"

if [ -n "\$CUDA_DEVICES" ]; then
    echo "__GPU_PRECHECK__ selected=\$CUDA_DEVICES max_used_mb=\$GPU_MAX_USED_MB"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v ids="\$CUDA_DEVICES" -v max="\$GPU_MAX_USED_MB" '
      BEGIN {
        n=split(ids,a,",");
        for (i=1;i<=n;i++) want[a[i]]=1;
      }
      {
        gsub(/ /,"",\$1);
        gsub(/ /,"",\$2);
        if ((\$1 in want) && ((\$2 + 0) > (max + 0))) {
          print "__ERROR__ GPU " \$1 " is not free: " \$2 " MiB";
          bad=1;
        }
      }
      END { exit bad }
    '
fi

if [ -n "\$REMOTE_ENV" ]; then
    eval "export \$REMOTE_ENV"
fi

WRAPPER_LOG="\$WRAPPER_LOG_DIR/\${RUN_NAME}_\$(date +%Y%m%d_%H%M%S).log"

echo "__COMMAND__ bash \$SCRIPT_PATH"
echo "__WRAPPER_LOG__ \$WRAPPER_LOG"

if [ "\$DRY_RUN" = "1" ]; then
    echo "__DRY_RUN__ nothing launched"
    exit 0
fi

mkdir -p "\$WRAPPER_LOG_DIR"

if [ "\$DETACH" = "1" ]; then
    nohup bash "\$SCRIPT_PATH" > "\$WRAPPER_LOG" 2>&1 &
    echo "__LAUNCHED__ pid=\$! run=\$RUN_NAME wrapper_log=\$WRAPPER_LOG"
else
    bash "\$SCRIPT_PATH" 2>&1 | tee "\$WRAPPER_LOG"
fi
REMOTE
