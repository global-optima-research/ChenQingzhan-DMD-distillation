#!/usr/bin/env bash
set -euo pipefail

SELECTED=5,6
RUN=wan22_dmd2_65f_cfg5_bs1_5000iter_20260430_g34
LOG=/data/chenqingzhan/logs/${RUN}_resume_from5000_20260501_g56.log
FASTGEN=/data/chenqingzhan/FastGen

echo "__PRECHECK__ $(date +%F_%T_%Z)"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v ids="$SELECTED" '
  BEGIN {
    n=split(ids,a,",");
    for (i=1;i<=n;i++) want[a[i]]=1;
  }
  {
    gsub(/ /,"",$1);
    gsub(/ /,"",$2);
    if (($1 in want) && $2 > 500) {
      print "GPU " $1 " is not free: " $2 " MiB";
      bad=1;
    }
  }
  END { exit bad }
'

cd "$FASTGEN"

export CUDA_VISIBLE_DEVICES=$SELECTED
export FASTGEN_OUTPUT_ROOT=/data/chenqingzhan/fastgen_output
export HF_HOME=/data/chenqingzhan/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data/chenqingzhan/FastGen
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export FASTGEN_DISABLE_MEDIA_LOGGING=true

nohup /data/chenqingzhan/miniconda3/envs/fastgen/bin/torchrun --standalone --nproc_per_node=2 train.py \
  --config=fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py \
  - \
  trainer.ddp=False \
  trainer.fsdp=True \
  trainer.fsdp_cpu_offload=True \
  trainer.resume=True \
  trainer.batch_size_global=2 \
  trainer.max_iter=1000000 \
  trainer.logging_iter=100 \
  trainer.save_ckpt_iter=1000 \
  dataloader_train.batch_size=1 \
  'dataloader_train.datatags=["WDS:/data/datasets/OpenVid-1M/webdataset"]' \
  dataloader_train.sequence_length=65 \
  'model.input_shape=[48,17,44,80]' \
  model.net.model_id_or_local_path=/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers \
  model.student_sample_steps=2 \
  model.guidance_scale=5.0 \
  log_config.name="$RUN" \
  log_config.wandb_mode=disabled \
  ~trainer.callbacks.wandb > "$LOG" 2>&1 &

PID=$!
echo "__LAUNCHED__ pid=$PID run=$RUN log=$LOG"
sleep 10
echo "__LOG_HEAD__"
sed -n '1,100p' "$LOG" || true
