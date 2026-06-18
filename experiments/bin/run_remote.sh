#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

CONFIG="${1:-experiments/configs/wan22_dmd2_no_cfg_stage1.env}"

if [ ! -f "$CONFIG" ]; then
    echo "[run_remote] config not found: $CONFIG" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG"

SSH_HOST="${SSH_HOST:-ust_ip}"
EXPERIMENT_KIND="${EXPERIMENT_KIND:-train}"
EXPERIMENT_ID="${EXPERIMENT_ID:?EXPERIMENT_ID is required}"
RUN_NAME="${RUN_NAME:-$EXPERIMENT_ID}"

REMOTE_FASTGEN="${REMOTE_FASTGEN:-/data/chenqingzhan/FastGen}"
REMOTE_CONDA="${REMOTE_CONDA:-/data/chenqingzhan/miniconda3}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/data/chenqingzhan/fastgen_output}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/data/chenqingzhan/logs}"
HF_HOME_REMOTE="${HF_HOME_REMOTE:-/data/chenqingzhan/.cache/huggingface}"
HF_ENDPOINT_REMOTE="${HF_ENDPOINT_REMOTE:-https://hf-mirror.com}"
CONFIG_PATH="${CONFIG_PATH:-}"
MODEL_PATH="${MODEL_PATH:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"

quote() {
    printf "%q" "$1"
}

ssh "$SSH_HOST" "bash -s" <<REMOTE
set -euo pipefail

DRY_RUN=$(quote "$DRY_RUN")
EXPERIMENT_KIND=$(quote "$EXPERIMENT_KIND")
EXPERIMENT_ID=$(quote "$EXPERIMENT_ID")
RUN_NAME=$(quote "$RUN_NAME")
REMOTE_FASTGEN=$(quote "$REMOTE_FASTGEN")
REMOTE_CONDA=$(quote "$REMOTE_CONDA")
REMOTE_OUTPUT_ROOT=$(quote "$REMOTE_OUTPUT_ROOT")
REMOTE_LOG_DIR=$(quote "$REMOTE_LOG_DIR")
HF_HOME_REMOTE=$(quote "$HF_HOME_REMOTE")
HF_ENDPOINT_REMOTE=$(quote "$HF_ENDPOINT_REMOTE")
CONFIG_PATH=$(quote "$CONFIG_PATH")
MODEL_PATH=$(quote "$MODEL_PATH")
WANDB_MODE=$(quote "$WANDB_MODE")

CUDA_DEVICES=$(quote "${CUDA_DEVICES:-}")
GPU_ID=$(quote "${GPU_ID:-0}")
GPU_MAX_USED_MB=$(quote "${GPU_MAX_USED_MB:-100}")
NPROC_PER_NODE=$(quote "${NPROC_PER_NODE:-}")
DETACH=$(quote "${DETACH:-1}")
DATA_TAG=$(quote "${DATA_TAG:-}")
MAX_ITER=$(quote "${MAX_ITER:-}")
SAVE_CKPT_ITER=$(quote "${SAVE_CKPT_ITER:-}")
LOGGING_ITER=$(quote "${LOGGING_ITER:-100}")
BATCH_SIZE_GLOBAL=$(quote "${BATCH_SIZE_GLOBAL:-}")
DATALOADER_BATCH_SIZE=$(quote "${DATALOADER_BATCH_SIZE:-1}")
SEQUENCE_LENGTH=$(quote "${SEQUENCE_LENGTH:-}")
INPUT_SHAPE=$(quote "${INPUT_SHAPE:-}")
STUDENT_SAMPLE_STEPS=$(quote "${STUDENT_SAMPLE_STEPS:-2}")
STUDENT_SAMPLE_TYPE=$(quote "${STUDENT_SAMPLE_TYPE:-ode}")
SAMPLE_T_LIST=$(quote "${SAMPLE_T_LIST:-}")
GUIDANCE_SCALE=$(quote "${GUIDANCE_SCALE:-null}")
FSDP_CPU_OFFLOAD=$(quote "${FSDP_CPU_OFFLOAD:-True}")
RESUME=$(quote "${RESUME:-False}")
DISABLE_WANDB_CALLBACK=$(quote "${DISABLE_WANDB_CALLBACK:-1}")
WANDB_SAMPLE_LOGGING_ITER=$(quote "${WANDB_SAMPLE_LOGGING_ITER:-1000000}")
LR_NET=$(quote "${LR_NET:-}")
LR_FAKE_SCORE=$(quote "${LR_FAKE_SCORE:-}")
LR_DISCRIMINATOR=$(quote "${LR_DISCRIMINATOR:-}")

CKPT_PATH=$(quote "${CKPT_PATH:-}")
PROMPTS_FILE=$(quote "${PROMPTS_FILE:-}")
IMAGES_FILE=$(quote "${IMAGES_FILE:-}")
INFERENCE_PY=$(quote "${INFERENCE_PY:-/data/chenqingzhan/scripts/video_model_inference_decode_offload.py}")
REMOTE_INFERENCE_ROOT=$(quote "${REMOTE_INFERENCE_ROOT:-/data/chenqingzhan/inference_outputs}")
SAMPLE_COUNT=$(quote "${SAMPLE_COUNT:-1}")
FPS=$(quote "${FPS:-16}")

PYTHON="\$REMOTE_CONDA/envs/fastgen/bin/python"
TORCHRUN="\$REMOTE_CONDA/envs/fastgen/bin/torchrun"
LOG_PATH="\$REMOTE_LOG_DIR/\${RUN_NAME}.log"

count_gpus() {
    awk -F, '{print NF}' <<< "\$1"
}

check_selected_gpus() {
    local selected="\$1"
    local max_used="\$2"
    echo "__GPU_PRECHECK__ \$(date +%F_%T_%Z) selected=\$selected max_used_mb=\$max_used"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v ids="\$selected" -v max="\$max_used" '
      BEGIN {
        n=split(ids,a,",");
        for (i=1;i<=n;i++) want[a[i]]=1;
      }
      {
        gsub(/ /,"",\$1);
        gsub(/ /,"",\$2);
        if ((\$1 in want) && ((\$2 + 0) > (max + 0))) {
          print "GPU " \$1 " is not free: " \$2 " MiB";
          bad=1;
        }
      }
      END { exit bad }
    '
}

print_command() {
    printf '%q ' "\$@"
    printf '\n'
}

mkdir -p "\$REMOTE_LOG_DIR"

case "\$EXPERIMENT_KIND" in
    train)
        : "\${CUDA_DEVICES:?CUDA_DEVICES is required for train}"
        : "\${CONFIG_PATH:?CONFIG_PATH is required for train}"
        : "\${MODEL_PATH:?MODEL_PATH is required for train}"
        : "\${DATA_TAG:?DATA_TAG is required for train}"
        : "\${MAX_ITER:?MAX_ITER is required for train}"

        if [ -z "\$NPROC_PER_NODE" ]; then
            NPROC_PER_NODE="\$(count_gpus "\$CUDA_DEVICES")"
        fi
        if [ -z "\$BATCH_SIZE_GLOBAL" ]; then
            BATCH_SIZE_GLOBAL="\$NPROC_PER_NODE"
        fi
        if [ -z "\$SAVE_CKPT_ITER" ]; then
            SAVE_CKPT_ITER="\$MAX_ITER"
        fi

        check_selected_gpus "\$CUDA_DEVICES" "\$GPU_MAX_USED_MB"
        cd "\$REMOTE_FASTGEN"

        export CUDA_VISIBLE_DEVICES="\$CUDA_DEVICES"
        export FASTGEN_OUTPUT_ROOT="\$REMOTE_OUTPUT_ROOT"
        export HF_HOME="\$HF_HOME_REMOTE"
        export HF_ENDPOINT="\$HF_ENDPOINT_REMOTE"
        export PYTHONPATH="\$REMOTE_FASTGEN"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export TOKENIZERS_PARALLELISM=false
        export FASTGEN_DISABLE_MEDIA_LOGGING=true

        cmd=(
            "\$TORCHRUN" --standalone --nproc_per_node="\$NPROC_PER_NODE" train.py
            --config="\$CONFIG_PATH"
            -
            trainer.ddp=False
            trainer.fsdp=True
            trainer.fsdp_cpu_offload="\$FSDP_CPU_OFFLOAD"
            trainer.resume="\$RESUME"
            trainer.batch_size_global="\$BATCH_SIZE_GLOBAL"
            trainer.max_iter="\$MAX_ITER"
            trainer.logging_iter="\$LOGGING_ITER"
            trainer.save_ckpt_iter="\$SAVE_CKPT_ITER"
            dataloader_train.batch_size="\$DATALOADER_BATCH_SIZE"
            "dataloader_train.datatags=[\"\$DATA_TAG\"]"
            model.net.model_id_or_local_path="\$MODEL_PATH"
            model.student_sample_steps="\$STUDENT_SAMPLE_STEPS"
            model.guidance_scale="\$GUIDANCE_SCALE"
            log_config.name="\$RUN_NAME"
            log_config.wandb_mode="\$WANDB_MODE"
        )

        if [ -n "\$SEQUENCE_LENGTH" ]; then
            cmd+=(dataloader_train.sequence_length="\$SEQUENCE_LENGTH")
        fi
        if [ -n "\$INPUT_SHAPE" ]; then
            cmd+=(model.input_shape="\$INPUT_SHAPE")
        fi
        if [ -n "\$LR_NET" ]; then
            cmd+=(model.net_optimizer.lr="\$LR_NET")
        fi
        if [ -n "\$LR_FAKE_SCORE" ]; then
            cmd+=(model.fake_score_optimizer.lr="\$LR_FAKE_SCORE")
        fi
        if [ -n "\$LR_DISCRIMINATOR" ]; then
            cmd+=(model.discriminator_optimizer.lr="\$LR_DISCRIMINATOR")
        fi
        if [ "\$DISABLE_WANDB_CALLBACK" = "1" ]; then
            cmd+=(~trainer.callbacks.wandb)
        else
            cmd+=(trainer.callbacks.wandb.sample_logging_iter="\$WANDB_SAMPLE_LOGGING_ITER")
        fi

        echo "__TRAIN_COMMAND__"
        print_command "\${cmd[@]}"
        echo "__LOG_PATH__ \$LOG_PATH"

        if [ "\$DRY_RUN" = "1" ]; then
            exit 0
        fi

        if [ "\$DETACH" = "1" ]; then
            nohup "\${cmd[@]}" > "\$LOG_PATH" 2>&1 &
            echo "__LAUNCHED__ pid=\$! run=\$RUN_NAME log=\$LOG_PATH"
        else
            "\${cmd[@]}" 2>&1 | tee "\$LOG_PATH"
        fi
        ;;

    inference)
        : "\${GPU_ID:?GPU_ID is required for inference}"
        : "\${CONFIG_PATH:?CONFIG_PATH is required for inference}"
        : "\${MODEL_PATH:?MODEL_PATH is required for inference}"
        : "\${CKPT_PATH:?CKPT_PATH is required for inference}"
        : "\${PROMPTS_FILE:?PROMPTS_FILE is required for inference}"
        : "\${IMAGES_FILE:?IMAGES_FILE is required for inference}"

        cd "\$REMOTE_FASTGEN"
        OUT_DIR="\$REMOTE_INFERENCE_ROOT/\$RUN_NAME"
        if [ "\$DRY_RUN" != "1" ]; then
            mkdir -p "\$OUT_DIR"
        fi

        echo "__INFERENCE__ run=\$RUN_NAME out=\$OUT_DIR log=\$LOG_PATH"

        for idx in \$(seq 0 \$((SAMPLE_COUNT - 1))); do
            line_no=\$((idx + 1))
            sample_dir="\$OUT_DIR/sample_\$idx"
            pfile="\$OUT_DIR/prompt_\$idx.txt"
            ifile="\$OUT_DIR/image_\$idx.txt"
            if [ "\$DRY_RUN" != "1" ]; then
                mkdir -p "\$sample_dir"
                sed -n "\${line_no}p" "\$PROMPTS_FILE" > "\$pfile"
                sed -n "\${line_no}p" "\$IMAGES_FILE" > "\$ifile"
            fi

            cmd=(
                env
                CUDA_VISIBLE_DEVICES="\$GPU_ID"
                FASTGEN_OUTPUT_ROOT="\$REMOTE_OUTPUT_ROOT"
                HF_HOME="\$HF_HOME_REMOTE"
                HF_ENDPOINT="\$HF_ENDPOINT_REMOTE"
                PYTHONPATH="\$REMOTE_FASTGEN"
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                TOKENIZERS_PARALLELISM=false
                "\$PYTHON" "\$INFERENCE_PY"
                --ckpt_path "\$CKPT_PATH"
                --do_student_sampling True
                --do_teacher_sampling False
                --save_as_gif False
                --fps "\$FPS"
                --prompt_file "\$pfile"
                --input_image_file "\$ifile"
                --video_save_dir "\$sample_dir"
                --config "\$CONFIG_PATH"
                -
                model.net.model_id_or_local_path="\$MODEL_PATH"
                model.input_shape="\$INPUT_SHAPE"
                dataloader_train.sequence_length="\$SEQUENCE_LENGTH"
                model.student_sample_steps="\$STUDENT_SAMPLE_STEPS"
                model.student_sample_type="\$STUDENT_SAMPLE_TYPE"
                model.sample_t_cfg.t_list="\$SAMPLE_T_LIST"
                model.guidance_scale="\$GUIDANCE_SCALE"
                log_config.name="\${RUN_NAME}_sample_\$idx"
                log_config.wandb_mode="\$WANDB_MODE"
                ~trainer.callbacks.wandb
            )

            echo "__INFERENCE_COMMAND_SAMPLE_\${idx}__"
            print_command "\${cmd[@]}"
            if [ "\$DRY_RUN" != "1" ]; then
                echo "__SAMPLE_\${idx}_START__ \$(date +%F_%T_%Z)" | tee -a "\$LOG_PATH"
                "\${cmd[@]}" >> "\$LOG_PATH" 2>&1
                echo "__SAMPLE_\${idx}_DONE__ \$(date +%F_%T_%Z)" | tee -a "\$LOG_PATH"
            fi
        done
        ;;

    *)
        echo "[run_remote] unknown EXPERIMENT_KIND: \$EXPERIMENT_KIND" >&2
        exit 1
        ;;
esac
REMOTE
