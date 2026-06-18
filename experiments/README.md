# Experiments

This directory is the default interface for fast research iteration.

## Workflow

1. Pick or create one config in `experiments/configs/`.
2. Check server state.
3. Dry-run the command.
4. Launch only after the command and GPU choice are clear.
5. Record the result under `experiments/results/`.

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

## Config Types

| Kind | Required fields |
|---|---|
| `train` | `EXPERIMENT_ID`, `RUN_NAME`, `CUDA_DEVICES`, `CONFIG_PATH`, `MODEL_PATH`, `DATA_TAG`, `MAX_ITER` |
| `inference` | `EXPERIMENT_ID`, `RUN_NAME`, `GPU_ID`, `CKPT_PATH`, `PROMPTS_FILE`, `IMAGES_FILE`, `SAMPLE_COUNT` |

## Current Configs

| Config | Purpose |
|---|---|
| `wan22_dmd2_no_cfg_stage1.env` | Validated no-CFG Wan2.2 DMD2 stage route |
| `wan22_dmd2_65f_cfg5_formal.env` | 65-frame CFG=5 formal training template |
| `wan22_dmd2_0013000_infer_10distinct.env` | 10-sample inference template from checkpoint `0013000` |

## Result Notes

Use `experiments/results/README.md` as the template. A result note should be short enough to scan and complete enough to resume:

- date and run name
- config path
- remote log path
- output/checkpoint path
- status
- key metrics or failure signature
- next action

