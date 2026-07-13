# FastGen Helper Scripts

These scripts are lower-level helpers for setup, data preparation, and direct FastGen debugging.

For normal experiments, start from:

```bash
bash experiments/bin/check_remote.sh experiments/configs/wan22_dmd2_no_cfg_stage1.env
bash experiments/bin/run_remote.sh --dry-run experiments/configs/wan22_dmd2_no_cfg_stage1.env
```

## Main Helpers

| Script | Use |
|---|---|
| `download_model.sh` | Download Wan2.1 model weights through the server-side downloader |
| `download_openvid.sh` | Download OpenVid data |
| `prepare_training_data.sh` | Build mp4+txt WebDataset shards |
| `convert_to_webdataset.py` | Convert CSV/video pairs to WebDataset |
| `convert_parquet_to_wds.py` | Convert latent parquet sources to WebDataset |
| `run_inference.sh` | Legacy direct inference wrapper |
| `run_wan22_ti2v5b_dmd2.sh` | Direct Wan2.2 DMD2 launcher skeleton |
| `train_lowmem.py` | Low-memory training wrapper |

## Legacy Method Scripts

These are preserved for method-level reproduction or debugging:

- `run_dmd2_single_gpu.sh`
- `run_ect_single_gpu.sh`
- `run_cd_single_gpu.sh`
- `run_fdistill_single_gpu.sh`
- `run_ladd_single_gpu.sh`
- `run_meanflow_single_gpu.sh`

They are not the preferred way to launch new research runs. Convert new variants into `experiments/configs/*.env` so run names, logs, GPU choices, and output paths remain consistent.

## Historical Script Tree

`archive/ip-2026-spring/` remains here as an imported historical script tree. Treat it as reference material, not an active entrypoint.

