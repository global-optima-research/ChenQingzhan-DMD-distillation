# Tools

Server and data utilities salvaged from the early project phase (2026-03/04). They built the
environment and the OpenVid WebDataset that the current WanT2V line still trains on. These are
one-time / setup helpers, not run entry points — runs go through `experiments/bin/`.

| Script | Use |
|---|---|
| `setup_server.sh` | One-time server conda/env setup for FastGen |
| `download_model.sh` | Download Wan2.1 model weights on the server |
| `download_openvid.sh` | Download OpenVid data |
| `prepare_training_data.sh` | Build mp4+txt WebDataset shards |
| `convert_to_webdataset.py` | Convert CSV/video pairs to WebDataset |
| `convert_parquet_to_wds.py` | Convert latent parquet sources to WebDataset |
| `hf-download.py` | HF-mirror model/dataset downloader (local mirror of the server copy at `/data/chenqingzhan/hf-download.py`, which `download_model.sh` calls) |

Legacy method launchers and the historical script tree stayed with the archived Wan2.2 line at
`archive/wan22-ti2v-line/scripts/`.
