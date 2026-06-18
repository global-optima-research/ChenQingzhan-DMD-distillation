# Experiment Log — Video Distillation Method Comparison

> **Author:** Chen Hing Chin (陈庆展)
> **Updated:** 2026-03-30
> **Hardware:** 8× RTX 5090 32GB, NVIDIA FastGen v0.1.0, Wan2.1-T2V-1.3B
> **Data:** OpenVid-1M (21,133 samples, WebDataset)

---

## Summary

| Exp | Method | Iter | GPUs | Steps | VRAM/GPU | Speed | Loss | Status |
|-----|--------|------|------|-------|----------|-------|------|--------|
| 1 | ECT | 1000 | 1 | 1 & 4 | — | — | — | ❌ Blurry |
| 2 | CD | 1000 | 1 | 4 | — | — | — | ❌ Poor |
| 3 | CD | 5000 | 2 | 4 | — | 14.8s | — | ❌ Degraded |
| 4 | DMD2 | 2000 | 4 | 4 | 22-27 GB | 16s | 1.38→1.10 ✅ | ✅ Converged |
| 5 | MeanFlow | 6000 | 2 | 1 & 4 | 26-29 GB | 29.6s | Not converged | ⚠️ Undertrained |
| 6 | f-distill | 4000 | 4 | 1 & 4 | 24-28 GB | 20s | 1.49→0.19 ✅ | ✅ Done |
| 7 | LADD | 4000 | 4 | 4 | 29.7 GB | 24s | 1.40→... | 🔄 Running |
| 8 | CausVid | — | 4 | — | 30.4 GB | — | — | ❌ OOM |
| 9 | Self-Forcing | — | 4 | — | 30.4 GB | — | — | ❌ OOM |

**Conclusions:**
- DMD2 and f-distill converge fast under small compute budgets (GAN-based)
- MeanFlow needs far more iterations (official 1M, we ran 0.6%)
- CausVid / Self-Forcing OOM on 4×32GB due to CausalWan attention overhead (~3 GB extra)
- Text Encoder (UMT5-XXL, ~10 GB replicated per GPU) is the main VRAM bottleneck

---

## Phase 0.5: ECT / CD (Failed)

**Exp 1-3:** ECT and CD (Consistency Model family) are not officially supported for video in FastGen. ECT produced unusable outputs; CD degraded with more training due to curriculum mismatch (batch 512× smaller than paper).

Videos: `results/phase05/ect_*`, `results/phase05/cd_*`

---

## Phase 1: Full Method Comparison

### Exp 4: DMD2 — 2000 iter, 4-GPU FSDP ✅

- **Config:** `config_dmd2.py` | Networks: S+T+FS+D (4)
- **Training:** 2000 iter, batch=4, ~16s/iter, ~9h
- **Loss:** total_loss 1.38 → 1.10 (stable convergence)
- **Inference:** 4-step, 6.6s/video
- **Videos:** `results/phase05/dmd2_2000iter_4step/`

### Exp 5: MeanFlow — 6000 iter, 2-GPU FSDP ⚠️

- **Config:** `config_mf_video.py` (fix: `enable_preprocessors=True`) | Networks: S+T (2) + EMA
- **Training:** 6000 iter, batch=2, ~29.6s/iter, ~49h
- **Loss:** mf_loss fluctuated 0.005-0.044 after 2000-step warmup, no downtrend. v_loss flat ~0.13-0.18
- **Inference:** 1-step 2.1s, 4-step 6.1s — quality mediocre
- **Analysis:** Official trains 1M iter; our 6000 is 0.6%. MeanFlow needs much more compute than GAN methods
- **Videos:** `results/phase1/meanflow_6000iter/`

### Exp 6: f-distill — 4000 iter, 4-GPU FSDP ✅

- **Config:** `config_fdistill.py` | Networks: S+T+FS+D (4)
- **Training:** 4000 iter (1000 first run + 3000 resumed from ckpt), batch=4, ~20s/iter
- **Note:** First run crashed at iter ~1950 due to NCCL timeout (WandB video encoding blocked >10min). Resumed with `validation_iter=99999` to disable
- **Loss:** total_loss 1.49 → 0.19 (converged)
- **Inference:** 1-step 1.76s, 4-step 6.6s
- **Videos:** `results/phase1/fdistill_2000iter/`
- **Checkpoints:** iter 1000, 2000, 3000, 4000

### Exp 7: LADD — 4000 iter, 4-GPU FSDP 🔄

- **Config:** `config_ladd.py` (fix: must add `dataloader_train.batch_size=1`) | Networks: S+T+D (3)
- **Training:** Started 2026-03-29 21:45, currently at ~iter 600
- **VRAM:** 29.7 GB/GPU (tightest of all working methods)
- **Note:** Failed on 3-GPU (30.5 GB OOM). Original config missing `batch_size=1` caused earlier 4-GPU OOM
- **ETA:** ~2026-03-31 00:30

### Exp 8-9: CausVid & Self-Forcing ❌ OOM

Both use CausalWan student (causal attention) which requires ~3 GB extra activations. Even without EMA and with `batch_size=1`, 4-GPU FSDP peaks at 30.4 GB/GPU → OOM on RTX 5090 32GB. **Not feasible on current hardware.**

---

## VRAM Requirements (RTX 5090 32GB, measured)

| Method | 2-GPU | 3-GPU | 4-GPU |
|--------|-------|-------|-------|
| MeanFlow | ✅ 26-29 GB | ✅ | ✅ |
| DMD2 | ❌ 30.4 GB | ❌ | ✅ 22-27 GB |
| f-distill | ❌ | ❌ | ✅ 24-28 GB |
| LADD | ❌ | ❌ 30.5 GB | ✅ 29.7 GB |
| CausVid | ❌ | ❌ | ❌ 30.4 GB |
| Self-Forcing | ❌ | ❌ | ❌ 30.4 GB |

---

## Standardized Experiment Configs

All experiments now use per-experiment config files (see `FastGen_Guide.md` §4):

```
fastgen/configs/experiments/WanT2V/our/
├── _common.py              # Shared constants (model path, data path)
├── exp01_dmd2_4gpu.py
├── exp02_meanflow_2gpu.py
├── exp03_fdistill_4gpu.py
└── exp04_ladd_4gpu.py
```

Launch: `bash run_exp.sh <config_path> <num_gpus> [gpu_ids]`

---

## Results Directory

```
results/
├── phase0/          # Pretrained inference (teacher, causvid, rcm, self_forcing)
├── phase05/         # Phase 0.5 experiments (ect, cd, dmd2)
├── phase1/          # Phase 1 comparison (meanflow, fdistill, ladd, ...)
└── frames/          # Screenshot frames for reports
```
