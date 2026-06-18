# Training Report — Video Distillation on Wan2.1-T2V-1.3B

> **Author:** Chen Hing Chin (陈庆展)
> **Date:** 2026-03-22
> **Branch:** `Task3_dev_ChenHingChin`
> **Framework:** NVIDIA FastGen v0.1.0

---

## 1. Infrastructure

| Item | Spec |
|------|------|
| Server | 8× NVIDIA RTX 5090 32GB, 384 cores, 1TB RAM |
| OS | Ubuntu Linux 5.15.0 |
| Python | 3.12.12 (Conda `fastgen`) |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| Framework | NVIDIA FastGen v0.1.0 |
| Base Model | Wan2.1-T2V-1.3B-Diffusers (1.3B DiT) |
| Dataset | OpenVid-1M — 21,133 samples, 22 WebDataset shards |
| Video Spec | 81 frames, 832×480 (480p), latent shape `[16, 21, 60, 104]` |
| Precision | bfloat16 |

---

## 2. Methods Overview

### 2.1 ECT (Enhanced Consistency Training)

- **Paper:** Geng et al., 2024 — *Consistency Models Made Easy*
- **Mechanism:** Student-only consistency loss. Maps any noisy input directly to clean output. No teacher needed (`use_cd=False`).
- **Designed for:** Image generation (CIFAR-10, ImageNet-64). **Not validated for video by FastGen.**
- **Networks in memory:** Student (1.3B) + Text Encoder

### 2.2 CD (Consistency Distillation)

- **Paper:** Song et al., 2023 — *Consistency Models*
- **Mechanism:** Student learns to match teacher's ODE trajectory. Teacher provides target via one-step ODE solve (`use_cd=True`).
- **Designed for:** Image generation. **Not validated for video by FastGen.**
- **Networks in memory:** Student (1.3B) + Teacher (1.3B, frozen) + Text Encoder

### 2.3 DMD2 (Distribution Matching Distillation v2)

- **Paper:** Yin et al., 2024 — *Improved Distribution Matching Distillation for Fast Image Synthesis*
- **Mechanism:** Combines VSD (variational score distillation) + GAN adversarial loss. Alternates student/discriminator updates.
- **Designed for:** Image & Video. **FastGen officially validated on WanT2V (VBench 83.24, 4-step).**
- **Networks in memory:** Student (1.3B) + Teacher (1.3B, frozen) + FakeScore (~1.3B) + Discriminator (~0.5B) + Text Encoder

---

## 3. Experiment Results

### Exp 1: ECT — 1000 iter, Single GPU

| Parameter | Value |
|-----------|-------|
| Config | Custom `config_cm_ct.py` (`use_cd=False`) |
| GPU | 1× RTX 5090 (GPU 3) |
| Optimizer | 8-bit AdamW (bitsandbytes) |
| Batch size | 1 |
| Iterations | 1000 |
| LR | 1e-5 |
| kimg_per_stage | 50 |
| EMA | power (gamma=96.99) |
| Time sampling | logitnormal (p_mean=-0.8, p_std=1.6) |
| Huber const | 0.06 |
| Peak VRAM | ~22 GB |

**Result:** Completely blurry at both 1-step and 4-step inference. No recognizable content.

**Conclusion:** ECT without teacher guidance cannot learn video generation from consistency loss alone. ❌

---

### Exp 2: CD — 1000 iter, Single GPU

| Parameter | Value |
|-----------|-------|
| Config | Custom `config_cm_cd.py` (`use_cd=True`) |
| GPU | 1× RTX 5090 (GPU 5) |
| Optimizer | 8-bit AdamW (bitsandbytes) |
| Batch size | 1 |
| Iterations | 1000 |
| LR | 1e-5 |
| Guidance scale | 5.0 (teacher CFG) |
| kimg_per_stage | 50 |
| Peak VRAM | ~28 GB |

**Result (1-step):** Blurry, unusable.
**Result (4-step):** Rough outlines and motion visible, decent temporal consistency, lacks sharpness.

**Diagnosis:** CTSchedule curriculum only 2% progressed (`kimg=50` too slow for `batch=1`, only 1 image per iteration).

**Conclusion:** CD with teacher guidance has potential at 4-step. Needs more iterations + faster curriculum. ⚠️

---

### Exp 3: CD — 5000 iter, 2-GPU FSDP, Accelerated Curriculum

| Parameter | Value |
|-----------|-------|
| Config | `config_cm_cd.py` + overrides |
| GPU | 2× RTX 5090 (GPU 3+5, FSDP) |
| Batch size (global) | 2 |
| Iterations | 5000 (resumed from 2000) |
| kimg_per_stage | **5** (10× faster than Exp 2) |
| Per-iter time | ~14.8s |
| Total time | ~20h |

**Result (2000 iter):** Outlines visible, slightly better than 1000 iter.
**Result (5000 iter):** Quality **DEGRADED** — more blurry than 2000 iter.

**Root Cause Analysis:**

| Factor | Our Setup | Original Paper |
|--------|-----------|---------------|
| kimg_per_stage | **5** | **3200** |
| Batch size | **2** | **1024** |
| Effective images/stage | 5K | 3.2M |
| Data domain | Video (81 frames, 480p) | Image (CIFAR-10, 32×32) |

The 640× faster curriculum pushes the model into hard consistency constraints (small timestep gaps) before it has learned easier ones (large gaps). With batch=2, gradient estimates are extremely noisy, amplifying this instability. At 5000 iter the curriculum has advanced far enough to destabilize training, while at 2000 iter it was still in the "easy" early stages.

**Conclusion:** Accelerated curriculum causes overfitting/collapse. CD on video remains fundamentally problematic. ⚠️

---

### Exp 4: DMD2 — Single GPU Attempt

| Attempt | Setup | Result |
|---------|-------|--------|
| Vanilla | 1× GPU, batch=1 | OOM: 30.74 GB allocated, need +96 MiB |
| + 8-bit AdamW | 1× GPU, batch=1 | OOM: optimizer saved ~5 GB but not enough |
| + Gradient checkpointing | 1× GPU, batch=1 | OOM: activations saved ~3 GB but 4 networks too large |
| + Teacher CPU offload | 1× GPU, batch=1 | OOM / Training instability from CPU↔GPU transfers |
| 4-GPU FSDP | 4× GPU, batch=8 | OOM on 32GB GPUs with FSDP overhead |

**Conclusion:** DMD2 requires too much memory for single 32GB GPU. Must use multi-GPU FSDP with careful configuration. ❌ (on single GPU)

---

## 4. Summary Table

| Exp | Method | Iter | GPUs | Batch | Steps | Quality | Status |
|-----|--------|------|------|-------|-------|---------|--------|
| 1 | ECT | 1000 | 1 | 1 | 1 & 4 | Blurry, unusable | ❌ |
| 2 | CD | 1000 | 1 | 1 | 4 | Outlines visible | ⚠️ |
| 3 | CD | 5000 | 2 (FSDP) | 2 | 4 | Degraded at 5000 | ⚠️ |
| 4 | DMD2 | — | 1 | 1 | — | OOM | ❌ |

**Key Finding:** ECT and CD are **image-only methods** not validated for video by FastGen. NVIDIA validated 6 video methods for WanT2V: DMD2, f-distill, LADD, MeanFlow, CausVid, Self-Forcing. Our experiments confirm ECT/CD cannot produce usable video output.

---

## 5. DMD2 Two-GPU FSDP Training Analysis

### 5.1 Why 2-GPU FSDP Can Work

FSDP (Fully Sharded Data Parallel) shards model parameters, gradients, and optimizer states across GPUs. Each GPU only holds 1/N of the full state, plus the currently-active layer's unsharded parameters during forward/backward.

**Memory Model (2-GPU FSDP, bf16):**

| Component | Per-GPU (no FSDP) | Per-GPU (2-GPU FSDP) |
|-----------|-------------------|---------------------|
| Student DiT (1.3B, trainable) | 2.6 GB params + 2.6 GB grads | ~1.3 GB + ~1.3 GB |
| Teacher DiT (1.3B, frozen) | 2.6 GB | ~1.3 GB |
| FakeScore (~1.3B, trainable) | 2.6 GB + 2.6 GB | ~1.3 GB + ~1.3 GB |
| Discriminator (~0.5B, trainable) | 1.0 GB + 1.0 GB | ~0.5 GB + ~0.5 GB |
| Text Encoder (UMT5, frozen) | ~10 GB | ~10 GB (not sharded, shared) |
| Adam optimizer states (fp32) | ~25 GB | ~12.5 GB |
| Activations (batch=1) | ~3-5 GB | ~3-5 GB |
| FSDP communication buffers | — | ~1-2 GB |
| **Total** | **~53-60 GB** | **~33-37 GB** |

**33-37 GB is still over 32 GB.** Additional optimizations are required:

### 5.2 Required Optimizations for 2-GPU

| Optimization | Saves | Priority |
|-------------|-------|----------|
| **8-bit AdamW** | ~6 GB/GPU (optimizer halved) | Must-have |
| **FSDP CPU offload** | ~3-5 GB/GPU | Likely needed |
| **Gradient checkpointing** | ~2-3 GB/GPU (activations) | Recommended |

**Estimated memory with all optimizations:**

| Component | Per-GPU |
|-----------|---------|
| Sharded params + grads | ~6.2 GB |
| Text Encoder | ~10 GB |
| 8-bit optimizer states | ~6.3 GB |
| Activations (with grad ckpt) | ~1-2 GB |
| FSDP buffers + unsharded active layer | ~3-4 GB |
| **Total** | **~27-29 GB** ✅ |

### 5.3 FastGen DMD2 Reference Configuration

From FastGen's official `config_dmd2.py` for WanT2V:

| Parameter | FastGen Official | Our Proposed 2-GPU |
|-----------|-----------------|-------------------|
| GPUs | 8 (H100 80GB) | 2 (RTX 5090 32GB) |
| FSDP | Yes | Yes |
| Batch size (global) | 64 | **2** |
| Batch size (per GPU) | 1 | 1 |
| Grad accumulation | 8 | 1 |
| Max iterations | 6000 | 1000-2000 |
| LR (all optimizers) | 1e-5 | 1e-5 |
| Student sample steps | 4 | 4 |
| Guidance scale | 5.0 | 5.0 |
| GAN loss weight | 0.03 | 0.03 |
| Discriminator type | multiscale_down_mlp_large | multiscale_down_mlp_large |
| Feature extraction | layers [15, 22, 29] | layers [15, 22, 29] |
| Timestep schedule | [0.999, 0.937, 0.833, 0.624, 0.0] | [0.999, 0.937, 0.833, 0.624, 0.0] |
| Checkpoint interval | 500 | 200 |
| VBench (official) | **83.24** (4-step) | TBD |

### 5.4 Key Differences and Risks

#### Batch Size: 2 vs 64

This is the **most critical difference**. DMD2 relies on adversarial training (GAN loss), where the discriminator must see a diverse batch to learn meaningful real/fake boundaries.

| Aspect | batch=64 | batch=2 |
|--------|----------|---------|
| GAN gradient quality | Stable, diverse | Noisy, high variance |
| Distribution matching | Well-estimated | Poorly estimated |
| Training stability | Good | Risk of mode collapse |

**Mitigation strategies:**
1. **Gradient accumulation**: Set `batch_size_global=8` with 2 GPUs → `grad_accum=4`. Each discriminator update sees 8 samples. This is the **minimum recommended** for adversarial training.
2. **Reduce GAN loss weight**: If training is unstable, lower `gan_loss_weight` from 0.03 to 0.01-0.02 to rely more on VSD loss.
3. **More iterations**: Compensate for fewer samples/step with more training iterations (2000-3000 instead of 1000).

#### VRAM with Gradient Accumulation

With `batch_size_global=8` and 2 GPUs, gradient accumulation = 4. This does **not** increase peak VRAM (only 1 sample in memory at a time), but each "iteration" takes 4× longer:

| Setup | Per-iter time (est.) | 1000 iter wall time |
|-------|---------------------|-------------------|
| batch_global=2, no accum | ~30-50s | 8-14h |
| batch_global=8, accum=4 | ~120-200s | 33-56h |
| batch_global=4, accum=2 | ~60-100s | 17-28h |

**Recommended:** Start with `batch_global=4` (accum=2) as a compromise between quality and training time.

### 5.5 Proposed 2-GPU DMD2 Training Script

```bash
torchrun --nproc_per_node=2 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.fsdp_cpu_offload=True \
      trainer.batch_size_global=4 \
      trainer.max_iter=2000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=200 \
      trainer.validation_iter=500 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=dmd2_2gpu_fsdp
```

**Additional requirements:**
- 8-bit AdamW: Via `train_lowmem.py` wrapper (monkey-patch `torch.optim.AdamW`)
- Gradient checkpointing: Check if FastGen's DMD2 method supports `gradient_checkpointing=True` in config, or requires code-level patching
- FSDP CPU offload: `trainer.fsdp_cpu_offload=True` (built into FastGen's BaseTrainerConfig)

### 5.6 Verification Plan

1. **Smoke test (10 iter):** Verify no OOM, check peak VRAM with `nvidia-smi`
2. **Loss check (100 iter):** Verify both VSD loss and GAN loss are decreasing, not NaN
3. **Checkpoint (200 iter):** First checkpoint, run 4-step inference to check early quality
4. **Full run (2000 iter):** Compare against CD 2000 iter and Teacher baseline

---

## 6. Comparison with FastGen Official Results

| Method | Source | Steps | VBench | GPUs | Iter |
|--------|--------|-------|--------|------|------|
| Teacher (Wan2.1-1.3B) | Baseline | 50 | ~85 | 1 | — |
| DMD2 | FastGen official | 4 | **83.24** | 8×H100 | 6000 |
| rCM | FastGen official | 4 | **84.43** | 8×H100 | — |
| CausVid (pretrained) | Our Phase 0 | 3 | — | 1 | — |
| ECT | Our Exp 1 | 4 | Unusable | 1 | 1000 |
| CD | Our Exp 2 | 4 | Poor | 1-2 | 1000-5000 |
| DMD2 | **Our Exp 5 (planned)** | 4 | **TBD** | 2 | 2000 |

---

## 7. Conclusions

1. **ECT and CD are not viable for video distillation.** They were designed for image generation (CIFAR-10, ImageNet) and FastGen intentionally excluded them from WanT2V configs. Our experiments confirm: ECT produces blurry output, CD shows minimal structure but degrades with more training.

2. **DMD2 is the primary path forward.** It is FastGen's official and validated method for WanT2V video distillation (VBench 83.24 at 4-step). However, it requires multi-GPU FSDP due to 4-network architecture.

3. **2-GPU FSDP DMD2 is feasible** with 8-bit AdamW + FSDP CPU offload + gradient checkpointing. Estimated VRAM: ~27-29 GB per GPU. The main risk is low batch size affecting adversarial training stability — mitigate with gradient accumulation (`batch_global=4`).

4. **Next step:** Deploy and run 2-GPU FSDP DMD2 with the proposed configuration. Target: 2000 iterations with checkpoints every 200 iter for quality monitoring.
