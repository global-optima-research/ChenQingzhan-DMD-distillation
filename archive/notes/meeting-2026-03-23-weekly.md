# Task 3 周报 — 2026-03-17 ~ 2026-03-23

**作者:** 陈庆展 (Chen Hing Chin)
**分支:** `Task3_dev_ChenHingChin`
**日期:** 2026-03-23

---

## 本周工作摘要

### 1. ECT & CD 训练实验 (03-17 ~ 03-19)

- 完成 ECT 1000 iter 训练 (单卡, 8-bit AdamW)
  - 结果: 1-step 和 4-step 推理均完全模糊
- 完成 CD 1000 iter 训练 (单卡, 8-bit AdamW)
  - 结果: 4-step 有粗略轮廓, 但清晰度不足
- CD 扩展训练至 5000 iter (2卡 FSDP, 加速 curriculum)
  - 结果: 5000 iter 质量反而下降 (curriculum kimg=5 过快导致训练不稳定)
- **结论:** ECT 和 CD 是图像方法, FastGen 未对视频验证, 实验确认不适用于视频生成

### 2. DMD2 显存优化探索 (03-22)

- 单卡训练 OOM (30.74 GB, 32GB GPU)
- 2卡 FSDP + 8-bit AdamW: 失败 (bitsandbytes 与 FSDP2 DTensor 不兼容)
- 2卡 FSDP + CPU Offload: 可运行但 ~1200s/iter, 不实际
- **最终方案: 4卡 FSDP + 标准 AdamW → 22-27 GB/卡, 16s/iter**

### 3. DMD2 正式训练 (03-22 ~ 03-23)

- 配置: 4x RTX 5090 FSDP, batch=4, 2000 iter
- 训练时间: ~9 小时 (20:47 ~ 05:49)
- Loss: 1.38 → ~1.10 (稳定收敛)
- 峰值显存: 22-27 GB/卡
- Checkpoint: 每 200 iter 保存一次 (共 10 个)

### 4. DMD2 推理评估 (03-23)

- 合并 5 个 FSDP checkpoint (400, 800, 1200, 1600, 2000 iter)
- 4-step 推理, guidance_scale=5.0
- 生成视频已下载到本地
- 实验报告 (含视频对比) 已生成

---

## 本周产出

| # | 产出 | 说明 |
|---|------|------|
| 1 | `Training_Report.md` | 完整训练报告 (所有实验环境/参数/结果) |
| 2 | `DMD2_Training_Report.html` | DMD2 实验报告 (含嵌入视频对比) |
| 3 | `scripts/training/run_dmd2_fsdp.sh` | DMD2 4卡 FSDP 训练脚本 |
| 4 | `scripts/inference/run_dmd2_inference_all.sh` | 批量推理脚本 |
| 5 | `results/comparison/dmd2_4step/` | 5 个时间点的推理视频 |

---

## 关键发现

### 1. ECT/CD 不适用于视频蒸馏

- FastGen 官方验证了 6 种视频方法 (DMD2, f-distill, LADD, MeanFlow, CausVid, Self-Forcing)
- ECT/CD 被有意排除, 我们的实验证实了这一点

### 2. DMD2 需要至少 4 卡 FSDP

- 4 个网络 (Student + Teacher + FakeScore + Discriminator) 导致显存需求大
- 8-bit AdamW 与 FSDP2 不兼容
- CPU Offload 可行但速度不实际 (~1200s/iter vs 16s/iter)

### 3. DMD2 训练收敛正常

- Loss 在前 200 iter 快速下降, 之后稳定在 1.04-1.17
- 视频质量需要视觉评估和 VBench 定量评估

---

## 下周计划

1. **DMD2 质量评估**
   - 视觉评估 2000 iter checkpoint 的视频质量
   - 如果质量不足, 延长训练至 6000 iter (FastGen 官方配置)

2. **VBench 评估** (如果时间允许)
   - 配置 VBench 评估管线
   - 对比 DMD2 训练结果与 FastGen 官方基准 (83.24)

3. **Phase 1 准备**
   - 调研 Progressive Distillation (50→16→8→4 步)
   - 等待 Task 2 Teacher Model 交付 (Week 12)

---

## 风险与问题

- DMD2 2000 iter 是否足够? FastGen 官方使用 6000 iter + batch_global=64
- 我们的 batch=4 远小于官方 64, 可能影响 GAN 训练稳定性
- 视频质量待评估
