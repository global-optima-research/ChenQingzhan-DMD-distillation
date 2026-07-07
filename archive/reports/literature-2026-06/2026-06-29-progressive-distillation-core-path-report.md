# 渐进式蒸馏路线报告：从图像 Progressive Distillation 到视频扩散加速

日期：2026-06-29
项目背景：Wan2.1-T2V-1.3B 的 DMD2 蒸馏加速
当前实验路线：`50-step teacher -> 8-step student -> 4-step student`

## 1. 前言

我们当前工作使用渐进式蒸馏路线加速 Wan2.1-T2V 视频生成模型：先从 50-step teacher 蒸馏到 8-step student，再从较好的 8-step checkpoint 蒸馏到 4-step student。这个路线需要文献支撑，尤其需要解释为什么不是直接做 `50 -> 4`，以及为什么中间 8-step 阶段有价值。

从文献脉络看，最早直接支撑渐进式蒸馏思想的是 Salimans 和 Ho 的 **Progressive Distillation for Fast Sampling of Diffusion Models**。但这篇工作主要在图像扩散模型上验证，对我们的视频任务只有方法论参考。真正与我们项目更接近的是 ByteDance 的两项 Lightning 工作：**SDXL-Lightning** 和 **AnimateDiff-Lightning**。前者在大规模图像扩散模型上证明 progressive adversarial distillation 的价值，后者进一步将这一路线迁移到视频扩散模型，和我们的 Wan T2V 蒸馏加速具有明确对应关系。

本文按照这条技术路径展开：

```text
Progressive Distillation 图像扩散基础思想
    -> SDXL-Lightning 渐进式 + 对抗蒸馏
    -> AnimateDiff-Lightning 视频扩散渐进式对抗蒸馏
    -> 我们的 Wan2.1-T2V DMD2 50 -> 8 -> 4
```

## 2. 相关工作脉络

### 2.1 Progressive Distillation：渐进式蒸馏思想起点

Progressive Distillation for Fast Sampling of Diffusion Models 是本路线最早、最直接的方法论来源。它提出把一个多步 diffusion sampler 逐阶段蒸馏成更少步的 sampler。核心思想是：student 不直接学习最终数据分布，而是学习用更少的采样步数近似 teacher 多步采样后的结果。

典型流程是：

```text
N-step teacher -> N/2-step student -> N/4-step student -> ... -> few-step student
```

该工作证明了 diffusion sampler 的采样步数可以通过逐阶段蒸馏大幅减少，而不是只能依赖更快的 ODE solver。

但它对我们的参考价值有边界：

- 它主要验证在图像生成任务上；
- 使用的是 DDIM / deterministic sampler 轨迹蒸馏；
- 重点关注图像 FID 和视觉质量；
- 没有处理视频生成中的时序一致性、物理规则、运动稳定性等问题；
- 没有直接讨论 DMD2 这类 distribution matching objective。

因此，它对我们的价值主要是“证明渐进式压缩采样步数是合理的”，而不是直接给出 Wan T2V 的训练 recipe。

### 2.2 SDXL-Lightning：确认渐进式蒸馏的核心价值，并加入对抗目标

SDXL-Lightning 是 ByteDance 在 2024 年 2 月发布的工作，完整题目是 **SDXL-Lightning: Progressive Adversarial Diffusion Distillation**。它的重要性在于：它不是单纯复用 Progressive Distillation，而是把 **progressive distillation** 和 **adversarial distillation** 结合起来，用于 SDXL 这类大规模高分辨率图像扩散模型。

它给出一个关键判断：

> Progressive distillation 可以保持原模型 probability flow 和 mode coverage，但仅用 MSE 目标在 8-step 以下容易产生模糊；adversarial loss 可以提升少步生成的清晰度和细节。

这比原始 Progressive Distillation 更贴近我们当前 DMD2 训练，因为我们的 DMD2 里也包含 distribution matching 和 GAN/discriminator 约束，不是简单 MSE 回归。

SDXL-Lightning 的训练路径是：

```text
128 -> 32: MSE distillation
32 -> 8 -> 4 -> 2 -> 1: adversarial distillation
```

这条路线说明两点：

1. 大步数到中等步数阶段，可以先用相对稳定的蒸馏目标；
2. 进入 8-step、4-step、2-step、1-step 等少步区域后，仅靠 MSE 不够，需要对抗性目标来保证清晰度和细节。

这对我们当前 `50 -> 8 -> 4` 具有直接参考价值：8-step 是中间阶段，4-step 是低步部署目标，而低步阶段必须重视判别器、GAN loss、有效 batch 和训练稳定性。

### 2.3 AnimateDiff-Lightning：将渐进式对抗蒸馏迁移到视频扩散模型

AnimateDiff-Lightning 是 ByteDance 在 2024 年 3 月发布的工作，完整题目是 **AnimateDiff-Lightning: Cross-Model Diffusion Distillation**。它的重要性在于：它明确把 SDXL-Lightning 的 progressive adversarial distillation 迁移到视频扩散生成任务上。

这是我们最应重点参考的工作，因为它处理的是视频生成，而不仅是图像生成。

AnimateDiff-Lightning 的核心贡献包括：

- 使用 progressive adversarial diffusion distillation 加速 AnimateDiff；
- 提供 1-step、2-step、4-step、8-step video checkpoint；
- 认为 4-step 在视频质量和速度之间更平衡；
- 使用 cross-model distillation，让一个 motion module 同时适配多个 base model；
- 设计 flow-conditional video discriminator，用于判别 video transition 是否符合 teacher flow；
- 使用大规模训练和 gradient accumulation 稳定少步视频蒸馏。

它的实验和 model card 都说明，1-step 更实验性，2-step 容易有 flicker，4-step 是更实用的速度-质量折中。这和我们当前把 4-step 作为主要目标非常一致。

## 3. 重点论文具体内容

### 3.1 Progressive Distillation for Fast Sampling of Diffusion Models

这篇论文的核心机制是“teacher 两步，student 一步”。假设当前 teacher 是 `N-step` sampler，目标是训练 `N/2-step` student。训练时，从某个 noisy state 出发，teacher 先走两小步，得到目标位置；student 被训练成从同一起点直接走一大步到达类似位置。

形式上可以理解为：

```text
Teacher: z_t -> z_{t-1/(2N)} -> z_{t-1/N}
Student: z_t -------------> z_{t-1/N}
```

训练完成后，student 变成下一阶段 teacher，再继续减少步数。

这篇论文给我们的核心经验是：

- 少步采样最好逐阶段训练，而不是一次性大幅压缩；
- 中间 student 的质量决定下一阶段上限；
- 每个阶段都需要训练到足够稳定；
- 低步数模型对参数化、loss weighting 和 timestep 设计非常敏感。

但由于它是图像领域工作，对我们视频任务的直接帮助有限。它没有解决视频扩散里的 temporal consistency、motion coherence 和物理规则问题。

### 3.2 SDXL-Lightning

SDXL-Lightning 继承了 Progressive Distillation 的 step-wise 压缩思想，但指出一个关键问题：少步 student 容量不足以精确匹配 teacher 的复杂 trajectory。如果只用 MSE 匹配 teacher output，student 容易学到平均化结果，导致模糊。

因此它提出 progressive adversarial diffusion distillation：

```text
progressive distillation: 保持 teacher probability flow
adversarial loss: 改善少步清晰度和细节
```

其 discriminator 不是只判断最终图像真假，而是判断从当前 flow location 出发，student 预测的下一位置是否像 teacher 产生的下一位置。也就是说，它更关注 transition 是否符合 teacher flow。

这个设计对我们很有启发，因为我们当前的 DMD2 也有 discriminator/GAN loss，但需要进一步确认它是否足够：

- timestep-aware；
- flow-aware；
- transition-aware；
- video-aware。

SDXL-Lightning 还提供一个重要实践经验：每个 step 数对应独立 checkpoint，例如 1-step、2-step、4-step、8-step，而不是一个模型随便切换 step。我们当前分别训练 8-step 和 4-step student，并对每个 checkpoint 做 sweep，是与这个经验一致的。

### 3.3 AnimateDiff-Lightning

AnimateDiff-Lightning 把 SDXL-Lightning 的方法迁移到视频。它的贡献说明：渐进式对抗蒸馏不仅适用于图像扩散，也可以用于视频扩散。

它的关键修改包括：

1. **只训练 motion module。** AnimateDiff 由 frozen image base model 和 motion module 组成，AnimateDiff-Lightning 主要蒸馏 motion module。
2. **Cross-model distillation。** 多个 GPU rank 加载不同 base model，共享并更新同一个 motion module，让少步 motion module 兼容更多风格和模型。
3. **Flow-conditional video discriminator。** 判别器不仅看视频结果，还带有 flow/base-model condition，用 3D convolution 处理时空特征。
4. **大 batch / gradient accumulation。** 由于视频显存占用大，每卡 batch 只能是 1，但通过 64 张 A100 和 gradient accumulation 达到 total batch size 256。
5. **4-step 作为实用折中。** 论文和 model card 都表明 4-step 在速度和质量之间比较平衡，1-step/2-step 更容易不稳定。

这对我们非常直接。我们当前 Wan T2V 不是 motion module 架构，但视频蒸馏的核心困难类似：

- 少步视频容易模糊；
- 少步视频容易出现 motion inconsistency；
- 物理规则和 temporal consistency 比图像任务更难；
- batch 和稳定训练非常关键；
- discriminator 最好能看视频时空结构，而不是只看单帧质量。

## 4. 对我们工作的参考价值

### 4.1 我们的路线可以更明确地表述

我们当前路线应表述为：

> 基于 Progressive Distillation 和 Lightning 系列工作的经验，我们采用 progressive distribution-matching / adversarial distillation 路线，将 Wan2.1-T2V 的 50-step teacher 先蒸馏为 8-step intermediate student，再从最佳 8-step checkpoint 初始化 4-step student。

这比单纯说“训练一个 8-step 和 4-step 模型”更有方法论依据。

### 4.2 8-step intermediate student 是必要桥梁

从 Progressive Distillation 和 SDXL-Lightning 看，中间阶段的价值是降低 teacher-student gap。直接 `50 -> 4` 可能让 student 一次承担过大的 denoising transition，导致模糊和结构错误。

因此，8-step student 的意义不是最终部署，而是为 4-step 提供更容易学习的初始化和 teacher-like trajectory。

如果后续仍不稳定，可以考虑更平滑路线：

```text
50 -> 16 -> 8 -> 4
```

### 4.3 少步质量不能只靠 MSE 或简单匹配

SDXL-Lightning 明确指出 MSE progressive distillation 在少步下会模糊，所以引入 adversarial loss。我们的 DMD2 有 distribution matching 和 GAN/discriminator 约束，这是合理方向。

但下一步需要检查：

- 当前 discriminator 是否显式输入 timestep；
- 是否能判断 teacher/student transition，而不只是最终 fake/real；
- 是否有足够 temporal receptive field；
- 是否需要 video-aware 或 motion-aware discriminator。

这可能是解决物理规则和 temporal consistency 的关键。

### 4.4 Batch 和训练稳定性是核心变量

Lightning 系列使用非常大的 global batch：

- SDXL-Lightning：batch 512；
- AnimateDiff-Lightning：64 A100，每卡 batch 1，gradient accumulation 4，总 batch 256。

这和我们的观察高度一致：

- 低 LR / 小有效 batch 的 8-step student 很差；
- 恢复默认 LR 并提升 batch 后，8-step 进入可用阶段；
- 4-step from 8 在 8node / batch-improved 设置下质量可用，并且 `0000500-0002500` 呈提升趋势。

因此，我们后续报告必须记录 effective batch，而不只是 iter：

- GPU 数；
- per-GPU batch；
- gradient accumulation；
- `batch_size_global`；
- `student_update_freq`；
- 有效 student update 次数。

### 4.5 每个 step 数需要独立 checkpoint 和独立评估

SDXL-Lightning 和 AnimateDiff-Lightning 都提供 1/2/4/8-step 独立 checkpoint。这说明低步模型不是同一个模型随便换 `num_inference_steps`，而是每个 step 设置都需要专门训练。

这支持我们当前做法：

- 8-step student 单独训练；
- 4-step from 8 单独训练；
- 每 500 iter 保存 checkpoint；
- 每个 checkpoint 做同一组 prompt 推理；
- 用 HTML 同步对比 teacher / student / checkpoint。

### 4.6 4-step 是当前最合理目标

AnimateDiff-Lightning 明确说明 2/4/8-step 质量较好，1-step 更实验性；论文也观察到 2-step 可能 flicker，4-step 是较好的质量速度平衡。

因此，我们目前不应该急着追 1-step 或 2-step。更稳妥的路线是：

```text
先把 4-step 做稳定
再考虑 2-step
最后才讨论 1-step
```

## 5. 对我们已有实验的解释

| 我们的实验现象 | 文献解释 |
|---|---|
| 早期低 LR / 小 batch 8-step 模糊 | 少步蒸馏对训练稳定性敏感；student transition 没学好时，多步也会累积误差。 |
| 8-step 默认 LR + batch 12 后质量变好 | 有效学习率和 batch 提升后，intermediate student 更接近可用 teacher-like model。 |
| 第一轮 4-step from 8 后期物理崩坏 | 低步蒸馏可能在 flow preservation、sharpness 和 mode coverage 之间失衡；最后 checkpoint 不一定最好。 |
| 8node / batch-improved 后 4-step from 8 质量提升 | 增大有效训练规模提高了 adversarial / distribution matching 稳定性，说明路线本身可行。 |
| 4-step 当前最适合作为部署目标 | AnimateDiff-Lightning 同样将 4-step 视为视频质量和速度的较好折中。 |

## 6. 后续实验建议

### 6.1 短期建议

1. 继续围绕 `8-step lr_original 0002500 -> 4-step` 的 batch-improved 路线做复现。
2. 把 `batch_size_global=16` 或更稳定的 effective batch 作为当前主线配置。
3. 继续用 10 prompt eval 做 checkpoint sweep，但要明确记录物理规则、时序一致性和清晰度。
4. 不默认选择最后 checkpoint，保持肉眼评估和同步 HTML 对比。

### 6.2 中期建议

1. 检查 FastGen DMD2 discriminator 的输入设计，确认它是否 timestep-aware、flow-aware、video-aware。
2. 尝试增加 temporal discriminator 或 motion-aware discriminator。
3. 对 discriminator 做 ablation：原始 discriminator vs flow-conditioned discriminator vs temporal discriminator。
4. 如果 `50 -> 8` 仍不稳定，尝试 `50 -> 16 -> 8 -> 4`。

### 6.3 报告建议

最终汇报可以把文献支撑写成三层：

```text
第一层：Progressive Distillation 证明逐阶段减少采样步数可行。
第二层：SDXL-Lightning 证明 progressive + adversarial 更适合少步高质量生成。
第三层：AnimateDiff-Lightning 证明该路线可以迁移到视频扩散模型。
```

然后再接我们的实验：

```text
Wan2.1-T2V DMD2: 50 -> 8 -> 4
低 batch 失败 -> 提升 LR/batch 后 8-step 可用 -> 8node 4-step from 8 可用
```

这样逻辑链条最完整。

## 7. 可直接复用的总结表述

> 我们的渐进式蒸馏路线首先受到 Progressive Distillation 的启发。该工作证明了 diffusion sampler 可以通过逐阶段压缩采样步数来实现快速生成，但其验证主要集中在图像扩散模型上，对视频生成任务只有方法论参考。进一步地，ByteDance 的 SDXL-Lightning 将 progressive distillation 与 adversarial distillation 结合，指出普通 MSE 渐进蒸馏在少步阶段容易变糊，而对抗性目标可以提升少步生成的清晰度和细节。随后，AnimateDiff-Lightning 将这一 progressive adversarial distillation 路线迁移到视频扩散模型，并通过 cross-model distillation 和 flow-conditional video discriminator 实现高质量 4-step 视频生成。这两项 Lightning 工作与我们的 Wan2.1-T2V DMD2 蒸馏更接近，说明 `50 -> 8 -> 4` 的 progressive route 具有明确文献支撑。我们的实验也与其经验一致：低有效 batch 和不稳定训练会导致 8-step/4-step 质量下降，而提升 learning rate、batch 和有效训练稳定性后，4-step from 8 的视频质量和物理一致性明显改善。

## 8. 参考文献

1. Tim Salimans, Jonathan Ho. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512). ICLR 2022.
2. Shanchuan Lin, Anran Wang, Xiao Yang. [SDXL-Lightning: Progressive Adversarial Diffusion Distillation](https://arxiv.org/abs/2402.13929). arXiv 2024.
3. SDXL-Lightning HTML version: [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.13929).
4. ByteDance. [SDXL-Lightning model card](https://huggingface.co/ByteDance/SDXL-Lightning).
5. Shanchuan Lin, Xiao Yang. [AnimateDiff-Lightning: Cross-Model Diffusion Distillation](https://arxiv.org/abs/2403.12706). arXiv 2024.
6. AnimateDiff-Lightning HTML version: [ar5iv](https://ar5iv.labs.arxiv.org/html/2403.12706v1).
7. ByteDance. [AnimateDiff-Lightning model card](https://huggingface.co/ByteDance/AnimateDiff-Lightning).
