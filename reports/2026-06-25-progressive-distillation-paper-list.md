# Progressive Distillation Paper List for Wan T2V DMD2

Date: 2026-06-25

Current project question: whether the `50 -> 8 -> 4` progressive distillation route has paper support.

Short answer: yes. The exact combination `Wan T2V + DMD2 + 50 -> 8 -> 4` is an engineering route in our codebase, but the core idea is well supported by diffusion distillation literature. The strongest evidence is:

- Progressive distillation explicitly trains a student by repeatedly reducing the number of sampling steps, down to 4 steps.
- Consistency/latent consistency models support few-step and multi-step inference as a quality-speed tradeoff.
- DMD/DMD2 gives the distribution-matching framework closest to our current method.
- Recent video papers show that 50-step video teachers can be distilled to 4-step students, often using consistency, adversarial, distribution, or reward-based losses.

## Ranked Papers


| Rank | Title                                                                                                    | URL                                                                  | Venue / status      | 大概内容与方向                                                                                                                                              | 与当前 `50 -> 8 -> 4` 的关系                                                                      |
| ---- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1    | Progressive Distillation for Fast Sampling of Diffusion Models                                           | [https://arxiv.org/abs/2202.00512](https://arxiv.org/abs/2202.00512) | ICLR 2022           | 提出把多步 deterministic diffusion sampler 蒸馏成更少步 student，并反复执行该过程，每阶段减少采样步数。论文从高步数逐步蒸馏到 4 step，质量损失较小。                                                   | 这是 `50 -> 8 -> 4` 最直接的理论来源。我们的路线可以表述为 progressive step reduction，而不是一次性把 50 step 压到 4 step。 |
| 2    | Improved Distribution Matching Distillation for Fast Image Synthesis                                     | [https://arxiv.org/abs/2405.14867](https://arxiv.org/abs/2405.14867) | arXiv 2024, DMD2    | DMD2 改进 DMD：去掉昂贵 teacher pair regression，使用 two-time-scale update、GAN loss，并显式讨论 multi-step sampling 的 train-inference mismatch。                     | 与我们当前方法名和训练逻辑最接近。尤其支持“multi-step student 需要模拟推理时输入分布”，可解释 8 step/4 step 训练稳定性问题。            |
| 3    | One-step Diffusion with Distribution Matching Distillation                                               | [https://arxiv.org/abs/2311.18828](https://arxiv.org/abs/2311.18828) | arXiv 2023, DMD     | 提出 DMD，用 distribution matching 让 student 输出分布接近 teacher，而不是逐轨迹拟合 teacher。关注 one-step/few-step 高速生成。                                                  | DMD2 的前身。说明少步蒸馏不一定要逐点模仿 teacher trajectory，也可以做分布级匹配。                                       |
| 4    | Consistency Models                                                                                       | [https://arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469) | ICML 2023           | 提出 consistency model，可从 diffusion teacher 蒸馏，也可独立训练；支持 one-step 生成，也支持 multi-step sampling 提升质量。                                                     | 支持“少步 student + 多步采样质量速度权衡”的基础路线。和 DMD2 不同，但同样证明少步生成可行。                                     |
| 5    | Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference                   | [https://arxiv.org/abs/2310.04378](https://arxiv.org/abs/2310.04378) | arXiv 2023, LCM     | 将 consistency distillation 放到 latent diffusion 上，实现 2-4 step 高分辨率图像生成。                                                                               | 支持 latent-space few-step distillation。对 Wan 这类 latent/video diffusion 有方法论参考。               |
| 6    | T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback         | [https://arxiv.org/abs/2405.18750](https://arxiv.org/abs/2405.18750) | arXiv 2024          | 面向 text-to-video，把 consistency distillation 和 reward feedback 结合；报告 4-step T2V student 可超过 50-step teacher 的 VBench 表现。                              | 视频领域非常相关。说明 50-step teacher 到 4-step video student 是被近期工作认真验证过的路线。                          |
| 7    | DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization                        | [https://arxiv.org/abs/2412.15689](https://arxiv.org/abs/2412.15689) | arXiv 2024, 2026 修订 | 结合 variational score distillation、consistency distillation 和 latent reward optimization，做 10 秒视频 few-step 生成。报告 4-step student 与 50-step teacher 对比。 | 强支持“视频少步蒸馏 + reward/quality feedback”方向。后续如果我们要改善物理规则和清晰度，可参考 reward 模块。                    |
| 8    | Motion Consistency Model: Accelerating Video Diffusion with Disentangled Motion-Appearance Distillation  | [https://arxiv.org/abs/2406.06890](https://arxiv.org/abs/2406.06890) | NeurIPS 2024        | 指出图像蒸馏方法直接套到视频上可能导致帧质量不佳；提出把 motion distillation 和 appearance enhancement 拆开，并处理 training-inference discrepancy。                                     | 对我们当前问题很关键：8 step student 模糊、物理规则差，可能不是“步数越多越好”，而是视频蒸馏需要分开处理运动和外观目标。                        |
| 9    | VideoLCM: Video Latent Consistency Model                                                                 | [https://arxiv.org/abs/2312.09109](https://arxiv.org/abs/2312.09109) | arXiv 2023          | 把 LCM/consistency distillation 扩展到 video latent diffusion，实现少步视频生成，并强调 temporal consistency。                                                         | 早期视频 LCM baseline。支持 latent video diffusion 可以用 4-step 级别的 consistency student。             |
| 10   | AnimateDiff-Lightning: Cross-Model Diffusion Distillation                                                | [https://arxiv.org/abs/2403.12706](https://arxiv.org/abs/2403.12706) | arXiv 2024          | 用 progressive adversarial diffusion distillation 加速 AnimateDiff，提出跨 base model 的 motion module distillation。                                         | 直接关联视频生成加速，且明确使用 progressive adversarial distillation。对“4 step 为什么能好”有参考价值。                 |
| 11   | SDXL-Lightning: Progressive Adversarial Diffusion Distillation                                           | [https://arxiv.org/abs/2402.13929](https://arxiv.org/abs/2402.13929) | arXiv 2024          | 在 SDXL 上结合 progressive distillation 和 adversarial distillation，训练 1/2/4/8 step 高速模型。                                                                 | 虽然是图像模型，但对我们的 `8 -> 4` 阶段很重要：少步质量通常需要 progressive + adversarial/real-data 约束共同稳定。           |
| 12   | Adversarial Diffusion Distillation                                                                       | [https://arxiv.org/abs/2311.17042](https://arxiv.org/abs/2311.17042) | arXiv 2023          | 用 score distillation teacher signal 加 adversarial loss，把大型图像 diffusion 蒸馏到 1-4 step。                                                                 | 说明单靠轨迹/分布损失可能不够，adversarial 或真实图像约束常用于保证低步数图像清晰度。                                           |
| 13   | Phased Consistency Models                                                                                | [https://arxiv.org/abs/2405.18407](https://arxiv.org/abs/2405.18407) | arXiv 2024          | 分析 LCM 在高分辨率 text-conditioned latent 生成中的缺陷，提出 phased 设计，覆盖 1-16 step，并声称可扩展到视频。                                                                     | 支持“不同 step 不应同质处理”的思想。对我们 8 step 不如 4 step 的异常现象有解释价值：多步 student 的 timestep/phase 设计可能更敏感。  |
| 14   | TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation                              | [https://arxiv.org/abs/2303.04248](https://arxiv.org/abs/2303.04248) | arXiv 2023          | 扩展 binary time-distillation，用 transitive closure 思路做 time distillation，提高单步/少步 DDIM 表现。                                                              | 和 progressive halving 思路接近，说明“时间段合并/逐步压缩”是已有方法线。                                            |
| 15   | InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation                  | [https://arxiv.org/abs/2309.06380](https://arxiv.org/abs/2309.06380) | arXiv 2023          | 基于 rectified flow 和 reflow，把 Stable Diffusion 蒸馏成 one-step 文生图模型。                                                                                    | 不是 DMD2，但属于“把多步生成轨迹变直/变少步”的强相关加速路线，可作为 related work。                                        |
| 16   | Flash Diffusion: Accelerating Any Conditional Diffusion Model for Few Steps Image Generation             | [https://arxiv.org/abs/2406.02347](https://arxiv.org/abs/2406.02347) | arXiv 2024          | 面向 conditional diffusion 的快速 few-step 蒸馏方法，强调较低训练成本和多任务适配。                                                                                           | 可作为工程对照：少步蒸馏不仅有 progressive/DMD/LCM，也有更通用的 few-step distillation 系列。                        |
| 17   | Learning Few-Step Diffusion Models by Trajectory Distribution Matching                                   | [https://arxiv.org/abs/2503.06674](https://arxiv.org/abs/2503.06674) | arXiv 2025          | 结合 trajectory matching 与 distribution matching，提出 sampling-steps-aware objective，并扩展到 text-to-video。                                                 | 很贴近我们的痛点：DMD 多步不够灵活、trajectory matching 质量有限，该论文试图在二者之间折中。                                  |
| 18   | Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis | [https://arxiv.org/abs/2507.18569](https://arxiv.org/abs/2507.18569) | arXiv 2025          | 指出 DMD 的 reverse-KL 可能带来 mode collapse/mode seeking，提出 adversarial distribution matching，并应用到图像和视频模型。                                                | 可用于解释训练后期质量不升反降、物理规则/多样性变差的现象。若后续继续 DMD2，可关注 mode collapse 风险。                              |


## What This Means for Our Experiments

1. `50 -> 8 -> 4` 有文献依据。最稳的表述是：我们采用 progressive few-step distillation，将 50-step teacher 先蒸馏到 8-step student，再进一步压缩到 4-step student。
2. 直接从 50 step 到 4 step 也有论文做，但 progressive 路线通常更容易训练，因为每一阶段 teacher-student gap 更小。
3. 8 step 理论上不一定天然比 4 step 好。多步 student 需要学习更多 timestep/transition 的一致性，如果 batch、learning rate、timestep sampling 或 train-inference input distribution 不匹配，8 step 可能反而模糊或物理不稳定。
4. 我们当前观察到 `8 step lr=1e-5, batch=12` 明显优于低学习率小 batch 的结果，与文献中的稳定训练需求一致。
5. `4 step from 8 step` 中 500 ckpt 最好、后续 ckpt 物理规则崩坏，可以在报告中解释为 few-step distillation 的过训练/分布坍缩/teacher-student mismatch 风险。DMD2、MCM、DOLLAR、ADM/DMDX 都给了相关线索。

## Recommended Related Work Framing

For the report, the related work can be grouped as:

- Progressive step distillation: Progressive Distillation, TRACT.
- Consistency/latent consistency distillation: Consistency Models, LCM, VideoLCM, T2V-Turbo, DOLLAR.
- Distribution matching distillation: DMD, DMD2, TDM, ADM/DMDX.
- Adversarial/reward-enhanced few-step generation: ADD, SDXL-Lightning, AnimateDiff-Lightning, T2V-Turbo, DOLLAR.
- Video-specific failure modes: Motion Consistency Model, DOLLAR, T2V-Turbo.

Practical claim for our project:

> Our `50 -> 8 -> 4` route is best understood as progressive few-step distillation for text-to-video diffusion. It is supported by progressive distillation and consistency distillation literature, while DMD2 provides the concrete distribution-matching training objective used in our implementation. The remaining challenge is not whether the route is valid, but how to stabilize multi-step video distillation under limited batch size and compute.
