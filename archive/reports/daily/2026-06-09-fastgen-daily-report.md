# 2026-06-09_UST_projectDailyReport

# 2026-06-09 UST Project 日报

## 今日重点

今天主要围绕 OpenVide 上的视频生成蒸馏训练做可行性验证，重点确认 DMD2 和 fdistill 在小 global batch 设置下的训练可运行性、生成效果和收敛问题。同时开始探索 MeanFlow 方向，并完成 Sana lab 相关工作的服务器环境配置。

## 主要进展

- 确认 DMD2 在 OpenVide 数据上可使用 `8` 卡 RTX 5090 32G 进行 FSDP 训练，单卡 batch size 为 `1`，global batch size 为 `8`。
- DMD2 当前训练速度约为 `16.8s/iter`。
- DMD2 保存 `1000` step checkpoint 后，生成效果整体较好，说明当前配置下训练链路和初步生成质量是可用的。
- 探索 fdistill 训练配置，确认 `4` 卡、单卡 batch size 为 `1`、global batch size 为 `8` 的设置可以正常训练。
- fdistill 保存 `1500` step checkpoint 后，生成效果较差，当前配置下还没有达到可用质量。
- 观察到 DMD2 和 fdistill 的 loss 都存在较大波动，整体表现出不太能稳定收敛的情况。
- 参考 Kaiming 相关论文中的 batch size 与 learning rate 线性缩放思路，当前从 global batch size `64` 降到 `8`，理论上 learning rate 应同步降到原来的 `1/8`。
- 开始关注 MeanFlow 方案。MeanFlow 支持 1-step 生成，目标形式在数学上与当前蒸馏任务较接近，后续值得作为候选方案尝试。
- 今日探索 Sana lab 相关工作，并完成服务器环境下的基础配置。

## 当前结论

当前 DMD2 的训练可行性和阶段性生成效果优于 fdistill：DMD2 在 `1000` step checkpoint 已有较好生成效果，而 fdistill 在 `1500` step checkpoint 效果仍较差。不过两个方法都存在 loss 波动明显的问题，说明目前主要瓶颈可能不只是能否训练，而是小 global batch 下的学习率、优化稳定性和收敛口径需要重新调整。

从 batch size 变化看，直接沿用 global batch size `64` 的 learning rate 到 global batch size `8` 风险较高。下一步优先验证 `1/8 learning rate` 是否能改善 loss 波动和生成质量，再决定是否继续扩大训练或切换到 MeanFlow。

## 风险与建议

- 风险：DMD2 和 fdistill 在当前小 global batch 设置下 loss 波动较大，训练不够稳定。
    - 影响：即使短期 checkpoint 有可见效果，也可能难以稳定复现或继续提升。
    - 建议：优先按 batch size 线性缩放原则，将 learning rate 降到原来的 `1/8` 后重新训练，并对比 loss 曲线和固定 prompt 生成效果。
- 风险：fdistill 当前 `1500` step checkpoint 生成效果差。
    - 影响：继续按原配置训练可能消耗较多算力但收益有限。
    - 建议：fdistill 先做 `1/8 learning rate` 小规模验证；若仍无改善，再把资源优先转给 DMD2 或 MeanFlow。
- 风险：MeanFlow 数学形式接近，但工程适配和效果尚未验证。
    - 影响：可能需要额外环境、代码和训练配置成本。
    - 建议：先做最小可运行实验，确认 1-step 生成链路和训练目标能接入现有数据与评估流程。

## 明日计划

- 利用剩余空闲 `8` 卡资源启动 `1/8 learning rate` 训练实验，优先验证 fdistill，其次验证 DMD2。
- 对比原 learning rate 与 `1/8 learning rate` 下的 loss 曲线、训练稳定性和固定 checkpoint 生成效果。
- 继续整理 DMD2 / fdistill 的训练配置、checkpoint 路径、采样命令和样例结果，便于后续复现实验。
- 初步尝试 MeanFlow 1-step 生成方案，先确认代码入口、依赖环境和最小训练 / 推理链路。
- 继续推进 Sana lab 工作，完成环境稳定性检查后再进入具体实验验证。