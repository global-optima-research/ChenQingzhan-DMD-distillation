# FFGO-FastGen Feasibility Plan

## Goal

当前目标不是验证最终业务效果，而是验证一条可持续推进的蒸馏闭环：

- teacher 能被 FastGen 训练侧调用
- student 能完成全参数训练
- `50 -> 32 steps` 的设置能正常训练、存 ckpt、做验证
- 先跑 `1000 iter` smoke run，再切到 `5000 iter / every 500 ckpt` 的正式阶段

## Current Constraints

1. 服务器上目前没有公开 `FFGO` 的 adapter 或 merge 后 teacher 权重。
2. 这份 FastGen 代码库没有现成的 LoRA adapter 加载入口。
3. FastGen 当前稳定支持的是 `Wan2.1 T2V 1.3B` 与 `Wan2.2 TI2V 5B`，没有现成的 `Wan2.1 3B` 配置。
4. 如果采用公开 FFGO 路线，其 teacher 属于 `Wan2.2 I2V/TI2V` 家族，和 `Wan2.1 1.3B` student 在 latent shape 上并不兼容，不能直接做现有 loss 下的 teacher-student 蒸馏。

## Practical Decision

因此当前分两阶段推进：

### Stage 1: CD smoke run

- 方法：`Consistency Distillation (CD)`
- student：`Wan2.1 T2V 1.3B` 全参数训练
- target steps：`32`
- iter：`1000`
- data：现有 `OpenVid-1M` raw video WDS
- teacher path：独立变量 `FFGO_TEACHER_PATH`

这一阶段已经成功跑通 FastGen 训练闭环，并在服务器上完成首轮训练迭代。

### Stage 2: DMD2 stage run

- 方法：`DMD2`
- student：`Wan2.1 T2V 1.3B` 全参数训练
- target steps：`32`
- iter：`5000`
- checkpoint：每 `500 iter`
- data：现有 `OpenVid-1M` raw video WDS
- teacher path：独立变量 `FFGO_TEACHER_PATH`

当前默认让：

- `FFGO_TEACHER_PATH = Wan2.1-T2V-1.3B base path`

这样做的意义是先验证 FastGen 侧蒸馏链路本身，而不是在 teacher 资产未齐时卡死；一旦有 merged FFGO teacher，可直接替换 teacher 路径继续实验。

### Stage 3: Parallel stage run under 6-GPU budget

- 优先级：`DMD2 > CD > MeanFlow`
- 当前资源约束：总共只使用 `6` 张卡
- 当前实际部署：
  - `DMD2 5000 iter / 500 ckpt` on `GPU 0,1,2,3`
  - `CD 5000 iter / 500 ckpt` on `GPU 4,5`
  - `GPU 6` 保留为空闲
  - `MeanFlow 5000 iter / 500 ckpt` 仅保留脚本和配置，暂不启动

这样安排的原因是：

1. `DMD2` 更接近当前主目标，优先级最高；
2. `CD` 显存相对更轻，适合在剩余 `2` 张卡上做并行对照；
3. `MeanFlow` 先不占卡，等 `DMD2/CD` 首个 `500 iter` checkpoint 产出后再决定是否补跑。

## Current Execution Status

截至 `2026-04-18`：

- `DMD2` run name: `ffgo_dmd2_32step_5000iter_20260417_6gpu_dmd`
  - `2026-04-17 22:13:41 CST` 开始
  - `2026-04-18 21:38:15 CST` 完成
  - 总时长约 `23h 25m`
  - `4950 iter` 最新统计：
    - `avg_total_loss = 1.1418`
    - `avg_fake_score_loss = 0.0928`
    - `avg_gan_loss_disc = 1.2658`
  - 已保存 `1000/2000/3000/4000/5000` checkpoint

- `CD` run name: `ffgo_cd_32step_5000iter_20260417_6gpu_cd`
  - `2026-04-17 22:13:41 CST` 开始
  - `2026-04-18 19:49:42 CST` 完成
  - 总时长约 `21h 36m`
  - `4950 iter` 最新统计：
    - `avg_total_loss = 498.5971`
    - `avg_cm_loss = 498.5971`
    - `avg_unweighted_cm_loss = 115.9439`
  - 已保存 `1000/2000/3000/4000/5000` checkpoint

- 实验结论：
  - `6-GPU` 并行训练策略成立；
  - `DMD2` 与 `CD` 都完成了正式阶段训练闭环；
  - 当前已经具备以 checkpoint 为单位做横向推理对比的条件。

截至 `2026-04-20`，基于上述训练结果又补做了 checkpoint 推理归档：

- checkpoint 范围：`1000 / 2000 / 3000 / 4000 / 5000`
- 方法：`DMD2` 与 `CD`
- 每个 checkpoint：固定 `2` 个 student-sampling 视频
- prompts：`scripts/inference/prompts/eval_2prompts.txt`
- seed：`42`
- 本地归档路径：
  - [artifacts/inference_videos/2026-04-20/dmd2](/Users/tchc5201/Desktop/ChenQingzhan-DMD-distillation/artifacts/inference_videos/2026-04-20/dmd2)
  - [artifacts/inference_videos/2026-04-20/cd](/Users/tchc5201/Desktop/ChenQingzhan-DMD-distillation/artifacts/inference_videos/2026-04-20/cd)

这意味着当前阶段已经同时具备：

1. 正式训练日志；
2. 全量 `1000~5000` checkpoint；
3. 对应 checkpoint 的固定 prompt 推理样例。

## Runtime Estimate

当前已有实测数据：

- `DMD2 5000 iter`：约 `23h 25m`
- `CD 5000 iter`：约 `21h 36m`
- 每 `500 iter` checkpoint：约 `2.2 ~ 2.4` 小时

因此在当前 `6-GPU` 并行策略下，总墙钟时间约 `1` 天，瓶颈主要在 `DMD2`。

## Upgrade Path

后续如果要真正切到 FFGO teacher，需要满足二选一：

1. 内部 `FFGO` 实际是 `Wan2.1 1.3B` 同架构 LoRA/merged teacher。
   - 只需要把 `FFGO_TEACHER_PATH` 指向 merge 后权重。
   - 现有脚本可直接复用。

2. `FFGO` 确实是 `Wan2.2 I2V/TI2V` 体系。
   - 需要把 student 也切到兼容的 `Wan2.2` 家族。
   - 或新增跨架构蒸馏对齐逻辑。
   - 当前这套 `Wan2.1 1.3B student` 方案不能直接成立。
