# experiment/ - 实验记录

工作线二：Wan2.1-T2V-1.3B 上的 DMD2 蒸馏加速评估与复现。每次实验独立记录，报告路径、日志路径、脚本路径都以当前整理后的路径为准。

## 命名约定

- 子文件夹：`YYYY-MM-DD-<简短实验名>/`，日期用实验当天真实日期。
- 记录文件：同名 `YYYY-MM-DD-<简短实验名>.md`，放在该子文件夹内。
- 子文件夹内通常包含：报告 `.md`、原始结果 `.json/.csv`、样例视频路径、`scripts/` 复现实验脚本副本或脚本路径索引。
- 本记录是当前 Wan DMD2 工作线的主索引，后续新实验继续追加到“索引”和“进行中”部分。

## 索引

| 日期 | 实验 | 一句话 |
|---|---|---|
| 2026-06-15 | `wan-dmd2-4step-eval-sweep` | DMD2 4-step student 对 1000-5000 checkpoint 做 10 prompt 推理评估：teacher 50-step 平均 165.24s，4-step student 平均约 6.59-6.63s，速度约 25x；0000500 checkpoint 缺失。 |
| 2026-06-15 | `wan-dmd2-report-log-integration` | 生成 teacher-student 同步播放 HTML 报告，并把训练日志统计加入报告；可用 train log 覆盖 iter 1520-5140。 |
| 2026-06-16 | `wan-dmd2-8step-training-retry` | 重新配置 8-step student，修正 `student_sample_steps=8`；训练中 GPU3 出现 `Unknown Error`，随后从 500 checkpoint 用 7 张卡恢复训练到 2530。 |
| 2026-06-17 | `wan-dmd2-8step-2500-infer` | 8-step student `0002500` 已完成 10 prompt 高质量 MP4 推理，平均采样 13.16s，但肉眼质量明显差于 4-step `0001000`。 |
| 2026-06-17 | `wan-dmd2-8step-0500-2000-eval5` | 8-step student `0000500/0001000/0001500/0002000` 已完成前 5 条 prompt 高质量 MP4 推理，用于定位 8-step 质量从早期到中期的变化。 |
| 2026-06-18 | `wan-dmd2-8step-3000-5000-eval5` | 8-step student `0003000/0003500/0004000/0004500/0005000` 已完成前 5 条 prompt 推理；肉眼观察质量相比早期 checkpoint 有提升，但提升幅度有限，仍明显落后于 4-step 优质 checkpoint。 |
| 2026-06-17 | `wan-dmd2-next-debug` | 进行中：分析 8-step 质量退化，重点排查 `t_list` 设计、每个时间锚点 student update 不足、7 卡续训造成的 batch 改变，以及是否应放弃当前 8-step 作为 4-step 初始化。 |

## 已完成实验

### 1. DMD2 4-step checkpoint sweep

- 目标：比较 teacher 50-step 与 4-step student `0001000-0005000` 在同一组 10 prompt 上的速度和质量变化。
- 结果：teacher 平均 `165.24s`；4-step student 平均约 `6.59-6.63s`，速度约 `25x`；`0000500` 缺失，`0001000-0005000` 完整。
- 报告：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/index.html`
- 指标：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/metrics.csv`
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/per_prompt_times.csv`
- 整理后日志：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/logs/organized/inference/eval_10prompts/`
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/logs/organized/training/train_stats.csv`
- 复现脚本：
  - `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_infer_dmd2_eval_sweep.sh`

### 2. 10 prompt eval set

- 目标：把原 prompt 扩展为 10 条覆盖运动、液体、反射、机械、动物、人物等失败模式的评估集。
- Prompt 文件：
  - `/data/chenqingzhan/FastGen/scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts.txt`
- Negative prompt：
  - `/data/chenqingzhan/FastGen/scripts/inference/prompts/negative_prompt.txt`

### 3. 8-step student `0002500` inference

- 目标：检查 8-step student `0002500` 能否作为下一阶段 4-step progressive distillation 的初始化。
- 结果：10 prompt 推理完成，高质量 MP4 输出正常；平均 sampling time `13.16s`，但质量明显糟糕，模糊且不如 4-step `0001000`。
- 视频：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/inference/eval_10promprs/0002500_student_8step/wan21_dmd2_openvid_eval_prompts`
- 整理后日志：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/logs/organized/inference/infer_0002500_student_8step_20260617_102410.log`
- 复现脚本：
  - `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_infer_dmd2_step8_2500.sh`
- 相关修复：
  - `scripts/inference/video_model_inference.py` 中 `save_high_quality=True` 分支已显式设置 `save_as_gif=False`，避免高质量输出误存为 GIF。

### 4. 8-step student `0000500-0002000` eval-5 sweep

- 目标：只跑前 5 条 prompt，快速判断 8-step student 质量是早期已坏、还是训练过程中退化。
- Checkpoints：`0000500/0001000/0001500/0002000`。
- 视频根目录：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/inference/eval_5prompts`
- 整理后日志：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/logs/organized/inference/eval_5prompts_step8/`
- 复现脚本：
  - `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_infer_dmd2_step8_eval5_sweep.sh`
- 文件核验：每个 checkpoint 均为 `5` 个 MP4，`0` 个 GIF。

### 5. 8-step student `0003000-0005000` eval-5 sweep

- 目标：延续前 5 条 prompt 快速评估，检查 `2500` 之后继续训练是否改善 8-step student 的模糊问题。
- Checkpoints：`0003000/0003500/0004000/0004500/0005000`。
- 视频根目录：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/inference/eval_5prompts`
- 日志根目录：
  - `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/logs/eval_5prompts_step8`
- 复现脚本：
  - `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_infer_dmd2_step8_eval5_sweep.sh`
- 肉眼结论：`0003000-0005000` 相比 `0000500-0002000` 有可见质量提升，说明继续训练确实在修复一部分模糊和结构问题；但提升空间已经不大，整体仍没有接近 4-step `0001000` 等优质 checkpoint 的清晰度、局部细节和稳定性。

## 进行中实验与卡点

### 8-step student 质量退化

当前判断：`0002500` 之后继续训练到 `0005000` 的确带来可见改善，但收益有限；当前 8-step run 仍不适合作为 4-step 下一阶段初始化，除非后续用新的 schedule / 配置排查证明质量可恢复。

主要证据：

- `0002500` 在同一 10 prompt 上明显比 4-step `0001000` 更糊。
- 8-step 平均采样约 `13.16s`，速度约是 4-step 的 2 倍慢，但质量没有换来提升。
- 8-step 训练最终可见日志为 `world_size=7`，而原 4-step run 是 `world_size=8`；断点发生后从 `0000500` 恢复，训练条件不完全一致。
- DMD2 中 `student_update_freq=5`，所以 `2500 iter` 约等于 `500` 次 student update；8-step 又把训练覆盖分摊到 8 个时间锚点，每个锚点的有效 student 监督偏少。
- 当前 8-step `t_list` 为：
  - `[0.999, 0.968, 0.937, 0.885, 0.833, 0.729, 0.624, 0.312, 0.0]`
  - 高噪声段过密，尾部 `0.312 -> 0.0` 仍然过大，可能导致细节恢复不足和误差累积。

代码适配排查：

- 不是明显的推理代码不适配：`scripts/inference/video_model_inference.py` 在 student 推理时会传入 `student_sample_steps=model.config.student_sample_steps` 和 `t_list=model.config.sample_t_cfg.t_list`；`fastgen/methods/model.py::generator_fn` 也会校验 `len(t_list)-1 == student_sample_steps`。
- 推理日志已打印 `Evaluating student sample steps: 8`，且单条 sampling time 约 `13.2s`，约为 4-step 的 2 倍，说明实际确实执行了 8 个 student step，而不是误跑 4-step。
- 更可能的问题是训练 recipe 没学稳：DMD2 默认 `student_update_freq=5`，`5000 iter` 约 `1000` 次 student 更新；分到 8 个区间后，每个区间平均有效更新显著少于 4-step。多一步并不会自动更清晰，如果每个子区间的映射都弱，8 次误差累积会比 4 次强映射更差。
- 当前 8-step `t_list` 是从 4-step anchor 手工插值，高噪声段密、低噪声细节段仍有大跳变；这会让模型在结构阶段反复训练，但在纹理/清晰度恢复阶段监督不足。
- 训练中断、坏卡后 7 卡续训和 batch/日志口径变化是额外变量，但从 `0003000-0005000` 仍只有限改善看，主因更像 schedule 与有效 student 更新不足，而不是单纯训练没跑够。

下一步建议：

1. 暂停把当前 8-step checkpoint 直接用于 `8step -> 4step`；优先排查 8-step 训练配置、时间步 schedule 和 loss/更新频率，再决定是否重训 8-step。
2. 重新设计 8-step `t_list`，优先尝试更均匀或中低噪声补点的 schedule，例如：
   - `[0.999, 0.875, 0.750, 0.625, 0.500, 0.375, 0.250, 0.125, 0.0]`
   - `[0.999, 0.937, 0.833, 0.729, 0.624, 0.468, 0.312, 0.156, 0.0]`
3. 若继续 7 卡训练，配置中显式写 `config.trainer.batch_size_global = 7`，避免记录与真实 global batch 不一致。

## 当前脚本路径

| 用途 | 路径 |
|---|---|
| 4-step checkpoint sweep inference | `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_infer_dmd2_eval_sweep.sh` |
| 8-step training config | `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_dmd2_step8_2k.py` |
| 8-step `0002500` inference | `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_infer_dmd2_step8_2500.sh` |
| 4-step from 8-step config draft | `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_dmd2_step4_from_step8_2p5k.py` |
| 4-step from 8-step train draft | `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/run_train_dmd2_step4_from_step8_2p5k.sh` |

## 整理后的日志路径

| 实验 | 整理后路径 |
|---|---|
| 4-step training scalar log | `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/logs/organized/training/train_stats.csv` |
| 4-step inference sweep logs | `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/logs/organized/inference/eval_10prompts/` |
| 8-step training scalar log | `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/logs/organized/training/train_stats.csv` |
| 8-step `0002500` inference log | `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/logs/organized/inference/infer_0002500_student_8step_20260617_102410.log` |
| 8-step eval-5 logs | `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8_step8/logs/organized/inference/eval_5prompts_step8/` |

## 注意事项

- `eval_10promprs` 是当前视频输出目录的历史拼写，暂时保留，避免破坏已有路径引用。
- `logs/organized/` 是符号链接视图，原始日志仍保留在原位置。
- 8-step 现有 checkpoint 可用于诊断，但暂不建议作为 4-step progressive distillation 的初始化。
