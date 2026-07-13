# Wan2.2 TI2V 5B Execution Plan

这份文档用于启动下一阶段工作：在服务器的 `FastGen` 框架上验证并推进 `Wan2.2 TI2V 5B` 蒸馏。

## 1. 当前基准

当前阶段应以最近这组正式 run 作为主基准：

- `DMD2 32-step / 5000 iter / 500 ckpt`
- `CD 32-step / 5000 iter / 500 ckpt`
- 完成时间：`2026-04-18`
- 已产出：`1000~5000` checkpoint
- 已补做：`16 / 32 / 50` step-offset inference

它们的价值不是证明最终 teacher 已经正确，而是证明以下基础设施已经可靠：

- FastGen 训练侧可以稳定长时间运行
- 当前服务器的 `6-GPU` 并行策略可行
- checkpoint 节奏、推理评估协议、归档方式已经建立

因此，`Wan2.2 TI2V 5B` 阶段不应从零重新设计流程，而应复用这套节奏。

## 2. 当前已知前提

根据现有周报和汇报稿，服务器上的 FastGen 已经被确认包含：

- `WanI2V` 网络实现
- `Wan22_I2V_5B_Config`
- `fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
- `fastgen/configs/experiments/WanI2V/config_sft_wan22_5b.py`
- I2V inference 入口

当前仓库本地还没有这部分源码镜像，因此正式启动前仍需在服务器上做一次只读确认。

## 3. 分阶段执行

### Stage 0: Server Read-Only Audit

目标：确认服务器上的 FastGen 代码与模型资产确实存在。

必须确认的内容：

1. `FastGen` 当前 commit
2. `WanI2V` 相关源码和 config 文件是否存在
3. `Wan2.2-TI2V-5B-Diffusers` 权重目录是否存在
4. I2V/TI2V 对应推理脚本是否存在
5. 当前服务器上是否已有合适的数据目录或 dataloader 样例

通过条件：

- 上述 5 项全部存在

失败条件：

- 缺少 config
- 缺少模型权重
- 当前 dataloader 与目标任务不匹配

### Stage 1: Inference Smoke Test

目标：先确认模型可加载、推理可完成、输出路径正常。

建议设置：

- 单卡
- 固定 `seed`
- 固定 `1~2` 个 prompt
- 仅做 teacher/native inference

通过条件：

- 能成功生成结果
- 无 shape mismatch
- 单次推理显存和耗时可记录

### Stage 2: DMD2 Smoke Run

目标：先验证服务器现成原生 `Wan2.2 TI2V 5B` `DMD2` 配置能完成训练初始化和首轮 loss。

建议设置：

- 方法：`DMD2`
- 配置起点：`fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
- 注意：
  - 服务器当前这份原生 config 默认是 `2-step training`
  - 因此第一里程碑不是立刻复刻 `50 -> 32 step`
  - 而是先确认原生 `2-step` 路线在当前环境下可运行
- iter：`50~100`
- checkpoint：不需要频繁保存
- 优先单卡或最小 FSDP 组

通过条件：

- 完成初始化
- 进入训练循环
- 正常打印 loss

### Stage 3: Mid-Run Validation

目标：在正式长训前确认原生 `2-step` 路线的稳定性。

建议设置：

- iter：`1000`
- checkpoint：每 `100` 或 `200`
- 固定 prompts 做推理抽查

重点观察：

- loss 是否稳定
- 是否出现 OOM
- checkpoint 是否可恢复
- student sampling 是否可正常运行

### Stage 4: Formal Run

目标：先复用当前成功实验的节奏，完成 `Wan2.2 TI2V 5B` 原生 `DMD2` 正式阶段验证。

建议设置：

- 方法：优先 `DMD2`
- iter：`5000`
- checkpoint：每 `500`
- 评估协议：尽量沿用当前 `seed / prompts / checkpoint cadence`

只有在 Stage 3 稳定后才进入这一阶段。

### Stage 5: Step-Schedule Adaptation

目标：在原生 `2-step` 路线稳定后，再判断是否改造成更接近当前主线经验的少步蒸馏设定。

这一阶段再回答两个问题：

1. 现成 `WanI2V` 路线能否从原生 `2-step` 改到更接近 `32-step`
2. 改动后是否仍能保持 teacher/student 接口和训练稳定性

## 4. 为什么先选 DMD2

下一阶段主线建议优先选 `DMD2`，原因是：

- 你最近最成功、最完整的正式 run 之一就是 `DMD2`
- `DMD2` 已经提供了较稳定的预算基线
- 现有资料里对 `Wan2.2 TI2V 5B` 的明确入口就是 `config_dmd2_wan22_5b.py`
- 服务器实查确认该 config 已存在，且当前是原生 `2-step` 训练入口

`CD` 仍然保留价值，但应作为 teacher path 或 consistency side 的辅助对照，而不是第一条 `Wan2.2` 主线。

## 5. 当前阻塞项

当前机器到服务器的 SSH 已经到达认证阶段，但还不能直接登录：

- 主机：`111.17.197.107`
- 用户：`chenqingzhan`
- 认证现状：服务器要求 `password`，本机 `~/.ssh/id_ed25519` 未被接受

因此当前最小解锁方式有两种：

1. 把本机公钥加入服务器 `authorized_keys`
2. 直接提供服务器密码

拿到访问后，优先执行 Stage 0。

## 6. 进入正式运行前的 Go / No-Go

满足以下条件才进入正式 run：

- `WanI2V/config_dmd2_wan22_5b.py` 在服务器上真实存在
- `Wan2.2-TI2V-5B-Diffusers` 权重可加载
- inference smoke test 通过
- `50~100 iter` 原生 `2-step` smoke run 可打印稳定 loss
- `1000 iter` 原生 `2-step` 中间阶段无 OOM / 无 checkpoint 恢复问题

只要其中一项失败，就先停在对应阶段修问题，不直接上 `5000 iter`。

## 7. 当前服务器实查结果

截至 `2026-04-22`，已确认：

- 服务器可登录，主机名为 `RTX-5090-32G-X8`
- `FastGen` 路径为 `/data/chenqingzhan/FastGen`
- 当前 `FastGen` commit 为 `34f30e8`
- 服务器存在：
  - `fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py`
  - `fastgen/configs/experiments/WanI2V/config_sft_wan22_5b.py`
  - `fastgen/networks/WanI2V/network.py`
  - `scripts/inference/video_model_inference.py`
- `config_dmd2_wan22_5b.py` 使用的是 `VideoLoaderConfig`
- 服务器原生 config 当前默认 `student_sample_steps = 2`
- `OpenVid-1M` WebDataset 路径 `/data/datasets/OpenVid-1M/webdataset` 可见
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 仓库可被 `huggingface_hub` 解析
- 完整模型下载已在服务器后台启动
