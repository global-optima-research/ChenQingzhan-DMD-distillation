# 2026-06-06 FastGen Daily Report

## Summary

今天围绕 NVlabs/FastGen 完成了一轮从本地学习、环境隔离、数据准备、训练 smoke test、服务器整理、GitHub 同步、8 卡训练到推理验证的闭环。当前 FastGen 已经从一个刚下载的研究仓库，整理成了一个可以继续做科研实验的基础工程流程。

最重要的结果是：我们明确了 FastGen 的原子流程。

```text
数据/预训练模型资产 -> 训练 run -> 推理 run
```

并将这套流程写入仓库根目录 `LOCAL_USAGE.md`，同步到了 GitHub fork 和服务器。

## Local Work

本地仓库位置：

```text
/Users/x3y/Desktop/FastGen
```

主要完成事项：

- 使用 `uv` 创建隔离环境，避免污染全局 Python 环境。
- 完成 `uv pip install -e .` editable install，并解释 editable install 的含义。
- 补装缺失依赖 `pandas`。
- 完成 import / PyTorch 基础检查。
- 处理 W&B 登录问题，明确如何禁用 W&B 或使用 offline/disabled 模式。
- 诊断 Mac 本地 CUDA 不可用问题，确认原始配置默认面向 CUDA 训练。
- 针对本地 CPU / MPS 不适配问题，制作了 CPU smoke test config。
- 跑通 CPU smoke test，验证配置加载、teacher checkpoint 加载、dataloader 构造、训练 loop 进入、checkpoint 保存这些基础链路。

新增或整理的本地配置包括：

```text
fastgen/configs/experiments/EDM/config_dmd2_smoke_cpu.py
fastgen/configs/experiments/EDM/config_dmd2_test_cpu.py
fastgen/configs/experiments/EDM/config_dmd2_test_8gpu.py
fastgen/configs/experiments/EDM/run_dmd2_test_8gpu.sh
```

## Data And Model Assets

我们确认 FastGen 的 CIFAR-10 EDM 工作流需要两类资产：

```text
FASTGEN_OUTPUT/DATA/cifar10/cifar10-32x32.zip
FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-cond-vp.pth
FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-uncond-vp.pth
```

理解结论：

- `cifar10-32x32.zip` 是训练 dataloader 使用的数据集，FastGen 可以直接读取 zip，不需要手动解压。
- `edm-cifar10-32x32-cond-vp.pth` 是 DMD2 EDM CIFAR-10 实验使用的 conditional teacher diffusion model。
- `edm-cifar10-32x32-uncond-vp.pth` 是 unconditional EDM checkpoint，用于需要 unconditional EDM 的工作流。
- 这些文件是上游资产，不是本次训练生成的输出。

## Training Flow

我们确认 `train.py` 不是 EDM 专用脚本，而是 FastGen 的通用训练入口。

核心逻辑是：

```python
config.model_class.config = config.model
model = instantiate(config.model_class)
trainer = Trainer(config)
trainer.run(model)
```

真正决定训练什么的是 config，而不是 `train.py` 本身。

EDM DMD2 test config 的核心依赖关系：

```text
fastgen/configs/experiments/EDM/config_dmd2_test.py
  -> fastgen/configs/methods/config_dmd2.py
  -> FASTGEN_OUTPUT/DATA/cifar10/cifar10-32x32.zip
  -> FASTGEN_OUTPUT/MODEL/cifar10/edm-cifar10-32x32-cond-vp.pth
```

训练输出结构：

```text
FASTGEN_OUTPUT/fastgen/cifar10/<run_name>/
├── config.yaml
├── checkpoints/
│   ├── 0001000.pth
│   ├── 0002000.pth
│   └── ...
└── wandb/ or wandb_id.txt
```

其中 `config.yaml` 是 resolved config，是实验复现最重要的记录之一。

## Inference Flow

我们跑通了 image model inference，并解决了一个关键误区：

多卡训练 config 里如果设置了 `trainer.ddp=True`，直接用普通 `python scripts/inference/image_model_inference.py` 启动推理会报：

```text
KeyError: 'LOCAL_RANK'
```

原因是 DDP 配置要求由 `torchrun` 注入 `LOCAL_RANK` 等分布式环境变量。

正确理解：

- 多卡训练 config 可以用于训练。
- 单进程推理时，应该使用描述模型结构的 base config，例如 `config_dmd2_test.py`。
- 推理时通过 `--ckpt_path` 指向多卡训练得到的 checkpoint。

示例推理命令：

```bash
python scripts/inference/image_model_inference.py \
  --config fastgen/configs/experiments/EDM/config_dmd2_test.py \
  --classes=10 \
  --prompt_file=scripts/inference/prompts/classes.txt \
  --ckpt_path=FASTGEN_OUTPUT/fastgen/cifar10/dmd2_test_8gpu/checkpoints/0005000.pth \
  - log_config.name=test_inference_8gpu log_config.wandb_mode=disabled
```

## Server Work

服务器入口：

```text
ssh ust_ip
```

服务器 FastGen 当前主目录：

```text
/data/chenqingzhan/FastGen
```

服务器 conda 环境：

```text
fastgen
```

已确认服务器环境具备 8 GPU CUDA 训练条件：

```text
Python 3.12.12
PyTorch 2.10.0+cu128
CUDA available
8 GPUs visible
```

完成的服务器整理：

- 发现旧服务器目录非常混乱，包含历史 FastGen、FASTGEN_OUTPUT、log、shell 脚本、旧实验文件。
- 将旧实验、旧输出、根目录散落的 log/sh 文件统一归档到：

```text
/data/chenqingzhan/legacy_fastgen_20260606/
```

- 对旧 checkpoint 做清理策略：每个 checkpoint 目录只保留训练 iteration 最大的一个，其余删除。
- 释放了大量磁盘空间，避免历史实验输出继续影响新实验。
- 重新从 GitHub fork 下载干净 FastGen 仓库。
- 将 CIFAR-10 数据和 EDM checkpoints 放回仓库内的：

```text
/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/
```

而不是外部的：

```text
/data/chenqingzhan/fastgen_output
```

## Output Path Issue

我们定位到一次输出跑到 `~/fastgen_output` 的原因：

服务器 `.bashrc` 里曾经设置过：

```bash
FASTGEN_OUTPUT_ROOT=/data/chenqingzhan/fastgen_output
```

这会覆盖 FastGen 默认的仓库内输出路径。

处理策略：

- 注释掉会干扰仓库内输出的环境变量。
- 保持 upstream 风格：默认输出落在仓库内 `FastGen/FASTGEN_OUTPUT/`。
- 明确原则：如果要复现实验，优先让数据、模型资产、训练输出都在 repo-local `FASTGEN_OUTPUT` 下形成完整闭环。

## GitHub Synchronization

Fork 地址：

```text
git@github.com:Tonkmy/FastGen.git
```

今天完成的主要提交包括：

```text
Add EDM DMD2 8 GPU config
Make EDM DMD2 8 GPU config self contained
Document FastGen data training inference workflow
```

当前最新文档提交：

```text
e150434 Document FastGen data training inference workflow
```

本地和服务器均已同步到该提交。

注意：服务器上仍有一些未提交的本地实验改动，主要涉及：

```text
fastgen/configs/experiments/EDM/config_dmd2_test.py
fastgen/configs/experiments/EDM/config_dmd2_test_8gpu.py
fastgen/configs/experiments/EDM/run_dmd2_test_8gpu.sh
```

这些改动可能来自服务器上的训练实验和临时命令调整，后续需要单独决定是否保留、清理或提交。

## Key Engineering Lessons

今天最大的工程收获是理解 FastGen 的配置驱动结构。

可以概括为：

```text
入口脚本负责流程
config 负责实验定义
FASTGEN_OUTPUT 负责资产和产物落盘
checkpoint 连接训练和推理
resolved config.yaml 连接实验和复现
```

这对科研代码改造很有参考价值：

- 不要为每个实验复制训练脚本。
- 应该让训练脚本保持通用。
- 每个实验用 config 表达依赖关系和超参数。
- 数据、预训练模型、训练 checkpoint、推理结果要有稳定目录约定。
- 每次训练都保存 resolved config，方便复现和对比。

## Current Status

目前我们已经完成：

- 本地可安装、可理解、可 smoke test。
- 服务器可 8 卡训练。
- 训练 checkpoint 可用于推理。
- 推理流程已跑通。
- FastGen 数据、训练、推理的原子流程已写入仓库文档。
- GitHub fork 已经成为本地与服务器同步的主线。

## Next Steps

建议下一步围绕科研流程继续整理：

- 把服务器上未提交的 3 个实验改动做一次审查。
- 决定是否保留 `config_dmd2_test_8gpu.py` 作为标准 8 卡训练 config。
- 将临时 shell 命令整理成稳定脚本，而不是混在 config 目录或手动命令历史里。
- 为实验记录建立统一命名规则，例如 `<method>_<dataset>_<gpu>_<iter>_<date>`。
- 后续训练时，每次记录 checkpoint、config.yaml、推理 samples、日志路径和 commit hash。
