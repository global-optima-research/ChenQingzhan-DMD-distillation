# FastGen 实操手册

> 基于 NVIDIA FastGen v0.1.0 源码和 Wan2.1-1.3B 实验验证
> 最后更新: 2026-03-29
> 位置注记（2026-07-11）：本手册自 `03-dmd-distillation/` 移入 `docs/`。内容为 2026-03 的代码事实，总体仍适用于当前 WanT2V 主线；个别细节已有更新（如 `dmd2.py` 已内置近似 R1 正则 `gan_r1_reg_weight`、判别特征由冻结 teacher forward 提取），以远端代码与 `research/T0_project_analysis.md` 为准。

---

## 1. 架构概述

FastGen 是 NVIDIA 的蒸馏框架，将多步扩散模型压缩为少步生成器。

```
train.py ← 统一入口
    │
    ├── --config  Python 配置文件 (.py)，非 YAML
    ├── -  key=value 命令行覆盖（注意 - 前缀和空格）
    │
    ├── Model (fastgen/methods/)
    │   ├── Student net     ← 可训练，FSDP 分片
    │   ├── Teacher          ← 冻结 (requires_grad=False)，FSDP 分片
    │   ├── FakeScore        ← 可训练（DMD2/f-distill/CausVid/SF）
    │   ├── Discriminator    ← 可训练（DMD2/f-distill/LADD/CausVid/SF）
    │   └── Text Encoder     ← 冻结，每卡复制不分片 (~10 GB)
    │
    └── Trainer (fastgen/trainer.py)
        ├── FSDP2 分布式
        ├── Gradient Checkpointing（默认开启）
        └── bf16 混合精度
```

**关键事实（代码验证）：**
- **无 LoRA** — FastGen 不支持 LoRA，全部是全参数蒸馏
- **Python 配置** — 不是 YAML，位于 `fastgen/configs/experiments/WanT2V/`
- **Teacher 已冻结** — `model.py:204`: `self.teacher.eval().requires_grad_(False)`
- **Gradient Checkpointing 默认开启** — `network.py:630`: `enable_gradient_checkpointing()`
- **Text Encoder 是显存瓶颈** — UMT5-XXL ~10 GB/GPU，不参与 FSDP 分片

---

## 2. 蒸馏方法与硬件需求

### 2.1 方法分类

| 类别 | 方法 | 网络数 | Config | 特点 |
|------|------|--------|--------|------|
| **分布匹配** | DMD2 | S+T+FS+D (4) | `config_dmd2.py` | GAN+VSD，小算力收敛快 |
| | f-distill | S+T+FS+D (4) | `config_fdistill.py` | f-散度匹配 |
| | LADD | S+T+D (3) | `config_ladd.py` | 纯对抗，R1 正则 |
| **轨迹匹配** | MeanFlow | S+T (2) | `config_mf_video.py` | JVP 流匹配，最轻量 |
| **因果蒸馏** | CausVid | CausalS+T+FS+D (4) | `config_causvid.py` | 因果注意力 student |
| | Self-Forcing | CausalS+T+FS+D (4) | `config_sf.py` | 自回归训练 |
| **其他** | KD | S+T (2) | `config_kd.py` | 知识蒸馏，需预计算 pairs |
| | SFT | S (1) | `config_sft.py` | 监督微调，最轻 |

### 2.2 实测显存需求 (4× RTX 5090 32GB, Wan2.1-1.3B)

| 方法 | 2-GPU | 3-GPU | 4-GPU | 速度 (4-GPU) |
|------|-------|-------|-------|-------------|
| MeanFlow | ✅ 26-29 GB | ✅ | ✅ | ~30s/iter |
| DMD2 | ❌ OOM | ❌ | ✅ 22-27 GB | ~16s/iter |
| f-distill | ❌ | ❌ | ✅ 24-28 GB | ~20s/iter |
| LADD | ❌ | ❌ 30.5 GB | ✅ 29.7 GB | ~24s/iter |
| CausVid | ❌ | ❌ | ❌ 30.4 GB | — |
| Self-Forcing | ❌ | ❌ | ❌ 30.4 GB | — |

> CausVid / Self-Forcing 使用 CausalWan（因果注意力），激活值额外占 ~3 GB，4×32GB 不够。

### 2.3 显存分布 (DMD2, 4-GPU FSDP, 每卡)

| 组件 | 显存 | 可优化？ |
|------|------|---------|
| Text Encoder (UMT5-XXL) | **~10 GB** | ❌ 每卡复制 |
| Student DiT (bf16, 分片) | ~0.65 GB | FSDP 已分片 |
| Student AdamW (fp32, 分片) | ~5 GB | 8-bit 不兼容 FSDP2 |
| Teacher DiT (bf16, 分片, 冻结) | ~0.65 GB | 仅前向 |
| FakeScore + 优化器 (分片) | ~5.65 GB | 仅 DMD2/f-distill |
| Discriminator + 优化器 | ~0.16 GB | 很小 |
| 激活值 (gradient ckpt) | ~5-8 GB | 已启用 checkpointing |

---

## 3. 配置系统

### 3.1 配置层级

```
fastgen/configs/
├── methods/                    # 方法基础配置 (不要直接改)
│   ├── config_dmd2.py
│   ├── config_ladd.py
│   ├── config_mean_flow.py
│   └── ...
├── experiments/WanT2V/         # 模型+方法的组合配置
│   ├── config_dmd2.py          # ← 官方提供
│   └── our/                    # ← 我们的实验配置（见 §4）
├── data.py                     # 数据加载器定义
├── net.py                      # 网络配置 (Wan_1_3B, CausalWan_1_3B, etc.)
└── discriminator.py            # 判别器配置
```

### 3.2 关键配置字段

**Model 配置：**

```python
config.model.precision = "bfloat16"    # 模型精度
config.model.input_shape = [16, 21, 60, 104]  # [C, T_latent, H_latent, W_latent]
config.model.guidance_scale = 5.0      # CFG 引导强度
config.model.student_sample_steps = 4  # 推理步数
config.model.net = Wan_1_3B_Config     # Student 网络配置
config.model.enable_preprocessors = True  # ⚠️ 用 VideoLoaderConfig 时必须 True
```

**Trainer 配置：**

```python
config.trainer.fsdp = True             # FSDP2 分布式（多卡必开）
config.trainer.ddp = False             # FSDP 时关闭 DDP
config.trainer.batch_size_global = 4   # 全局 batch size
config.trainer.max_iter = 4000         # 最大迭代数
config.trainer.save_ckpt_iter = 1000   # checkpoint 保存间隔
config.trainer.logging_iter = 50       # 日志间隔
config.trainer.validation_iter = 99999 # 验证间隔（设大数可关闭）
```

**DataLoader 配置：**

```python
config.dataloader_train = VideoLoaderConfig    # 读原始视频 (mp4)
config.dataloader_train.batch_size = 1         # ⚠️ 每卡 batch，必须设为 1
config.dataloader_train.datatags = ["WDS:/path/to/webdataset"]
```

> **⚠️ 两个常见陷阱：**
> 1. `config_mf_video.py` 的 `enable_preprocessors` 必须改为 `True`
> 2. `config_ladd.py` 未设 `dataloader_train.batch_size`，默认值很大会导致 OOM

### 3.3 命令行覆盖语法

```bash
# 格式: - key.subkey=value （注意 - 后有空格）
torchrun ... train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.max_iter=4000 \
      model.net.model_id_or_local_path=/path/to/model
```

---

## 4. 标准实验流程

### 4.1 原则：一个实验 = 一个 config

不要在命令行堆覆盖参数。为每个实验创建独立 config：

```
fastgen/configs/experiments/WanT2V/our/
├── exp01_dmd2_4gpu.py
├── exp02_meanflow_2gpu.py
├── exp03_fdistill_4gpu.py
├── exp04_ladd_4gpu.py
└── ...
```

### 4.2 实验 Config 模板

```python
# exp03_fdistill_4gpu.py — f-distill 4000 iter, 4-GPU FSDP
import fastgen.configs.experiments.WanT2V.config_fdistill as base

MODEL_PATH = "/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_PATH = "WDS:/data/datasets/OpenVid-1M/webdataset"

def create_config():
    config = base.create_config()

    # Model
    config.model.net.model_id_or_local_path = MODEL_PATH

    # Data
    config.dataloader_train.batch_size = 1
    config.dataloader_train.datatags = [DATA_PATH]

    # Trainer
    config.trainer.fsdp = True
    config.trainer.ddp = False
    config.trainer.batch_size_global = 4
    config.trainer.max_iter = 4000
    config.trainer.save_ckpt_iter = 1000
    config.trainer.logging_iter = 50
    config.trainer.validation_iter = 99999

    # Logging
    config.log_config.wandb_mode = "disabled"
    config.log_config.name = "exp03_fdistill_4gpu"

    return config
```

### 4.3 标准训练命令

```bash
#!/bin/bash
# run_exp.sh — 通用实验启动脚本
export CUDA_VISIBLE_DEVICES=1,4,6,7
export FASTGEN_OUTPUT_ROOT=/data/chenqingzhan/fastgen_output
export HF_HOME=/data/chenqingzhan/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/data/chenqingzhan/FastGen

cd /data/chenqingzhan/FastGen

# 用法: bash run_exp.sh <config_path> <num_gpus>
CONFIG=${1:?Usage: bash run_exp.sh <config> <num_gpus>}
NUM_GPUS=${2:-4}

torchrun --nproc_per_node=$NUM_GPUS --standalone train.py \
    --config=$CONFIG
```

启动方式：

```bash
# 干净、可复现、一行搞定
bash run_exp.sh fastgen/configs/experiments/WanT2V/our/exp03_fdistill_4gpu.py 4
```

---

## 5. 推理

### 5.1 FSDP Checkpoint 合并

训练产出的是 FSDP 分片 checkpoint，推理前需合并：

```python
import torch, torch.distributed.checkpoint as dcp, os
from torch.distributed.checkpoint import FileSystemReader

base = "/path/to/checkpoints"
iter_pad = "0004000"

# Load sharded net_model
reader = FileSystemReader(f"{base}/{iter_pad}.net_model")
md = reader.read_metadata()
net = {k: torch.empty(m.size) for k, m in md.state_dict_metadata.items() if hasattr(m, "size")}
dcp.load(net, storage_reader=reader)

# Load metadata
meta = torch.load(f"{base}/{iter_pad}.pth", map_location="cpu", weights_only=False)

# Save consolidated
torch.save({
    "model": {"net": net},
    "iteration": meta.get("iteration", 4000),
}, "consolidated_4000iter.pth")
```

> MeanFlow 需额外合并 `ema_1_model`：`"model": {"net": net, "ema_1": ema}`

### 5.2 运行推理

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) \
python scripts/inference/video_model_inference.py \
    --ckpt_path /path/to/consolidated.pth \
    --do_student_sampling True \
    --do_teacher_sampling False \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
    --prompt_file scripts/inference/prompts/eval_2prompts.txt \
    - trainer.ddp=False \
      trainer.seed=42 \
      model.student_sample_steps=4 \
      model.net.model_id_or_local_path=/path/to/model
```

注意：
- 推理用 **单 GPU**，无需 torchrun
- `student_sample_steps` 控制推理步数（1 或 4）
- MeanFlow 默认 1-step；DMD2/f-distill/LADD 可设 4-step

---

## 6. 已知问题与解决方案

| 问题 | 原因 | 解决 |
|------|------|------|
| LADD 3-GPU OOM | 未设 `batch_size=1`，R1 正则化增加显存 | 必须 4 GPU + `batch_size=1` |
| MeanFlow `TypeError: str` | `enable_preprocessors=False` | 改为 `True` |
| NCCL timeout | WandB 视频编码阻塞 >10 min | 设 `validation_iter=99999` 关闭 |
| CausVid/SF OOM | CausalWan 因果注意力额外 ~3 GB | 4×32GB 无解，需更大显存 |
| 8-bit AdamW 崩溃 | bitsandbytes 不兼容 FSDP2 DTensor | 只能用标准 AdamW fp32 |
| 推理 `KeyError: 'model'` | FSDP 分片 checkpoint 格式不同 | 先合并再推理（见 §5.1） |
| 单 GPU 推理 OOM | 模型 + Text Encoder + VAE 解码 | 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

## 7. 参考资料

- FastGen GitHub: https://github.com/NVlabs/FastGen
- DMD2 论文: https://arxiv.org/abs/2405.14867
- CausVid: https://github.com/tianweiy/CausVid
- Wan2.1: https://github.com/Wan-Video/Wan2.1
