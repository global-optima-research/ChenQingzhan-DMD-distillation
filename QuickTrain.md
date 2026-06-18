# QuickTrain：FastGen Wan2.1 快速训练任务

更新日期：`2026-06-05`。

本文档记录当前阶段“回到 Wan2.1，并准备搭建 FastGen 快速训练框架”的初步任务目标与入口流程调研结果。本文只记录目标和已观察到的事实，暂不提出框架设计、实验矩阵或启动脚本。

## 当前阶段

- 阶段：任务定义与 FastGen 训练入口调研。
- 目标模型族：FastGen 下的 `Wan2.1-T2V-1.3B`。
- 当前意图：先弄清 FastGen 现有训练入口、配置入口、数据入口和方法入口，后续再决定 QuickTrain 应该怎样组织。
- 本阶段不做：不提出架构方案，不设计实验矩阵，不新建启动自动化，不启动训练。

## 初步目标

- 后续搭建一套基于 FastGen 的 QuickTrain 工作流，避免继续堆积一次性 shell 命令。
- 第一阶段先聚焦 `Wan2.1-T2V-1.3B`，因为它已经在 Phase 0 FastGen 实验中跑过，资源压力低于后续 Wan2.2 5B 路线。
- 保留当前仓库的实验管理原则：一个有意义的 run 应该能对应到清晰的 config、远端 log、输出路径和简短结果记录。
- 优先复用 FastGen 已验证机制：Python config、命令行 override、FSDP、WebDataset 输入、现有 checkpoint 布局。
- 当前阶段只记录事实：先理解 FastGen 当前代码怎样工作，再讨论 QuickTrain 结构。

## 已检查资料

本地仓库资料：

- `03-dmd-distillation/FastGen_Guide.md`
- `archive/reports/phase0-dmd-distillation/Training_Report.md`
- `archive/reports/phase0-dmd-distillation/experiment-method-comparison-2026-03-30.md`
- `03-dmd-distillation/scripts/archive/ip-2026-spring/training/run_dmd2_fsdp.sh`
- `03-dmd-distillation/scripts/archive/ip-2026-spring/training/run_meanflow.sh`

服务器 `ust_ip` 上只读检查过的 FastGen 文件：

- `/data/chenqingzhan/FastGen/train.py`
- `/data/chenqingzhan/FastGen/fastgen/utils/scripts.py`
- `/data/chenqingzhan/FastGen/fastgen/trainer.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/config.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/data.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/net.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/config_utils.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/methods/config_dmd2.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_dmd2.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_fdistill.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_mf.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_mf_video.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/our/_common.py`
- `/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/our/exp01_dmd2_4gpu.py`
- `/data/chenqingzhan/FastGen/fastgen/methods/model.py`
- `/data/chenqingzhan/FastGen/fastgen/methods/distribution_matching/dmd2.py`
- `/data/chenqingzhan/FastGen/fastgen/datasets/wds_dataloaders.py`
- `/data/chenqingzhan/FastGen/run_exp.sh`
- `/data/chenqingzhan/FastGen/scripts/phase05_training/run_dmd2_fsdp.sh`

## 服务器事实

- FastGen 根目录：`/data/chenqingzhan/FastGen`
- 历史脚本使用的 conda 环境：`/data/chenqingzhan/miniconda3/envs/fastgen`
- 历史脚本使用的输出根目录：`/data/chenqingzhan/fastgen_output`
- Hugging Face home：`/data/chenqingzhan/.cache/huggingface`
- OpenVid WDS 数据路径：`/data/datasets/OpenVid-1M/webdataset`
- `2026-06-05` 只读检查到 OpenVid WDS 目录下有 `22` 个 tar shard。
- 当前观察到的 Wan2.1 T2V 1.3B hub cache 路径：
  `/data/chenqingzhan/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers`
- 历史脚本仍引用旧的非 hub 路径：
  `/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers`
  该路径在本次只读检查中不存在，应视为过期路径，后续使用前必须修正或重新验证。

## FastGen 训练入口流程

观察到的主流程：

1. `train.py` 是统一训练入口。
2. `train.py` 调用 `fastgen.utils.scripts.parse_args()` 读取 `--config`、`--log_level`、`--dryrun` 和尾部 override 参数。
3. `fastgen.utils.scripts.setup()` 导入 Python config，应用 Hydra 风格命令行 override，写出 `config.yaml`，在启用 `trainer.ddp` 或 `trainer.fsdp` 时初始化分布式训练，并根据 `trainer.batch_size_global` 调整梯度累积轮数。
4. `train.py` 使用 `config.model` 实例化 `config.model_class`。
5. `Trainer(config)` 初始化 callbacks 和 checkpointer。
6. `Trainer.run(model)` 依次执行 `model.on_train_begin()`、DDP/FSDP 包装、optimizer 初始化、checkpoint resume、dataloader 实例化，然后进入训练循环。
7. 每个 iteration 从 dataloader 取数据，经 trainer 预处理成 latent/text embedding，调用 `model_ddp.single_train_step(data, iteration)`，随后 backward、optimizer step、callback logging，并按 `trainer.save_ckpt_iter` 保存 checkpoint。

命令行 override 事实：

- FastGen config 是 Python 文件，不是 YAML。
- override 参数前必须有一个单独的 `-` 分隔符。
- 命令形态示例：

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
  - trainer.ddp=False \
    trainer.fsdp=True \
    trainer.batch_size_global=4 \
    model.net.model_id_or_local_path=/path/to/model \
    dataloader_train.datatags='["WDS:/path/to/webdataset"]'
```

## Wan2.1 配置入口

Wan2.1 T2V 相关 config 位于：

`/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/`

相关入口：

- `config_dmd2.py`：Wan2.1 T2V 1.3B DMD2，官方风格视频蒸馏配置。
- `config_fdistill.py`：Wan2.1 T2V 1.3B f-distill 配置。
- `config_ladd.py`：Wan2.1 T2V 1.3B LADD 配置。
- `config_mf.py`：MeanFlow 配置，默认使用 latent loader。
- `config_mf_video.py`：MeanFlow 原始视频入口，使用 raw video loader，并启用 preprocessors。
- `our/_common.py`：历史实验共享 override，包括模型路径、数据路径、FSDP、batch、logging 和关闭 validation。
- `our/exp01_dmd2_4gpu.py`：历史 DMD2 4-GPU 实验配置，包装 `config_dmd2.py`。

`config_dmd2.py` 观察到的默认值：

- Model net：`Wan_1_3B_Config`
- `net.py` 中默认模型 ID：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Precision：`bfloat16`
- latent input shape：`[16, 21, 60, 104]`
- 由 latent shape 推导的视频分辨率：`832 x 480`
- 由 latent 时间长度推导的序列长度：`81`
- Student sampling：`4` steps，`ode`
- Timestep list：`[0.999, 0.937, 0.833, 0.624, 0.0]`
- Guidance scale：`5.0`
- Dataloader：`VideoLoaderConfig`
- 单 GPU dataloader batch size：`1`
- 官方默认 global batch size：`64`
- 默认 max iterations：`6000`
- 默认 save interval：`500`
- Log group：`wan_dmd2`

## 数据入口

Wan2.1 T2V 训练使用 `VideoLoaderConfig`，底层是 `VideoWDSLoader`。

观察到的默认映射：

- `datatags`：`WDS:<path>` 列表
- `key_map`：`{"real": "mp4", "condition": "txt"}`
- `presets_map`：`{"neg_condition": "neg_prompt_wan"}`
- `sequence_length`：`81`
- `img_size`：`(832, 480)`
- `num_workers`：`2`

Trainer 预处理事实：

- 如果输入还是视频空间，trainer 会通过 `model.net.vae` 把 raw video 编码为 latent。
- 如果文本条件还是字符串列表，trainer 会通过 `model.net.text_encoder` 编码 prompt。
- 对 T2V，`_prepare_training_data()` 直接把 `condition` 和 `neg_condition` 交给方法实现。
- I2V 和 vid2vid 有额外条件分支，但当前 Wan2.1 T2V QuickTrain 初始目标暂不涉及这些分支。

## DMD2 方法流程

观察到的 DMD2 实现：

- 基类：`DMD2Model(FastGenModel)`
- 模型组件：student `net`、冻结 `teacher`、可训练 `fake_score`、可选可训练 `discriminator`。
- `teacher` 在提供 `config.teacher` 时从该配置构建，否则从 `config.net` 构建。
- 对 Wan2.1 DMD2 config，teacher 因此使用 `Wan_1_3B_Config`。
- 构建时会执行 `teacher.eval().requires_grad_(False)`。
- `student_update_freq` 在 `fastgen/configs/methods/config_dmd2.py` 中默认是 `5`。
- 当 `iteration % student_update_freq == 0` 时更新 student。
- 其他 iteration 更新 `fake_score` 和 `discriminator`。
- 当 `guidance_scale` 不是 `None` 时，student update 会额外做一次 negative-condition teacher forward 来执行 CFG。
- Checkpoint 不只包含 student net，还会包含 DMD2 辅助状态，例如 fake-score、discriminator 及其 optimizer。

## 既有 Wan2.1 结论

来自本地 Phase 0 报告：

- 已用硬件：8 x RTX 5090 32GB。
- Base model：`Wan2.1-T2V-1.3B-Diffusers`。
- Dataset：OpenVid-1M，21,133 samples，22 WDS shards。
- 视频规格：81 frames，832 x 480，latent shape `[16, 21, 60, 104]`。
- ECT 和 CD 在既有实验中不适合作为视频蒸馏主线。
- DMD2 4-GPU FSDP 完成过 2000 iterations，记录显存约 22-27 GB/GPU，速度约 16 s/iter。
- f-distill 完成过 4000 iterations，记录显存约 24-28 GB/GPU，速度约 20 s/iter。
- MeanFlow 可在 2 GPU 上运行，但 6000 iterations 仍偏 undertrained；既有记录认为 MeanFlow 需要远高于当前小预算的训练量。
- LADD 在 4 GPU 上显存很紧；CausVid 和 Self-Forcing 在 4 x 32GB 上 OOM，原因是 causal attention 额外增加显存压力。
- Text encoder 复制到每张 GPU 是既有记录中反复出现的显存瓶颈。

## 已知陷阱

- 不要直接信任旧的非 hub Wan2.1 模型路径；`2026-06-05` 观察到的可见缓存路径在 `HF_HOME/hub/` 下。
- 不要同时设置 `trainer.ddp=True` 和 `trainer.fsdp=True`；FastGen 会 assert。
- FSDP 路线应使用 `trainer.ddp=False` 和 `trainer.fsdp=True`。
- `trainer.batch_size_global` 会在 `setup()` 中反推并设置 `trainer.grad_accum_rounds`。
- `config_mf.py` 使用 `VideoLatentLoaderConfig`；`config_mf_video.py` 才是 raw video MeanFlow 入口，并且 `enable_preprocessors=True`。
- `VideoLoaderConfig` 期望 WDS 样本包含 `mp4` 和 `txt`。
- WandB 视频/媒体 logging 以前触发过长时间阻塞或 NCCL timeout；历史配置常用 `trainer.validation_iter=99999` 或 `log_config.wandb_mode=disabled` 降低风险。
- DMD2 有四个模型侧组件；既有记录显示单 GPU Wan2.1 DMD2 会 OOM。

## 后续待复核事实

- hub cache 路径是否可以稳定传给 `model.net.model_id_or_local_path`，还是应该依赖默认 model ID + `HF_HOME`。
- 当前服务器 FastGen patch 是否会影响 Wan2.1 的 media logging 行为。
- 历史 Wan2.1 checkpoint 目前还保留哪些，是否需要纳入 active experiment layer 索引。
- 后续是否要扩展当前仓库的 `experiments/bin/run_remote.sh` 来承载 Wan2.1 QuickTrain。
- QuickTrain 首个可执行目标应从 DMD2、f-distill、MeanFlow 还是 smoke-only 抽象开始。本文档故意不做决定。
