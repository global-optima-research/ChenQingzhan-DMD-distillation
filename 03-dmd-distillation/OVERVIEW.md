# PVTT DMD Distillation Overview

这份文档是 `03-dmd-distillation/` 的当前主入口，作为本仓库在 Task 3 阶段的单一事实来源。

历史记录、个人看板、组会纪要仍然保留，但默认只作为参考，不再作为当前状态的主文档。

## 1. 当前任务是什么

当前处于 **Phase 0：开源方案复现**。

目标不是立即做 PVTT 业务适配，而是先基于公开代码跑通 Wan 蒸馏训练和推理流程，明确哪套框架适合作为后续开发底座。

当前聚焦：
- 框架：`FastGen`
- 基座模型：`Wan2.1-T2V-1.3B`
- 数据：`OpenVid-1M -> WebDataset(mp4 + txt)`
- 方法对比：`DMD2 / ECT / CD / f-distill / LADD`

## 2. 整体路线

### Phase 0：复现验证

1. 搭建 FastGen 运行环境。
2. 下载 Wan2.1 1.3B 权重并验证 teacher 50-step 推理。
3. 准备统一训练数据：
   `OpenVid-1M -> 过滤 -> WebDataset tar shards`
4. 在单卡模式下复现多种蒸馏方法。
5. 记录训练配置、显存、Loss 曲线、样本质量。
6. 形成复现报告，选定后续主 codebase。

### Phase 1：PVTT 任务适配

1. 等 Task 2 交付 PVTT teacher model。
2. 把蒸馏框架改成支持：
   `source video + reference image + mask`
3. 做渐进式蒸馏：
   `50 -> 16 -> 8 -> 4`
4. 增加视频编辑任务专用损失：
   `temporal / background / identity / adversarial`
5. 训练最终 4-step student model。

## 3. 当前预期产物

- Phase 0：
  - 训练可跑通的 FastGen 单卡脚本
  - OpenVid-1M 的 WebDataset 数据
  - 多方法复现结果对比
  - 复现报告

- Phase 1：
  - PVTT 适配后的蒸馏训练代码
  - 渐进式 student checkpoints
  - 4-step 推理模型

## 4. 当前服务器信息

仓库内已明确的信息：

- Server IP: `111.17.197.107`
- GPU: `8x RTX 5090 32GB`
- CPU: `384 cores`
- RAM: `1TB`
- Disk: `21TB total`, 约 `10TB free`
- OS: `Ubuntu`, Linux `5.15.0`
- Python env: `conda env fastgen`, Python `3.12.12`
- PyTorch: `2.10.0+cu128`
- CUDA Toolkit: `12.8`

当前默认运行模式：
- 单卡
- `CUDA_VISIBLE_DEVICES=0`

仓库中记录了服务端路径约定：
- FastGen：`/data/chenqingzhan/FastGen`
- 输出目录：`/data/chenqingzhan/fastgen_output`
- HuggingFace 缓存：`/data/chenqingzhan/.cache/huggingface`
- OpenVid 数据：`/data/datasets/OpenVid-1M`

注意：
- 仓库里只有服务器 IP、环境和目录约定。
- **没有完整记录 ssh 用户名、端口、密钥位置、标准登录命令。**
- 如果后续要让别人无歧义接手，应该补一份最小化登录说明。

## 5. 当前推荐执行顺序

### A. 环境与模型

- `03-dmd-distillation/setup_server.sh`
- `03-dmd-distillation/scripts/download_model.sh`
- `03-dmd-distillation/scripts/run_inference.sh`

目的：
- 把环境装好
- 下载 Wan2.1 1.3B
- 先确认 50-step teacher inference 正常

### B. 数据准备

- `03-dmd-distillation/scripts/download_openvid.sh`
- `03-dmd-distillation/scripts/convert_to_webdataset.py`
- `03-dmd-distillation/scripts/prepare_training_data.sh`

目标输出：
- `/data/datasets/OpenVid-1M/webdataset/shard-xxxxxx.tar`

### C. 方法复现

- `run_dmd2_single_gpu.sh`
- `run_ect_single_gpu.sh`
- `run_cd_single_gpu.sh`
- `run_fdistill_single_gpu.sh`
- `run_ladd_single_gpu.sh`

说明：
- `run_meanflow_single_gpu.sh` 需要 latent shards，不走当前这套 mp4+txt 数据准备流程。

## 6. 当前脚本状态

### 可以保留并继续完善

- `convert_to_webdataset.py`
- `download_openvid.sh`
- `prepare_training_data.sh`
- `run_inference.sh`
- `config_cm_ct.py`
- `config_cm_cd.py`

### 当前有明显对接问题，需要修

- `run_dmd2_single_gpu.sh`
  - 缺少数据路径覆盖
  - 缺少 `model.net.model_id_or_local_path` 覆盖

- `run_ect_single_gpu.sh`
- `run_cd_single_gpu.sh`
- `run_fdistill_single_gpu.sh`
- `run_ladd_single_gpu.sh`
  - 默认数据路径仍指向 `/data/chenqingzhan/training_data/video_shards`
  - 与当前 OpenVid 输出目录 `/data/datasets/OpenVid-1M/webdataset` 不一致

- `run_meanflow_single_gpu.sh`
  - 需要 latent 数据
  - 当前仓库没有提供 latent 预处理流程

## 7. 当前仓库文档建议怎么理解

### 主文档

- `03-dmd-distillation/OVERVIEW.md`

### 技术综述

- `03-dmd-distillation/README.md`

### 历史记录与参考

- `03-dmd-distillation/progress.md`
- `03-dmd-distillation/meeting-2026-03-06.md`
- `03-dmd-distillation/ChenHingChinReadMe.md`
- `TASK-ASSIGNMENT.md`
- 根目录 `README.md`

## 8. 当前明确的待办

1. 统一所有训练脚本的数据路径和参数风格。
2. 把 DMD2 脚本补成真正可运行的主入口。
3. 明确哪些脚本是单卡可跑、哪些只是占位。
4. 增加最小化服务器登录说明。
5. 决定是否把重复文档移到 `docs/archive/`，减少噪音。

## 9. 清理建议

如果下一步做仓库瘦身，建议按这个原则：

- 保留：
  - 当前主入口文档
  - 真正可运行的脚本
  - 自定义 FastGen 配置

- 下沉到参考区：
  - 个人看板
  - 组会纪要
  - 大段调研性质文档

- 删除或归档前先确认：
  - 是否仍有人依赖这些文档做周报或交接
  - 是否已经把关键信息迁移到主文档
