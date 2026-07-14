# T0 项目分析:Wan2.1-T2V DMD2 渐进式少步蒸馏

- 写作日期:2026-07-06
- 作者:research planner agent(依 `research/planner_startprompt.md` 第五步)
- 事实来源:本地仓库文档 + 2026-07-06 对 `ust_ip:/data/chenqingzhan/FastGen` 的只读核查(命令与观察见附录 A)
- 状态:**待用户确认**。确认后本文件作为 T1-T4 调研任务书的方法描述与证据底本。

---

## 0. 当前状态校正(2026-07-06)

结论先行:本地文档与远端真实状态存在三类冲突。后续所有任务书、报告、对外口径一律以本节为准;旧文档在未更新前只能当历史背景。

### 0.1 校正一:mainline 归属——旧文档指向 Wan2.2 TI2V / WanI2V,实际主线已是 WanT2V(Wan2.1-T2V-1.3B)progressive 50 -> 8 -> 4

| 来源 | 声称的 mainline | 实际状态 |
|---|---|---|
| `CLAUDE.md` / `README.md` / `03-dmd-distillation/OVERVIEW.md` / `03-dmd-distillation/HANDOFF.md` | FastGen native `Wan2.2 TI2V 5B / WanI2V / DMD2`,commit `34f30e8` | **过期**。该线的远端活动止于 2026-05-10(最后日志 `wan22_dmd2_65f_cfg5_bs1_lr10x_fromscratch_20260504_g56.log`);且 5 月已推进到 `0013000` checkpoint 推理,超出 HANDOFF 记载(其记录止于 2026-04-27) |
| `experiments/configs/wan22_*.env` 三个配置 | 同上 Wan2.2 线 | 属旧线,不再是默认入口 |
| 远端 git(2026-07-06 核实) | — | branch `main`,commit `e66f6c6`,dirty 文件全部集中在 `fastgen/configs/experiments/WanT2V/` 与 `scripts/reports/` |
| 远端输出/近期报告(2026-06-06 至 2026-06-25) | — | 活跃主线是 **Wan2.1-T2V-1.3B(WanT2V)+ OpenVid-1M + DMD2 progressive `50 -> 8 -> 4`** |

远端 artifact 重建的时间线:

- 2026-04-23 ~ 2026-05-10:Wan2.2 TI2V 5B / WanI2V DMD2 线(HANDOFF 所载 + 5 月 `65f cfg5` 系列日志),之后停止。
- 2026-06-06:下载 `Wan2.1-T2V-1.3B-Diffusers`;首个 WanT2V run `wan21_t2v_dmd2_OpenVid` 落在旧输出根。
- 2026-06-08 ~ 06-15:4-step 基线 run `wan21_t2v_dmd2_OpenVid_global_8`(6000 iter);2026-06-15 完成 10-prompt checkpoint sweep(本地记录 teacher 50-step 平均 165.24s vs 4-step student 约 6.6s,数值引用前需重读远端 `metrics.csv`)。同期有 `wan_fdistill`、`wan_mf` 两条旁线 smoke。
- 2026-06-16 ~ 06-17:8-step 首训(`..._step8`,GPU 故障后 7 卡续训到 2530);本地 2026-06-17 索引记录其 `0002500` 推理明显模糊、差于 4-step `0001000`。**该 run 目录现已不在远端,以上结论 artifact 不可复核**(见 0.3)。
- 2026-06-18 ~ 06-19:两个 8-step 消融 run:`_step8_freq`(`student_update_freq=2`)与 `_step8_normalize`(均匀 `t_list`),均低 LR、短训;**结论未写入任何本地报告**。
- 2026-06-21 ~ 06-22:`_step8_lr_original`:LR 恢复 `1e-5`、batch 12、2500 iter。
- 2026-06-22 ~ 06-23:第一轮 4-step-from-8 `_step4_from_step8_2p5k`(LR `1e-5`、batch 12,init = `step8_lr_original/0002500`)。
- 2026-06-24 ~ 06-25:第二轮 4-step-from-8 `_step4_from_step8_8node`(LR `5e-6`、batch 16,同一 init);2026-06-25 对其 `0000500-0002500` 共 5 个 checkpoint 完成 eval-10 推理。此后远端无任何新活动。
- 2026-07-06(本次核查):8 张 GPU 全部空闲(各约 1 MiB 占用、0% 利用率),无训练/推理进程;git 状态与 2026-07-03 快照完全一致。

待用户拍板的整改项(T0 确认后执行,均为本地文档修改):

1. `README.md` / `CLAUDE.md` / `OVERVIEW.md` / `HANDOFF.md` 的 mainline 描述更新为 WanT2V 线,或在文件头部加带日期的"已过期,现状见 research/T0_project_analysis.md"注记。
2. `research/README.md` 的 2026-07-03 远端快照一节回填更正注记(输出根路径,见 0.2)。

### 0.2 校正二:输出根路径新旧差异

| 路径 | 角色 | 状态(2026-07-06 核实) |
|---|---|---|
| `/data/chenqingzhan/fastgen_output` | 旧输出根(4-5 月 wan22 线 + 2026-06-06 首个 WanT2V run) | 停更于 2026-06-06;其下 `fastgen/wan_dmd2/` 只有 `wan21_t2v_dmd2_OpenVid` 一个目录 |
| `/data/chenqingzhan/FastGen/FASTGEN_OUTPUT` | **当前实际输出根**(仓库内,`FASTGEN_OUTPUT_ROOT=FASTGEN_OUTPUT` 相对路径产物) | 2026-06 全部 WanT2V run、checkpoint、推理样例、multi-experiment 报告都在 `FASTGEN_OUTPUT/fastgen/wan_dmd2/` 下 |
| `/data/chenqingzhan/logs` | 旧集中日志目录 | 停更于 2026-06-06;6 月各 run 的日志在各自 run 目录的 `logs/` 子目录内 |

注意:`research/README.md`(2026-07-03)写的"Wan DMD2 output root = `/data/chenqingzhan/fastgen_output/fastgen/wan_dmd2`"指向的是停更目录,需回填更正。

### 0.3 校正三:8-step / 4-step-from-8 的结论与超参——哪些已核实,哪些待核实

**已核实**(直接读取各 run 输出目录下 FastGen dump 的 `config.yaml`,2026-07-06):

| run(`FASTGEN_OUTPUT/fastgen/wan_dmd2/` 下) | steps | LR(net/disc/fake 相同) | batch_size_global | student_update_freq | max_iter | t_list | init(pretrained_ckpt_path) |
|---|---|---|---|---|---|---|---|
| `..._global_8`(4-step 基线) | 4 | `1.25e-6` | 8 | 5 | 6000 | `[0.999, 0.937, 0.833, 0.624, 0.0]` | 无(teacher 权重起训) |
| `..._step8`(06-16 首训) | — | **目录已不存在,无法核实** | — | — | — | — | — |
| `..._step8_freq` | 8 | `1.25e-6` | 10 | **2** | 1000 | 插值 9 锚点(高噪密) | 无 |
| `..._step8_normalize` | 8 | `1.25e-6` | 10 | 5 | 1500 | **均匀** `[0.999, 0.875, ..., 0.125, 0.0]` | 无 |
| `..._step8_lr_original` | 8 | **`1e-5`** | **12** | 5 | 2500 | 插值 9 锚点 `[0.999, 0.968, 0.937, 0.885, 0.833, 0.729, 0.624, 0.312, 0.0]` | 无 |
| `..._step4_from_step8_2p5k`(第一轮 4-from-8) | 4 | **`1e-5`** | **12** | 5 | 2500 | 4-step 锚点 | `step8_lr_original/checkpoints/0002500`(仅 net,`key_map={"net":"net"}`) |
| `..._step4_from_step8_8node`(第二轮 4-from-8) | 4 | **`5e-6`** | **16** | 5 | 2500 | 4-step 锚点 | 同上 `step8_lr_original/checkpoints/0002500` |

其他已核实事实:

- "低 LR 时代"= `1.25e-6`(git 提交版 `config_dmd2.py` 的值,即 `1e-5 / 8`);工作区 dirty diff 已改回 `1e-5`,未提交。
- 全线 `guidance_scale = 5.0`(1.3B 模型上 CFG 教师前向可行;与 wan22 5B 线的 CFG OOM 问题无关)。
- `student_update_freq` 默认值 5 来自 `fastgen/configs/methods/config_dmd2.py:37`;因此 2500 iter ≈ 500 次 student 更新。
- 两轮 4-from-8 的 init checkpoint 相同且确认存在(`step8_lr_original/0002500`);训练脚本在启动前校验了前一阶段 `student_sample_steps=8`。
- `_step4_from_step8_8node` 的 5 个 checkpoint(`0000500`-`0002500`)的 eval-10 推理样例目录存在(2026-06-25 16:46-16:55 生成)。
- 4-step 基线 run 的 `0000500` checkpoint 确实缺失(checkpoints 目录从 `0001000` 开始),与 2026-06-17 实验索引记载一致。

**关键新发现(本地报告未记载,影响归因)**:第一轮(`2p5k`)与第二轮(`8node`)之间同时改了**三个因素**——LR `1e-5 -> 5e-6`、batch 12 -> 16、GPU 数 6 -> 8。2026-06-29 各报告把第二轮的质量改善归因于"提升有效 batch / 训练稳定性",**这个归因目前不成立**;只能说"配方 B 整体优于配方 A",LR 减半同样可能是主因。这直接违反"每次实验只改一个主要因素"的纪律,后续要么补受控实验,要么在对外表述中降级为配方级结论。

**待核实清单**:

| 条目 | 现状 | 核实方式 |
|---|---|---|
| 所有质量结论:"8-step lr_original 可用"、"第一轮 4-from-8 后期物理崩坏"、"第二轮 8node 可用且 0000500->0002500 随 iter 提升"、"4-step 基线 0001000 最佳" | 全部为肉眼判断,无任何量化指标 | VBench / FVD / CLIP 对齐 / 人评,任选其一先跑通;属 T3 之后的实验任务 |
| 06-16/17 首个 8-step run 的超参与"8-step 明显差"证据(含 13.16s) | run 目录已从远端消失,只剩本地 2026-06-17 索引的记载 | 用户 2026-07-06 拍板:是否有意清理不确认、不追查;一律按"本地报告记录、artifact 已不可复核"处理,不得作为已证明证据 |
| `_step8_freq` / `_step8_normalize` 两个消融的质量结论 | 推理样例在远端存在,但无本地结论记录 | 用户肉眼补看或纳入量化评估;它们恰好是 t_list 与更新频率两个 novelty 轴的既有数据点 |
| 实际 world_size / 梯度累积步数 | `config.yaml` 不含;"8node = 8 卡"来自命名与脚本注释 | 需要时从各 run `logs/` 的训练日志核实 |
| teacher 165.24s / student 6.59-6.63s(约 25x) | 本地索引记载,远端 `metrics.csv` 路径存在但本次未重读数值 | 引用前重读 `..._global_8/reports/eval_10prompts/metrics.csv` |
| `wan_fdistill` / `wan_mf` 旁线质量 | 仅有 run 目录,未评估 | 若要作为论文 baseline 需补推理评估 |

---

## 1. 方法的论文语言描述

一句话定位:**用 DMD2 式 distribution matching + 对抗目标,把一个 50-step 文生视频扩散 teacher(Wan2.1-T2V-1.3B)分阶段蒸馏成 8-step 中间 student、再蒸馏成 4-step 部署 student 的 progressive few-step distillation**。

**结构(什么冻结、什么可训、从哪分叉)**:

- Teacher:冻结的预训练 video diffusion transformer(1.3B 参数,latent 空间 `[16, 21, 60, 104]`,对应 832x480、81 帧视频),推理基线为 50-step 采样、classifier-free guidance 5.0。
- Student(generator):与 teacher 同架构;第一阶段从 teacher 权重起训,后一阶段(4-step)从前一阶段(8-step)student 的 checkpoint 初始化(仅生成网络权重,优化器与辅助网络重置)。Student 是 few-step 确定性(ODE 式)采样器,推理不用 CFG。
- 辅助可训网络两个:(a) fake score network,与 teacher 同架构,在线拟合 student 生成分布的 score;(b) 判别器 = **冻结 teacher backbone 中间层(第 15/22/29 层)特征上的可训 multiscale MLP 头**(`multiscale_down_mlp_large`)。【更正 2026-07-06:经远端 `fastgen/methods/distribution_matching/dmd2.py` 代码核实,判别特征由冻结 teacher 的 forward 提取(fake 分支复用 VSD 的同一次 teacher forward,real 分支 `return_features_early` 截断),先前"生成器 backbone"的推测不成立,属 LADD / teacher-feature 谱系;另确认 `gan_use_same_t_noise=True` 时 real/fake 共享同一 t 与同一 ε(FastGen 方法基类默认 False,但上游 WanT2V 全部 15 个公开实验配置出厂即 True——是承袭的出厂值,非我方主动设计;2026-07-14 依 T3 §6.1 更正),FastGen 内置近似 R1(`gan_r1_reg_weight`)但我们全部 run 为 0.0 未启用。】

**信号(什么信息、在线还是离线)**:

- Distribution matching 梯度:teacher score(带 CFG=5)与 fake score 之差,推动 student 输出分布靠近 teacher 分布(reverse-KL / VSD 式,在线)。
- 对抗信号:判别器区分真实视频 latent 与 student 生成 latent(二者加同一噪声,`gan_use_same_t_noise=True`),生成端权重 0.03。
- Fake score 以 x0-prediction 目标持续拟合 student 当前分布;two-time-scale 更新:student 每 5 次迭代更新一次,fake score 与判别器每次迭代更新(`student_update_freq=5`)。

**约束/目标函数与关键自由度**:

- 少步 student 的采样轨迹由离散时间锚点 `t_list` 固定(4-step:`[0.999, 0.937, 0.833, 0.624, 0.0]`;8-step 曾试插值/均匀两种 9 锚点方案),训练时的 t 采样为 shifted 分布(`min_t=0.001, max_t=0.999`)。
- 步数压缩分阶段:`50 -> 8 -> 4`,以"前一阶段最优 checkpoint 初始化下一阶段"为衔接机制。
- 实验证明的关键自由度:学习率(`1.25e-6` vs `1e-5` vs `5e-6`)、有效 batch(8/10/12/16)、`t_list` 形状、`student_update_freq`(5 vs 2)、checkpoint 选择(早停)。

工程名词到论文语言的对照:`FastGen` = 训练框架;`WanT2V` = Wan2.1-T2V-1.3B 文生视频模型;`OpenVid` = OpenVid-1M 视频-文本数据集;`lr_original` = 恢复 `1e-5` 学习率的 8-step run;`8node` = 8 GPU、batch 16、LR `5e-6` 的第二轮 4-step 阶段。

## 2. 证据分级

**A. 已证明(带数字,远端 artifact 支撑)**:

- 各 run 超参已核实(0.3 节表);`student_update_freq=5` 下 2500 iter ≈ 500 次 student 更新,分摊到 8 个时间锚点后每锚点监督量明显少于 4-step。
- 训练闭环健康:0.3 表中当前可复核的主要 run(4-step 基线、`step8_freq` / `step8_normalize` / `step8_lr_original`、两轮 4-from-8)均正常产出 checkpoint 与推理样例目录(存在性已核实,内容质量另评)。

**B. 报告记录但未复核或未量化(引用前按备注处理)**:

- 速度:teacher 50-step 平均 165.24s/视频,4-step student 平均 6.59-6.63s,约 25x(2026-06-15,10 prompts)——**本地报告记录且远端 `metrics.csv` 路径存在,引用前需重读确认**。25x 与"teacher 带 CFG 双前向(等效 100 NFE)vs student 4 NFE 无 CFG"自洽,属合理性检查而非复核。
- 06-16/17 首个 8-step run:`0002500` 平均采样 13.16s(约为 4-step 的 2 倍)、画面明显模糊差于 4-step `0001000`——**本地 2026-06-17 索引记录,远端 run 目录已不存在,artifact 不可复核**;除非重新定位到对应 metrics/样例文件,不得作为已证明证据引用。

- "8-step 低 LR 模糊、恢复 `1e-5` + batch 12 后可用"——LR/batch 值已核实,**质量判断未量化**。
- "第一轮 4-from-8 早期 `0000500` 最好、后期物理崩坏;第二轮 8node 可用且随 iter 提升"——同上;且两轮之间三因素同时变化,**改善归因不成立**(0.3 节)。
- "4-step 基线最佳 checkpoint 为 `0001000`"——肉眼结论。

**C. 进行中 / 有数据但无结论**:

- `_step8_freq`(更新频率消融)与 `_step8_normalize`(均匀 `t_list` 消融)已跑、有推理样例,无结论记录。
- 当前远端无任何 run 在跑(2026-07-06,8 卡空闲);自 2026-06-25 起项目处于停机分析期。

**D. 已过期或冲突**:

- Wan2.2 TI2V 5B / WanI2V 线(2026-04/05)整条为历史背景;其"no-CFG 才能训"的结论是 5B 模型 + 该硬件的特定结论,不适用于当前 1.3B 线(`guidance_scale=5.0` 全程可训)。
- 2026-06-17 索引"当前 8-step 不适合作为 4-step 初始化"与 2026-06-29 报告"4-from-8 可用"——按时间线不矛盾(前者指低 LR 首训 run,后者基于 `lr_original` run),但首训 run 目录已消失,引用时需注明。
- 本地 mainline 文档与输出根路径(0.1、0.2 节)。

## 3. Novelty 候选轴(每轴一句可被文献检验的主张;预期多数会被部分击破,属流程正常)

- **轴 A(step-schedule / t_list 设计)**:"multi-step DMD2 视频 student 的质量对离散时间锚点 `t_list` 的形状高度敏感(高噪密集插值 vs 均匀),且现有文献缺少针对 distribution-matching 目标的少步 schedule 设计原则。" 检验点:Phased Consistency Models、TDM(arXiv 2503.06674)、DMD2 原文的 multi-step 设置是否已系统处理;我们手上已有 freq/normalize 两个消融数据点可支撑。
- **轴 B(progressive DMD2 for video)**:"以 DMD2 为目标函数、用中间 8-step student 初始化 4-step student 的分阶段视频蒸馏配方(而非 Lightning 系列的 progressive adversarial 或一次性 50->4)在公开视频模型上未见完整先例。" 检验点:SDXL/AnimateDiff-Lightning(对抗目标而非 DMD2)、TDM、ADM(arXiv 2507.18569)、Wan 生态各蒸馏产物是否已做同构方案。
- **轴 C(少步视频蒸馏的稳定性归因)**:"视频 few-step DMD2 的质量瓶颈主要在优化稳定性(LR、有效 batch、每锚点有效更新量),存在可复现的失败-修复配方。" 现状:证据链有三因素混淆(0.3 节),**该轴要成立必须先补受控实验**;否则降级为经验报告素材,不作为论文主张。

## 4. 最近邻工作(凭记忆列出,**全部待 T1/T2 实时检索核实**,禁止直接引用)

| 类别 | 工作 | 为何是近邻 | 裁决归属 |
|---|---|---|---|
| 渐进式步数压缩 | Progressive Distillation(ICLR 2022)、TRACT | 阶段压缩思想的出处 | T1 |
| Distribution matching | DMD(2023)、DMD2(2024)、TDM(2503.06674)、ADM(2507.18569) | 目标函数同族;TDM/ADM 直接触及 multi-step 与 mode collapse | T2/T3 |
| Progressive + adversarial | SDXL-Lightning、AnimateDiff-Lightning、ADD | 与轴 B 竞争最直接 | T2/T3 |
| 视频少步蒸馏 | T2V-Turbo、DOLLAR、VideoLCM、Motion Consistency Model、Phased CM | 视频侧证据与失败模式 | T1/T2 |
| **Wan 生态直接竞品(最高风险)** | 社区/官方 Wan2.1-T2V 少步蒸馏(如 CausVid-Wan、Self-Forcing、FastVideo、lightx2v 等,名称与状态均待核实) | 若已有公开的 Wan2.1 1.3B 少步蒸馏 recipe/checkpoint,轴 B 将被显著收窄 | T3 重点对抗检索 |

## 5. 对 T1-T4 任务书的靶子(T0 确认后出)

- T1 领域主线谱系:视频扩散 few-step 蒸馏的路线图,重点回答"progressive 多阶段在视频上有哪些先例"。
- T2 方法组件近邻:DMD/DMD2 家族的 multi-step 处理、step schedule、video/flow-aware discriminator 设计盘点。
- T3 novelty 对抗核实:对轴 A/B/C 及 T1/T2 幸存主张逐条裁决;Wan 生态竞品实时检索;产出"能说/不能说"红线。
- T4 投稿策略:目标 venue 与截稿日期(实时查官方来源)、证据缺口分级(量化指标缺失是当前最大缺口)。

## 6. 本 T0 的边界

- 未启动任何训练/推理,未修改/删除远端任何文件;远端操作仅只读(附录 A)。
- 未全量扫描 `archive/`;Wan2.2 旧线仅作为历史背景引用。
- 第 4 节近邻工作未经实时检索,不构成结论。

---

## 附录 A:远端只读核查记录(2026-07-06)

```bash
# git 状态:branch main, commit e66f6c6, dirty 文件与 2026-07-03 快照一致
ssh ust_ip "hostname; cd /data/chenqingzhan/FastGen && git branch --show-current && git rev-parse --short HEAD && git status --short | head -80"

# 近期文件活动:止于 2026-06-25(build_wan_dmd2_multi_experiment_report.py 等)
ssh ust_ip "cd /data/chenqingzhan/FastGen && find fastgen/configs/experiments/WanT2V scripts/reports experiment -maxdepth 4 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -80"

# 旧输出根/旧日志目录:均停更于 2026-06-06
ssh ust_ip "ls -lt /data/chenqingzhan/logs | head -30; find /data/chenqingzhan/fastgen_output -maxdepth 4 -type d -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -80"

# 当前实际输出根:全部 6 月 WanT2V run 在 FastGen/FASTGEN_OUTPUT 下,最后活动 2026-06-25 16:55
ssh ust_ip "find /data/chenqingzhan/FastGen/FASTGEN_OUTPUT -maxdepth 4 -type d -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort -r | head -60"

# GPU / 进程:8 卡空闲,无训练进程
ssh ust_ip "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | head -12"

# 配置核实:git diff(LR 1.25e-6 -> 1e-5 等)、三个 config、两个启动脚本
ssh ust_ip "cd /data/chenqingzhan/FastGen && git diff -- fastgen/configs/experiments/WanT2V/"
ssh ust_ip "cat /data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/{config_dmd2.py,config_dmd2_smoke.py,config_dmd2_step8_2k.py,config_dmd2_step4_from_step8_2p5k.py,run_train_dmd2_step4_from_step8_2p5k.sh,run_infer_dmd2_step4_from_step8_8node_eval10.sh}"

# 各 run 实际超参:读取每个 run 输出目录下 dump 的 config.yaml + checkpoint 清单
ssh ust_ip 'BASE=/data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2; for d in <run dirs>; do grep -E "student_sample_steps:|batch_size_global:|max_iter:|student_update_freq:|guidance_scale:|pretrained_ckpt_path:" "$BASE/$d/config.yaml"; grep -A10 "t_list:" "$BASE/$d/config.yaml"; grep "lr:" "$BASE/$d/config.yaml"; ls "$BASE/$d/checkpoints"; done'

# DMD2 默认 student_update_freq=5
ssh ust_ip "grep -n 'student_update_freq' /data/chenqingzhan/FastGen/fastgen/configs/methods/config_dmd2.py"
```

关键观察已内联于正文 0.1-0.3 节。
