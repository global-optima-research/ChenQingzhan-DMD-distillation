# T1 任务书:视频扩散少步蒸馏领域主线谱系

- 生成:2026-07-06,planner agent,依 `research/workflow.md` 与 `research/task_brief_template.md`
- 状态:**最终版,可直接分发**
- 分发方式:把下方「任务书正文」整节粘贴给一个**新的**内容 agent 会话;完成后把它的 ≤10 行执行总结粘回 planner 会话验收
- 引用路径核实:`research/T0_project_analysis.md`、`reports/2026-06-25-progressive-distillation-paper-list.md` 均已于 2026-07-06 确认存在

---

## 任务书正文(从此行以下全部粘贴)

你是一个强力 research agent,本次任务是 **T1 文献调研:视频扩散少步蒸馏的领域主线谱系**。这是纯调研任务,不是工程任务。今天是 2026-07-06,所有"近两年/最新"判断以此为基准。

工作目录:

/Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation

## 角色与边界(先读这一节)

- 本目录的 CLAUDE.md 是为工程 agent 写的,本次会话**忽略其中的角色设定和上下文加载规则**(读序、远端检查、实验流程等一概不适用)。
- 不要读代码目录,不要 ssh 任何远端机器,不要跑训练/评估,不要修改 `research/` 以外的任何文件。
- 本地**只读以下 2 个文件**,其余背景本 prompt 已给足,不要再读别的:
  1. `research/T0_project_analysis.md`(重点第 1-4 节:方法、证据分级、novelty 轴、近邻候选)
  2. `reports/2026-06-25-progressive-distillation-paper-list.md`(已有 18 篇清单,用于避免重复劳动)
- 调研用可靠来源:论文原文、arXiv、OpenReview、CVF、官方代码库。**近两年的工作必须实时检索,不要凭记忆**(覆盖 CVPR / ICCV / ECCV / NeurIPS / ICML / ICLR / SIGGRAPH(Asia) / AAAI / ACM MM 的 2024-2026 与 arXiv 2024-07 至今)。
- 可并行的检索尽量并行。控制聚焦:清单表 10-20 篇(与已有 18 篇重叠的条目只补新信息,不重写)、精读 3-5 篇,不做百科式综述。

## 项目背景(一段话)

我们在 FastGen 框架上研究文生视频扩散模型的少步蒸馏加速:把 Wan2.1-T2V-1.3B 的 50-step teacher 用 DMD2 目标分阶段蒸馏为 8-step 中间 student、再到 4-step 部署 student。工程闭环已通、有多组已完成 run,现在要推进到科研级:厘清领域谱系、确定 novelty 边界、升级评估协议,服务后续论文投稿(目标 venue 与日期由 T4 任务另行确定)。本 T1 是四个调研任务(T1 谱系 / T2 组件近邻 / T3 novelty 对抗核实 / T4 投稿策略)中的第一个。

## 我们的方法(文献对比用的准确描述,来自 T0)

一句话:用 DMD2 式 distribution matching + 对抗目标,把 50-step 文生视频扩散 teacher(Wan2.1-T2V-1.3B,latent `[16,21,60,104]`,832x480、81 帧)分阶段蒸馏成 8-step 中间 student、再蒸馏成 4-step 部署 student 的 progressive few-step distillation。

- 结构:teacher 冻结;student 与 teacher 同架构、可训,few-step 确定性(ODE 式)采样,推理不用 CFG;4-step 阶段从 8-step 最优 checkpoint 初始化(仅生成网络权重,优化器与辅助网络重置)。辅助网络两个:在线拟合 student 分布的 fake score network(同架构),以及基于生成器 backbone 中间层(第 15/22/29 层)特征的 multiscale MLP 判别器。
- 信号:(a) distribution matching 梯度 = teacher score(CFG=5)− fake score(reverse-KL / VSD 式);(b) 对抗信号:判别器区分真实视频 latent 与 student 生成 latent(同 timestep 同噪声),生成端权重 0.03;(c) fake score 以 x0-prediction 在线拟合 student 分布;two-time-scale:student 每 5 次迭代更新一次,fake score 与判别器每次迭代更新。
- 约束/自由度:student 采样轨迹由离散时间锚点 `t_list` 固定(4-step:`[0.999, 0.937, 0.833, 0.624, 0.0]`;8-step 试过"高噪密集插值"与"均匀"两种 9 锚点方案);训练时 t 采样为 shifted 分布。实验证明的关键自由度:学习率、有效 batch、`t_list` 形状、student 更新频率、checkpoint 选择(早停)。

## 实验现状(结论可直接引用,不必读实验报告)

- 已证明(远端 artifact 核实,2026-07-06):各 run 超参已逐一核实——4-step 基线(LR `1.25e-6`、batch 8、6000 iter);8-step 两个消融:freq(`student_update_freq=2`)与 normalize(均匀 `t_list`),均 LR `1.25e-6`、batch 10、短训;8-step lr_original(LR `1e-5`、batch 12、2500 iter);4-step-from-8 两轮:第一轮(LR `1e-5`、batch 12)、第二轮(LR `5e-6`、batch 16、8 卡),init 同为 8-step lr_original 的 `0002500`。`student_update_freq=5` 下 2500 iter ≈ 500 次 student 更新,8-step 每锚点有效监督明显少于 4-step。
- 报告记录但未量化(肉眼结论):低 LR 8-step 模糊;lr_original 后 8-step 进入可用;第一轮 4-from-8 早期 checkpoint 最好、后期物理规则崩坏;第二轮 4-from-8 可用且 `0000500 -> 0002500` 随 iter 提升;4-step 基线最佳 checkpoint 为 `0001000`。
- 速度:teacher 50-step 平均 165.24s vs 4-step student 6.59-6.63s(约 25x)——本地记录,数值引用前需重读远端 metrics(你不需要也不允许去核,直接注明"内部记录"即可)。
- 关键混淆(诚实呈现):两轮 4-from-8 之间 LR、batch、GPU 数同时变化,"batch 提升带来改善"的归因不成立,只能表述为配方级结论。
- 最大证据缺口:无任何量化质量指标,全部质量结论为肉眼判断。

## 已确认锚点(来自 T0,2026-07-06 用户已确认)

- T0 的三条 novelty 候选轴(终裁属 T3,本任务只做谱系级初判):
  - 轴 A:multi-step DMD2 视频 student 的质量对 `t_list` 形状高度敏感,文献缺少针对 distribution-matching 目标的少步 schedule 设计原则。
  - 轴 B:以 DMD2 为目标函数、用中间 8-step student 初始化 4-step student 的分阶段视频蒸馏配方,公开视频模型上未见完整先例。
  - 轴 C:少步视频 DMD2 的质量瓶颈主要在优化稳定性(LR、有效 batch、每锚点有效更新量)——注意该轴当前证据有三因素混淆,主张成立需受控实验。
- 已有 18 篇清单(指定只读文件 2)覆盖:Progressive Distillation、TRACT、CM、LCM、VideoLCM、DMD、DMD2、TDM(2503.06674)、ADM(2507.18569)、ADD、SDXL-Lightning、AnimateDiff-Lightning、PCM、T2V-Turbo、DOLLAR、MCM、InstaFlow、Flash Diffusion。这些**不必从零重查**;但该清单未做逐条原文复核,凡支撑你关键判断的条目,引用前必须点开原文确认,发现错误在报告中写带日期的更正。
- 2026-06-29 本地报告对第二轮 4-from-8 改善的"batch 归因"说法已被 T0 降级,不要沿用。
- Wan 生态直接竞品(CausVid、Self-Forcing、FastVideo、lightx2v 等,名称与状态均待核实)是 T3 的对抗核实重点;你检索中遇到时记入独立小节「Wan 生态相关工作(交 T3 深查)」,不精读、不下结论。

## T1 要回答的问题

1. 2023-2026 视频扩散少步生成/蒸馏的主流方法谱系是什么?按机制族整理(consistency 系 / distribution-matching 系 / adversarial 系 / trajectory-flow 系 / progressive-staged 系 / 混合),各族在视频模态上的验证程度如何?
2. "分阶段步数压缩 + 用中间 student checkpoint 初始化下一阶段"这一层面的直接先例有哪些?在视频扩散上是否存在与我们同构的做法(不论目标函数)?
3. 多步(4-16 step)student 的离散 timestep schedule 在少步蒸馏文献中被如何处理?有没有把 schedule 形状当一等研究变量的工作?
4. 视频少步蒸馏的主流评估协议是什么(VBench 及子维度、FVD、CLIP 对齐、temporal consistency、人评)?对我们"只有肉眼评估"的现状,迁移成本最低且审稿可接受的最小量化协议是什么?
5. 基于上述与 T0 方法描述,我们属于谱系中哪个分支?与主流是互补还是竞争?列出**必比先例 3-5 个**(每个一句为什么必比)。
6. 如果要讲投稿故事,本调研能支持哪些主张?对 T0 三条轴逐条给谱系级初判:文献支持 / 部分支持 / 不支持(终裁留给 T3)。

## 产出

写入文件:`research/T1_video_fewstep_distillation_landscape.md`(覆盖写,文件名固定)

结构:
1. **Executive Summary**:5-8 条,直接说结论。
2. **论文清单表**:Paper / Year / Venue(或 arXiv+状态)/ Modality(图像/视频)/ Method family / Step regime / 是否阶段化 / Relation to our project / Useful or Risk。
3. **3-5 篇精读卡**:核心问题、方法机制、关键公式或思想、实验设置与证据、对我们的启发、我们如何区别于它(含"不能 claim 的部分")。精读优先级:与轴 B(progressive/staged 视频蒸馏)最接近的工作 > 评估协议代表作。
4. **Gap 分析**:逐主张裁决;对"没有先例"类结论,**必须附上检索过的关键词组合与覆盖范围**,不许直接断言空白。
5. **对投稿叙事的可用表达**:3-5 条可直接使用的 bullet。
6. **对后续调研/实验的建议**:含给 T2(组件近邻)与 T3(对抗核实)的锚点——必比先例、可复用检索词组合、"不要重复调研"的范围声明。

格式要求:每篇论文给 arXiv/DOI/OpenReview 链接;不确定的事实标注"待核实";每节结论先行;中文撰写,论文名与术语保留英文。

## 调研纪律

- 每篇论文都要说清与本项目的关系,不只列标题。
- **不夸大 novelty**:严格区分"已有工作已做过 / 类似但不完全一样 / 可能存在空白"三档。
- 不在本领域但机制相关的工作放补充部分并说明区别。
- 产出服务投稿定位与后续任务,不是百科综述;语气专业、清楚、克制。
- 完成后在对话里给 ≤10 行执行总结,必须包含:我们的谱系归属一句话;必比先例 3-5 个;推荐的最小量化评估协议;T0 三条轴的谱系级初判各一词(支持/部分支持/不支持)。
