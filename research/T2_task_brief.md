# T2 任务书:DMD2 方法组件近邻盘点

- 生成:2026-07-06;定稿:2026-07-06(T1 验收通过,锚点已注入)
- 状态:**最终版,可直接分发**
- 分发方式:把下方「任务书正文」整节粘贴给一个**新的**内容 agent 会话;完成后把它的 ≤10 行执行总结粘回 planner 会话验收
- 引用路径核实(2026-07-06):`research/T0_project_analysis.md`、`research/T1_video_fewstep_distillation_landscape.md` 均存在(T1 已验收)

---

## 任务书正文(从此行以下全部粘贴)

你是一个强力 research agent,本次任务是 **T2 文献调研:DMD2 方法组件的近邻盘点(multi-step DMD 变体 / timestep 锚点处理 / 判别器设计 / 训练稳定性)**。这是纯调研任务,不是工程任务。今天是 2026-07-06,所有"近两年/最新"判断以此为基准。

工作目录:

/Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation

## 角色与边界(先读这一节)

- 本目录的 CLAUDE.md 是为工程 agent 写的,本次会话**忽略其中的角色设定和上下文加载规则**。
- 不要读代码目录,不要 ssh 任何远端机器,不要跑训练/评估,不要修改 `research/` 以外的任何文件。
- 本地**只读以下 2 个文件**,其余背景本 prompt 已给足,不要再读别的:
  1. `research/T0_project_analysis.md`(重点第 1 节方法描述)
  2. `research/T1_video_fewstep_distillation_landscape.md`(**只读其中 2.2 清单表、3.4 DMD2 精读卡、6.1 给 T2 的锚点节;其余章节不要读**)
- 调研用可靠来源:论文原文、arXiv、OpenReview、CVF、官方代码库。**近两年的工作必须实时检索,不要凭记忆**(覆盖 CVPR / ICCV / ECCV / NeurIPS / ICML / ICLR / SIGGRAPH(Asia) / AAAI / ACM MM 2024-2026 与 arXiv 2024-07 至今)。
- 可并行的检索尽量并行。控制聚焦:清单表 10-20 篇、精读 3-5 篇。**T1 已完成谱系级盘点,不要重做谱系**;你的增量在组件粒度的机制细节与消融证据。

## 项目背景(一段话)

我们在 FastGen 框架上把 Wan2.1-T2V-1.3B 的 50-step teacher 用 DMD2 目标分阶段蒸馏为 8-step 中间 student、再到 4-step 部署 student。T1(2026-07-06 已验收)确定了谱系定位:我们与 2026 年 Wan 系 DMD 变体(CoDMD/SGMD/Data-Forcing/Phased DMD)的竞争焦点是"它们改目标函数 vs 我们改训练调度/阶段协议",两条改进轴正交。本 T2 下沉到组件层:把我们方法的每个组件在更广文献(不限视频)中的先例、消融证据和升级方案盘点清楚,为 T3 的 novelty 终裁和组件实验提供素材。

## 我们的方法(文献对比用的准确描述,来自 T0)

一句话:用 DMD2 式 distribution matching + 对抗目标,把 50-step 文生视频扩散 teacher(Wan2.1-T2V-1.3B,latent `[16,21,60,104]`,832x480、81 帧)分阶段蒸馏成 8-step 中间 student、再蒸馏成 4-step 部署 student 的 step-count relay(注意:不叫 progressive/phased DMD,该命名已被占用)。

- 结构:teacher 冻结;student 与 teacher 同架构、可训,few-step 确定性(ODE 式)采样,推理不用 CFG;4-step 阶段从 8-step 最优 checkpoint 初始化(仅生成网络权重,优化器与辅助网络重置)。辅助网络两个:在线拟合 student 分布的 fake score network(同架构),以及基于**生成器 backbone 中间层(第 15/22/29 层)特征**的 multiscale MLP 判别器(`multiscale_down_mlp_large`)。
- 信号:(a) distribution matching 梯度 = teacher score(CFG=5)− fake score(reverse-KL / VSD 式);(b) 对抗信号:判别器区分真实视频 latent 与 student 生成 latent(同 timestep 同噪声,`gan_use_same_t_noise=True`),生成端权重 0.03,真实数据 OpenVid-1M;(c) fake score 以 x0-prediction 在线拟合;two-time-scale:student 每 5 次迭代更新一次(`student_update_freq=5`),fake score 与判别器每次迭代更新。
- 约束/自由度:student 采样轨迹由离散时间锚点 `t_list` 固定(4-step:`[0.999, 0.937, 0.833, 0.624, 0.0]`,即社区 shift=5 标准配置;8-step 试过"高噪密集插值"与"均匀"两种 9 锚点);训练时 t 采样为 shifted 分布。

## 实验现状(结论可直接引用,不必读实验报告)

- 已证明(远端核实):8-step lr_original(LR `1e-5`、batch 12、2500 iter);两轮 4-from-8(第一轮 LR `1e-5`/batch 12,第二轮 LR `5e-6`/batch 16/8 卡),init 同为 8-step `0002500`;8-step 均匀 t_list 消融(`_step8_normalize`)与 `student_update_freq=2` 消融(`_step8_freq`)已跑但结论未整理。
- 未量化:全部质量结论为肉眼判断(低 LR 模糊、lr_original 后可用、第一轮 4-from-8 后期物理崩坏、第二轮可用且随 iter 提升)。
- 关键混淆:两轮 4-from-8 之间 LR、batch、GPU 数同时变化,改善归因不成立。
- 我们判别器的精确结构(骨干共享方式等)尚未从代码核实,引用我们自己的设计时保持在上述描述粒度。

## 已确认锚点(来自 T0 与 T1,2026-07-06 已验收;直接引用,不要重查)

- 必比先例 5 个(T1 终定):DMD2(NeurIPS 2024 Oral)、CoDMD(arXiv 2606.21982,同基座同步数 concurrent)、rCM(arXiv 2510.08431,ICLR 2026)、Phased DMD(arXiv 2510.27684)、GPD(arXiv 2602.01814)。T1 已对 DMD2/CoDMD/GPD/Phased DMD/SDXL-Lightning 做过精读卡,**不要重复精读这五篇**,可引用其结论。
- DMD2 机制事实(T1 已核实原文,可直接引用):TTUR fake:gen = 5:1(消融:1 不稳、10 收敛慢、5 最优);GAN 判别头寄生在 fake score UNet bottleneck,权重 1e-3(SDXL)/3e-3(ImageNet);multi-step 锚点 999/749/499/249 均匀、无任何锚点消融;backward simulation 消融 Patch FID 20.86→24.21。
- 判别器专题近邻(T1 已登记待深挖):One-Forcing(arXiv 2605.23458,**与我们配置最接近:Wan2.1-1.3B、transformer 层 {21,29} 特征、GAN 权重 0.03、critic 每 iter / 生成器每 5 iter——必精读并逐项对照**)、LADD(2403.12015,teacher 生成特征做判别)、Diffusion2GAN(2405.05967,扩散骨干判别器)、V-PAE(2508.21019,AAAI 2026,判别器复用生成器参数)、NitroFusion(2412.02030,判别器头池+refresh)、Flash-DMD(2511.20549,高噪 DMD/低噪 GAN 分工)、SF-V(2406.04324,spatial-temporal 头)、OSV(2409.11367)。
- fake score / DM 目标稳定化近邻(T1 已登记):SenseFlow(2506.00523,ICLR 2026,IDA/ISG)、MagicDistillation(2503.13319,LoRA fake DiT)、SGMD(2605.30116,免 fake score 追踪)、Phased DMD(子区间 score matching + 每相从 teacher 重置)、Few-Step SiD(2505.12674,multi-step DM 目标的"均匀混合"定义)、f-distill(2502.15681)、SiD(2404.04057)、SiD-DiT(2509.25127)、CDM(2605.06376,连续 schedule 取消锚点)。
- 可叠加组件候选(T1 提出,T2 评估可行性):CoDMD 关系正则(零额外网络)、GPD 频域高频损失与 CFG 退火、Phased DMD reverse nested interval(训练 t 高噪端覆盖)。
- 稳定性数字坐标(T1 已核实其一手/二手状态):Seaweed-APT(2501.08316,ICML 2025)视频侧 batch 256→mode collapse / 1024 不会、LR 5e-6→3e-6 求稳(**必精读细节**);ADM(2507.18569)fake 更新 1→8 次边际收益 CLIP 35.2557→35.3299 却 2.53x 训练时间。
- 轴 A 收窄版(T1 裁决"部分支持"):"在 4-8 步 DMD2 视频蒸馏设置下把推理锚点形状当显式变量做系统消融并给出准则"仍是空白;已知最近反例 TMD(2601.09881,shift 标量消融)、TDD(2409.01347,CM 目标锚点精选)、FlashMol(2605.07020,分子域)、Lip Forcing(2606.11180,单锚点落点)。
- **不要重复调研**:机制族谱系、staged 三语义分类、评估协议选型、Wan 生态存在性登记、solver 侧 schedule 谱系(AYS/GITS/OSS/S4S)——T1 已覆盖;只在与蒸馏内部锚点直接相关时引用。
- 命名红线:我们的措辞用 step-count relay / progressive step reduction,避开 phased/progressive DMD。

## T2 要回答的问题

1. **multi-step DM 目标的构造对比**:DMD2 backward simulation vs Few-Step SiD"均匀混合" vs Phased DMD 子区间 vs CDM 连续随机 schedule——各自如何定义 multi-step 训练分布?对 train-inference mismatch 与 mode collapse 各有什么消融证据?有没有 video-specific 的改法值得我们借鉴?
2. **蒸馏内部锚点处理的受控证据**(轴 A 组件基础):TDD、TMD、FlashMol、Lip Forcing、Phased DMD nested interval——逐个提取"改了什么锚点变量、量化差多少、结论是否可迁移到视频 multi-step DMD";综合回答:我们的 t_list 消融矩阵应扫哪些形状、观察哪些指标最敏感。
3. **判别器设计对比表(本任务核心)**:按 特征来源(生成器 backbone / fake score / teacher 生成特征 / 独立或外部骨干)× 时间维(单帧 / 3D conv / 时空注意力)× 条件(timestep、flow/起点、文本)× 正则(R1、头池 refresh、weak-GAN 权重)组织;每条路线给代表工作与消融证据。**One-Forcing 必须精读**并逐项对照我们的配置(层 15/22/29 vs {21,29}、权重 0.03、同 t 同噪声)。回答:我们当前判别器最可能的两个弱点与对应升级方向。
4. **训练稳定性定量结论盘点**:TTUR/更新率(DMD2、SenseFlow、Flash-DMD、ADM——核对 T1 给的数字并补漏)、batch 与 LR(Seaweed-APT 必精读)、EMA、判别器正则——哪些数字可直接引用、哪些必须我们自己补实验?
5. **组件升级建议 3-5 个**:每个给"改什么 / 预期解决哪个失败模式(模糊、动态弱、物理崩坏、后期退化)/ 文献依据 / 与我们'训练调度轴'故事的相容性(不能喧宾夺主)"。
6. **组件层主张支持度初判**(终裁留给 T3):围绕轴 A 与判别器设计,逐条判:文献支持 / 部分支持 / 不支持。

## 产出

写入文件:`research/T2_dmd2_component_neighbors.md`(覆盖写,文件名固定)

结构:
1. **Executive Summary**:5-8 条,直接说结论。
2. **论文清单表**:Paper / Year / Venue(标注"待核实"如未确认)/ Component(DM 变体 / 锚点 / 判别器 / 稳定性)/ Mechanism 一句话 / Evidence 强度 / Relation to our project / Useful or Risk。
3. **3-5 篇精读卡**:优先 One-Forcing、Seaweed-APT,再从 multi-step DM 变体与判别器路线中挑证据最强的 1-3 篇;不要与 T1 已精读的五篇重复。每卡含:核心问题、机制、关键证据数字、对我们的启发、我们如何区别于它(含"不能 claim 的部分")。
4. **Gap 分析**:逐主张裁决;"没有先例"类结论必须附检索词组合与覆盖范围。
5. **对投稿叙事的可用表达**:3-5 条 bullet(组件层)。
6. **对后续实验与 T3 的建议**:组件升级建议按"收益/风险/实现成本"排序;给 T3 的幸存主张清单与新增检索词。

格式要求:每篇论文给 arXiv/DOI/OpenReview 链接;不确定的事实标注"待核实";每节结论先行;中文撰写,论文名与术语保留英文。

## 调研纪律

- 每篇论文都要说清与本项目的关系,不只列标题。
- **不夸大 novelty**:严格区分"已有工作已做过 / 类似但不完全一样 / 可能存在空白"三档。
- 不在本领域但机制相关的工作(如经典 GAN 判别器正则)放补充部分并说明区别。
- 产出服务 T3 终裁与组件实验设计,不是百科综述;语气专业、清楚、克制。
- 完成后在对话里给 ≤10 行执行总结,必须包含:四类组件各自的最近邻一句话;我们判别器的两个最可能弱点;最值得先做的 1 个组件实验;轴 A 的组件层初判(支持/部分支持/不支持)。
