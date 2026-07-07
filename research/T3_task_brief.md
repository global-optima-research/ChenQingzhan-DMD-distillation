# T3 任务书:novelty 对抗核实与竞品裁决

- 生成:2026-07-06;定稿:2026-07-06(T1、T2 锚点与 planner 代码核实结论已注入)
- 状态:**最终版,可直接分发**
- 分发方式:把下方「任务书正文」整节粘贴给一个**新的**内容 agent 会话;完成后把它的 ≤10 行执行总结粘回 planner 会话验收
- 引用路径核实(2026-07-06):`research/T0_project_analysis.md`、`research/T1_video_fewstep_distillation_landscape.md`、`research/T2_dmd2_component_neighbors.md` 均存在且已验收

---

## 任务书正文(定稿后从此行以下全部粘贴)

你是一个强力 research agent,本次任务是 **T3 文献调研:novelty 对抗核实与 Wan 生态竞品裁决**。这是纯调研任务,不是工程任务。今天是 2026-07-06,所有"近两年/最新"判断以此为基准。你的职责是**对抗性**的:默认每条主张都能被文献击破,尽全力去击破;活下来的才算 novelty。

工作目录:

/Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation

## 角色与边界(先读这一节)

- 本目录的 CLAUDE.md 是为工程 agent 写的,本次会话**忽略其中的角色设定和上下文加载规则**。
- 不要读代码目录,不要 ssh 任何远端机器,不要跑训练/评估,不要修改 `research/` 以外的任何文件。
- 本地**只读以下 3 个文件**,其余背景本 prompt 已给足,不要再读别的:
  1. `research/T0_project_analysis.md`(第 1-3 节:方法、证据、novelty 轴)
  2. `research/T1_video_fewstep_distillation_landscape.md`(重点读第 4-4.5 节裁决与收窄表述、第 2.5 节 Wan 生态登记表、第 6.2 节遗留核实清单)
  3. `research/T2_dmd2_component_neighbors.md`(重点读第 4 节主张裁决与 6.2 节幸存主张/复核清单;**注意其 4.4 节末尾的 planner 验收注记——该报告 4.1 主张已作废**)
- 调研用可靠来源:论文原文、arXiv、OpenReview、CVF、官方代码库、官方 model card / release notes。**近两年的工作必须实时检索,不要凭记忆**;竞品的发布状态、模型规格必须以官方来源为准。
- 可并行的检索尽量并行。本任务不追求清单数量,追求**裁决质量**:每条主张至少 3 组不同角度的对抗检索;竞品逐个开官方页面核实。

## 项目背景(一段话)

我们在 FastGen 框架上把 Wan2.1-T2V-1.3B 的 50-step teacher 用 DMD2 目标分阶段蒸馏为 8-step、再到 4-step student。T1 已完成谱系盘点,T2 已完成组件近邻盘点。本 T3 是投稿叙事的守门任务:把幸存主张逐条对抗裁决,核实 Wan 生态直接竞品,产出"能说/不能说"红线与必引划界清单,并给出评估协议平移建议。

## 我们的方法(文献对比用的准确描述,来自 T0)

一句话:用 DMD2 式 distribution matching + 对抗目标,把 50-step 文生视频扩散 teacher(Wan2.1-T2V-1.3B,latent `[16,21,60,104]`,832x480、81 帧)分阶段蒸馏成 8-step 中间 student、再蒸馏成 4-step 部署 student 的 progressive few-step distillation。

- 结构:teacher 冻结;student 同架构可训,few-step ODE 式采样,推理不用 CFG;4-step 从 8-step 最优 checkpoint 初始化(仅生成网络权重)。辅助网络:fake score network(同架构,在线拟合 student 分布)+ 判别器 = **冻结 teacher backbone 中间层(15/22/29)特征上的可训 multiscale MLP 头**(LADD / teacher-feature 谱系;2026-07-06 经 FastGen 代码核实)。
- 信号:teacher score(CFG=5)− fake score 的 distribution matching 梯度;判别器对真实/生成 latent 的对抗信号(同 t 同噪声,生成端权重 0.03);fake score x0-prediction 在线更新;student 每 5 iter 更新一次的 two-time-scale。
- 自由度:离散时间锚点 `t_list`(4-step `[0.999, 0.937, 0.833, 0.624, 0.0]`;8-step 有插值/均匀两种 9 锚点)、LR、有效 batch、student 更新频率、checkpoint 早停。

## 实验现状(结论可直接引用,不必读实验报告)

- 已证明:各 run 超参已远端核实(8-step lr_original:LR `1e-5`/batch 12;两轮 4-from-8:LR `1e-5`/batch 12 与 LR `5e-6`/batch 16/8 卡,init 同为 8-step `0002500`);速度约 25x(内部记录)。
- 未量化:全部质量结论为肉眼判断(8-step 可用、4-from-8 第二轮可用且随 iter 提升、第一轮后期物理崩坏)。
- 关键混淆:两轮 4-from-8 之间 LR、batch、GPU 数同时变化,改善归因不成立。
- 这意味着:凡需要"我们实验证明了 X"支撑的主张,你在裁决时都要按"实验证据弱"处理;novelty 只能建立在机制/配方设计上,不能建立在未量化的质量对比上。

## 已确认锚点

来自 T0(2026-07-06 用户已确认):证据现状如上;三条轴原文见下。

来自 T1(2026-07-06 已验收;载重条目 CoDMD/GPD/Phased DMD/rCM 的存在与摘要主张已由 planner 联网抽查确认):

- 必比先例 5 个:DMD2(NeurIPS 2024 Oral,目标函数来源)、CoDMD(arXiv 2606.21982,同基座同步数 concurrent,Wan 官方团队参与,VBench 84.46/84.87 硬坐标)、rCM(arXiv 2510.08431,ICLR 2026,GAN-free 竞线,diversity 批评须回应)、Phased DMD(arXiv 2510.27684,占用 phased/progressive DMD 命名与 SNR 分相思想)、GPD(arXiv 2602.01814,同基座 48→6 步数接力先例,纯轨迹回归、无 DMD/GAN)。
- 谱系归属判词:我们 = DMD2 完整配方 × 步数接力(50→8→4),视频模态;与 2026 Wan 系 DMD 变体的竞争焦点是"它们改目标函数 vs 我们改训练调度/阶段协议",两轴正交。
- "步数接力在视频上无先例"已被 T1 裁决**不成立**(GPD/SwiftVideo/AnimateDiff-Lightning/Imagen Video),不要在这一层重复检索;三条轴的收窄表述见下方主张清单,即本任务的裁决对象。
- 命名红线:phased DMD / progressive distribution matching 已被占用,我们措辞用 step-count relay / progressive step reduction。
- 可复用检索词:T1 报告 4.2/4.3/4.4 节内联了全部检索串;Wan 生态社区面词组见其 6.2 节。
- T1 遗留给本任务的两张清单(venue 核实清单、数字复核清单)在其 6.2 节,**逐条处理并在报告中给出结果**。

来自 T2(2026-07-06 已验收)与 planner 代码核实(2026-07-06):

- 四类组件最近邻:multi-step DM 构造——Self-Forcing(同基座受控:on-policy 84.31 vs teacher-forcing 82.32);锚点——TMD(同基座 t_dmd/t_student 双 shift 消融,且证明"VBench 总分测不出 mode collapse");判别器——One-Forcing(0.03 / TTUR 5:1 / 无正则逐项重合,宿主为 fake score,经联网核实)及 ASD / Taming DiT / AAD-1 近同构;稳定性——Seaweed-APT(ICML 2025:视频 batch 256 崩 / 1024 稳,近似 R1 无它即崩)。
- **planner 代码核实(改变主张口径)**:我们判别器实为**冻结 teacher backbone 15/22/29 层特征 + 可训 multiscale MLP 头**(teacher-feature 谱系,先例充分:LADD / Diffusion2GAN / SF-V / NitroFusion 为划界必引)——T2 原"live 生成器骨干组合无先例"主张**已作废,不在裁决范围**;`gan_use_same_t_noise=True` 为 real/fake 共享同一 t 与同一 ε(FastGen 方法默认 False,主动配置);FastGen 内置近似 R1 但我们全部 run 未启用。
- T2 更正与已核 venue:ADM/DMDX = ICCV 2025 Highlight;SenseFlow = ICLR 2026 Poster;Seaweed-APT = ICML 2025;POSE = AAAI 2026;TDD = AAAI 2025。
- T2 遗留复核清单(venue 与数字,其 4.4 与 6.2 节)与 T1 6.2 清单合并,逐条处理并在报告中给出结果。

## 待裁决主张清单

对下列每条主张给出裁决:**成立 / 部分成立(给收窄后的窄带版本)/ 不成立**。每条裁决必须附:对抗检索词组合(至少 3 组)、覆盖范围(venue+年份)、击破/幸存的关键证据(带链接)。

- 轴 A(T1 收窄版,初判"部分支持"):在 4-8 步渐进式 DMD2 视频蒸馏(Wan2.1-T2V-1.3B)设置下,首次把推理锚点 `t_list` 形状(均匀 vs 不同高噪密集度)当显式实验变量做系统消融并给出经验准则。已知最近反例:TMD(shift 标量消融,1-2 步)、CDM(图像、取消锚点)、Phased DMD(训练区间而非推理锚点)、FlashMol(分子域)。裁决点:该收窄版是否仍站得住、是否需再收窄。
- 轴 B(T1 收窄版,初判"部分支持";判别器措辞经 planner 代码核实修正):把 DMD2 式分布匹配 + 轻量冻结-teacher-特征判别器作为**每一阶段统一目标**、沿 50→8→4 步数轴接力(只继承生成器、重置优化器/fake score/判别器)、在公开 Wan2.1-T2V-1.3B + OpenVid-1M 上给出完整可复现配方——据 T1 检索无完整先例。**条件项(最高优先核实)**:lightx2v step_distill 配方是否含"步数递减 + student checkpoint 接力";并给出 NVIDIA FastGen 官方公开线(14B DMD2)与本工作关系的准确对外表述。
- 轴 C(T1 收窄版,初判"部分支持"):"LR × 有效 batch × 每锚点有效更新量"的三因素受控归因在少步视频 DMD 上无先例;已知最近证据:Seaweed-APT(GAN 式,batch/LR 受控)、DMD2/SenseFlow/Flash-DMD/ADM(fake score 更新率消融)。硬前提:我们必须补受控实验,否则该轴从论文主张撤下。裁决点:空白是否真实、"每锚点有效更新量"变量是否确无人做、归因竞争(目标本性说)是否留有我们的空间。
- 主张 D(T2 幸存,经 planner 代码坐实口径):"real/fake 共享同一 timestep 与同一噪声 ε 的配对判别(`gan_use_same_t_noise`)在少步蒸馏**论文与公开实现**中均无单独消融"——论文侧 T2 已查(ASD 明确独立采样 ε);T3 需补扫开源 repo 配置旗标(lightx2v / FastVideo / Self-Forcing / Wan2.2-Lightning / ModelTC 等)后终裁。注意该主张的价值上限是"可自补的独家消融点",不是方法新颖性。
- 主张 E(T2 幸存,轴 C 支线):4-step relay 重置后的 TTUR / warmup / fake-EMA 选择零文献覆盖——T3 只需确认 2026H2 无新工作抢占;属"须自补实验"而非"可 claim 空白"。
- **已作废、不要裁决**:"live 生成器骨干 + 纯 MLP 头 + 同 t 同噪声组合无先例"(planner 代码核实推翻前提:我们属 teacher-feature 谱系)。

## T3 要回答的问题

1. 上述主张清单逐条裁决(格式见「待裁决主张清单」)。
2. **Wan 生态竞品实况**(本任务最高优先级):系统检索 Wan2.1 / Wan2.2 的少步蒸馏与加速产物——候选起点:CausVid(-Wan)、Self-Forcing、FastVideo、lightx2v、Wan 官方 distilled/turbo checkpoint、其他社区 recipe(名单本身待你核实与扩充)。每个竞品给:目标函数族(是否 DMD 家族)、step 数、模型规格(是否 1.3B T2V)、发布状态与日期、开源程度、与轴 B 的重叠度。**每个竞品必须开官方 repo/model card/论文原文核实,不许转引二手信息。**
3. 必引划界清单:哪些工作必须引用并显式划界?每个给一句"我们与它的本质区别"和一句"我们不能 claim 的部分"。
4. "能说/不能说"红线:给最终对外口径——能说 5-8 条,不能说 3-5 条(每条注明触发它的文献)。
5. 评估协议平移:从本文献线的成熟指标(VBench 子维度、FVD、CLIP 对齐、temporal consistency、人评协议)中,给我们选出最小可行协议与可选扩展;判断协议本身是否有机会成为一个轻量贡献(参照:该领域是否缺少针对"少步蒸馏退化模式"的标准化评估)。
6. 若三条轴全部被显著收窄:基于你检索中看到的真实空白,给 1-2 个替代窄带主张候选(标注证据需求)。

## 产出

写入文件:`research/T3_novelty_adjudication.md`(覆盖写,文件名固定)

结构(六段式适配本任务):
1. **Executive Summary**:5-8 条,先说三轴终裁与竞品是否击穿轴 B。
2. **主张裁决表**:Claim / 裁决 / 收窄后版本(如有)/ 检索词组合 / 覆盖范围 / 关键证据链接。
3. **竞品精查卡**(每个 Wan 生态竞品一卡):规格、目标函数、step、状态、与我们的重叠与区别、必引与否。
4. **Gap 分析与红线**:"能说/不能说"清单 + 必引划界清单。
5. **评估协议平移建议**:最小协议、可选扩展、协议贡献可行性判断。
6. **对写作与后续实验的建议**:含对前序报告(T1/T2)需要回填的更正注记(如发现错误,列出条目与日期,由 planner 执行回填)。

格式要求:每条证据给链接;太新无法二次核实的条目标"待核实";**载重条目**(支撑关键判词的)单独列"引用前必点原文"清单;每节结论先行;中文撰写,术语保留英文。

## 调研纪律

- 你的默认立场是击破,不是辩护;"未找到先例"必须以检索词组合+覆盖范围背书。
- **novelty 三分法**严格执行;禁止"首个/首创"式绝对化表述,除非附检索覆盖并加"据我们所知"。
- 竞品信息以官方来源为准,注明核实日期(发布状态会过期)。
- 与前序报告矛盾时不悄悄改口径:在本报告中列出更正条目,由 planner 回填原报告。
- 完成后在对话里给 ≤10 行执行总结,必须包含:三轴终裁各一行;竞品是否击穿轴 B(一句话);红线中最重要的一条"不能说";必引清单条数;建议的最小评估协议。
