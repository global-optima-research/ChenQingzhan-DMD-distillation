# T1 调研报告:视频扩散少步蒸馏领域主线谱系

- 写作日期:2026-07-06(所有"最新/近两年"判断以此为基准)
- 任务书:`research/T1_task_brief.md`;方法与证据底本:`research/T0_project_analysis.md`
- 调研方式:8 路并行实时检索(consistency / distribution-matching / adversarial / trajectory-flow / progressive-staged / timestep-schedule / evaluation / 2025-26 增量)+ 5 篇精读(arXiv 原文全文)+ 3 条 novelty 轴对抗反证检索;共约 100 组检索串、60+ 篇原文页面逐一核实。检索串明细见第 4 节。
- 结论定位:谱系级初判,终裁属 T3。

> **T3 §6.1 回填更正(2026-07-14)。** 本报告以下结论经 T3 终裁修正,凡冲突以 T3 为准:(1) **轴 A 作废**——TMD(已确认 CVPR 2026)对 t_list 形状消融的覆盖远超本报告记录(Table 5 的 1/2-step + Table 11 的 3/4-step γ=5 vs γ=10 + App. B.2 高噪曲率准则 + App. A.2 确认 shift 作用于推理时间网格),"首次把锚点形状作显式变量"不成立,t_list 降级为配方说明;(2) **轴 B 被上游压缩**——lightx2v"两段式"疑云解除(单段 data-free 纯 DMD、无接力、无 1.3B 蒸馏产物),但上游 NVlabs/FastGen 已公开我方全部单阶段超参(t_list、GAN 0.03、判别层 15/22/29、CFG 5、TTUR 1:5、lr 1e-5),可主张机制面只剩 50→8→4 接力编排 + 8-step 中间阶段 + 跨阶段重置 + OpenVid 实例化;(3) venue 与载重数字批量更正见 T3 §6.1 第 6-7 条。

---

## 1. Executive Summary

1. **谱系归属(一句话)**:我们属于 distribution-matching 系(DMD2 目标 + 轻量 GAN)× progressive 步数接力(step-count relay:50→8→4,用中间 student checkpoint 初始化下一阶段)的组合,落在视频模态;该**组合**在公开视频模型上未见完整先例,但**每个组成部分都已被近邻覆盖**,novelty 空间收窄至"组合 + 接力协议细节(只继承生成器、重置优化器/fake score/判别器)+ 消融证据"。
2. **撞车密度警报**:2025Q4–2026H1 DM 系视频蒸馏的主战场恰好是 Wan2.1-T2V 1.3B/14B + 4-step。CoDMD(arXiv 2606.21982,2026-06-20,Wan 官方团队参与)与我们设定几乎完全重合(Wan2.1-T2V-1.3B/14B,50→4,VBench 84.46/84.87);Data-Forcing(2606.18478)、SGMD(2605.30116)、TMD(2601.09881)、rCM(ICLR 2026)全部以 DMD/DMD2 为 baseline。"DMD2 + Wan2.1 + 4-step"本身已零 novelty,论文重心必须落在 progressive 接力与受控消融上。
3. **"progressive/staged"已分化出三种语义**,写作时必须显式区分:(a) SNR 子区间分相(Phased DMD——分相产出 MoE 多专家);(b) 训练范式串联(SwiftVideo/OSV/Hierarchical Distillation——先 consistency 后分布/对抗);(c) 步数递减 + 前阶段 student 权重接力(Progressive Distillation/SDXL-Lightning/Hyper-SD/GPD/我们)。**"Phased DMD"与"progressive distribution matching"命名已被占用**(arXiv 2510.27684),我们必须改用 step-count relay / progressive step reduction 一类措辞。
4. **轴 B(progressive DMD2 for video)谱系级初判:部分支持**。GPD(2602.01814)已在同基座 Wan2.1-T2V-1.3B 上做 48→6 分阶段蒸馏且明文"下一阶段 student 从上一阶段 checkpoint 初始化"——但其目标是纯轨迹回归(无 DMD、无 GAN、无真实数据)。"步数接力"本身不新;"每阶段统一用 DMD2 目标 + 接力 + 辅助网络重置"的完整配方仍无先例(该结论以 lightx2v 生态两段式配方的 T3 核实为条件)。
5. **轴 A(t_list schedule 设计)谱系级初判:部分支持**。敏感性现象已被 TMD 以数字记录(Wan2.1-1.3B,1–2 步 shift 消融,84.39 vs 83.44);我们的 4-step t_list [0.999,0.937,0.833,0.624,0.0] 与 Phased DMD/lightx2v 的 {1000,938,833,625}(shift=5)完全同源,是社区标准配置、不可当贡献。剩余空隙:视频 multi-step(4–8 步)DM 目标下**推理锚点形状**(均匀 vs 高噪密集程度)的系统研究与设计准则,文献确实没有。
6. **轴 C(优化稳定性归因)谱系级初判:部分支持**。文献主流把少步视频 DMD 失败归因于 reverse-KL 目标本性(rCM/CoDMD/Data-Forcing/Adaptive Video Distillation)或分布支撑错配(MagicDistillation/ADM);优化侧只有零散单点消融(DMD2 的 TTUR、Seaweed APT 的 batch/LR、Flash-DMD 的更新率)。"LR × 有效 batch × 每锚点有效更新量"的三因素受控归因在视频 DMD 上是空白;但该轴成立的硬前提是先补受控实验(我们内部证据当前三因素混淆)。
7. **评估协议结论**:主协议 = VBench full(946 prompts × 每 prompt 5 个视频,报 Total/Quality/Semantic);消融子协议 = VBench 官方支持自采视频的 6 个质量维度 + CD-FVD(不要用 I3D-FVD)+ JEDi(小样本);人评按 T2VHE 规范做 4-step vs teacher 50-step 的 side-by-side。**Dynamic Degree 与多样性指标必须主动报**——Phased DMD 已实证 DMD2 在 Wan 上动态性大幅下降(Optical Flow 10.26→3.23),且自曝 VBench 分数与人评趋势相反;这两点是审稿人现成的弹药。
8. **必比先例 5 个**:DMD2、CoDMD、rCM、Phased DMD、GPD(每个一句话理由见 4.5 节)。

---

## 2. 领域谱系与论文清单

### 2.1 机制族谱系与视频验证程度(回答 Q1)

结论先行:五个机制族在视频上的验证程度排序为 consistency ≈ distribution-matching > adversarial > trajectory-flow;progressive/staged 是与机制族正交的第六个维度;2025–2026 的明确趋势是**各族在视频上合流为 hybrid,且 DMD 项正在成为公共组件**。

**(a) Consistency 系**(CM→LCM→VideoLCM/AnimateLCM→PCM→MCM→T2V-Turbo v1/v2→sCM→rCM)。视频验证最充分的一族,但公认"可用而不够":离散时间 CD 在 4–8 步可用,失败模式为少步模糊/细节丢失、动态变弱、1 步不稳定(PCM 自述)、受公开视频数据画质拖累(MCM 原文)。几乎所有成功案例都要外挂辅助信号——GAN 判别器(MCM/OSV/PCM)、reward(T2V-Turbo v1/v2)、motion-appearance 解耦(MCM/AnimateLCM)。连续时间 consistency 直到 rCM(ICLR 2026,NVIDIA)才在 Wan2.1(含 14B)视频规模站住,且 rCM 自己承认纯 sCM 细节差、要靠 score distillation(DMD 式)正则来救。
**(b) Distribution-matching 系**(DMD→DMD2→SiD/f-distill 理论线→视频:2412.05899(AnimateDiff 4-step)→MagicDistillation→TDM(CogVideoX)→CoDMD/Data-Forcing/SGMD/Salt(Wan2.1)→AR 分支 CausVid/Self-Forcing(交 T3))。当前视频少步蒸馏的**主导族与主战场**;公认痛点收敛于两点:reverse-KL mode-seeking 导致过饱和/动态变弱/多样性坍缩(f-distill、rCM、Data-Forcing、CoDMD、pi-Flow 都在攻击),以及 fake score 追踪成本高与不稳定(SGMD、MagicDistillation、SenseFlow 都在修)。我们的 two-time-scale 与轻量 GAN 都是对这两点的既有回应。
**(c) Adversarial 系**(ADD→LADD→SDXL-Lightning/AnimateDiff-Lightning→UFOGen/NitroFusion/Diffusion2GAN→视频:SF-V、OSV、Seaweed-APT/APT2、V-PAE)。纯对抗已在 8B 级视频模型上做到 1-step(APT,ICML 2025),但 APT 自承结构完整性与文本对齐退化——反衬"DMD 主目标 + 轻量 GAN 辅助"的合理性。"DM 目标 + 对抗项混用"有充分先例链(DMD2→SANA-Sprint→ASD→One-Forcing→Flash-DMD/AdvDMD),**混用本身不构成新颖性**。
**(d) Trajectory / rectified-flow / flow-map 系**(PD→TRACT→InstaFlow/reflow→CTM→PeRFlow/Rectified Diffusion/ProReflow→Shortcut Models→MeanFlow→Align Your Flow→pi-Flow→视频:GPD、AnyFlow、AccVideo(交 T3))。纯轨迹方法的验证基本停留在图像;2025–2026 在视频上站住的几乎都带 reverse-KL 或对抗项(TMD = flow map + DMD;rCM = sCM + score 正则)。该族对我们的两条标准批评:pi-Flow/rCM 的"DMD quality–diversity 权衡"、Shortcut Models/AnyFlow 的"多阶段 pipeline 复杂/固定步数蒸馏范式退化"——回应口径:视频尺度纯轨迹未单独站住;部署步数固定时 staged 成本可摊销。
**(e) Progressive/staged 维度**(与机制族正交):见 Executive Summary 第 3 条的三种语义。步数接力(语义 c)在图像已完全成熟(SDXL-Lightning:"student 收敛后充当下一阶段 teacher"+"判别器每阶段重置";Hyper-SD:"resume the model weights from the previous stage"),在视频有 Imagen Video(2022,PD 至每级 8 步)、AnimateDiff-Lightning(motion module)、GPD(Wan2.1,48→6)。
**(f) Timestep-schedule 维度**(回答 Q3,详见 4.2 节):solver 侧"schedule 当一等变量"已完全成熟(DDSS→AYS(ICML 2024,含 SVD 视频)→GITS→Optimized Time Steps(CVPR 2024)→OSS→S4S→INDIS per-instance),但全部作用于未蒸馏模型的采样;蒸馏内部只有三条路线:等距目标选择(TDD/生态惯例)、区间划分分工(Phased DMD/Flash-DMD)、取消离散锚点(CDM)。**没有任何工作把视频 multi-step DM student 的推理锚点形状当显式变量做系统消融**——实践高度分裂且几乎从不被论证(DMD2 均匀 999/749/499/249 无消融;TDM 均匀;SGMD 极端高噪密集 {1000,960,889,727} 零论证)。

### 2.2 论文清单表(核心新条目,20 篇)

known-18 之外本次检索确认的最相关条目。凡标"待核实"的 venue 均为 arXiv 页面/搜索结果推断,未在会议官网确认。

| # | Paper | 年 | Venue/状态 | 模态 | 机制族 | 步数 | 阶段化 | 与本项目关系 | Useful/Risk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [rCM: Score-Regularized Continuous-Time Consistency](https://arxiv.org/abs/2510.08431) | 2025 | ICLR 2026 | 图+视频 | consistency+DM | 1–4 | 否 | 首次把 sCM 扩到 Wan2.1(含 14B)视频规模,以 DMD2 为对照,批评其 mode collapse;GAN-free | **Risk**:方法族正面竞品,必比 |
| 2 | [CoDMD: Copula-aware DMD](https://arxiv.org/abs/2606.21982) | 2026 | arXiv 06-20,concurrent | 视频 | DM | 50→4 | 否(单阶段) | 与我们设定几乎重合:Wan2.1-T2V-1.3B/14B、50→4、TTUR 5:1;VBench 84.46/84.87;data-free 无 GAN | **Risk**:同设定竞品 + 评测模板,必比 |
| 3 | [Data-Forcing Distillation](https://arxiv.org/abs/2606.18478) | 2026 | arXiv 06-16 | 视频 | DM | few-step | 否(在 DMD2 ckpt 上后训) | 同基座 Wan2.1-1.3B;直击 DMD 系过饱和 + 多样性坍缩;消融显示从 teacher 直接初始化会发散 | **Risk**:失败模式弹药 + 竞品 |
| 4 | [Phased DMD](https://arxiv.org/abs/2510.27684) | 2025 | 待核实(OpenReview 有投稿记录) | 图+视频 | DM | 4-step/2-phase | 是(SNR 分相→MoE) | 占用"progressive DMD"命名;Wan2.2-A14B/Qwen-Image-20B;无 GAN、data-free;每 phase fake score 从 teacher 重置 | **Risk**:命名与思想空间撞车,必比;精读卡见 3.1 |
| 5 | [GPD: Guided Progressive Distillation](https://arxiv.org/abs/2602.01814) | 2026 | arXiv 02-02,preprint | 视频 | trajectory | 48→6 | **是(步数接力)** | 同基座 Wan2.1-1.3B;明文"student 从上一阶段 ckpt 初始化";纯轨迹回归、无 DMD 无 GAN;VBench 84.04 | **Risk**:轴 B 最近邻,必比;精读卡见 3.2 |
| 6 | [SwiftVideo](https://arxiv.org/abs/2508.06082) | 2025 | 待核实 | 视频 | hybrid | 8→4→2 | 是(范式串联+逐级降步) | 同数据集 OpenVid-1M、Wan2.1-Fun-1.3B;consistency→对抗→跨步对齐三阶段,沿用前一 ckpt;无 DMD | Both:staging 近邻,需划界 |
| 7 | [TMD: Transition Matching Distillation](https://arxiv.org/abs/2601.09881) | 2026 | 待核实 | 视频 | flow-map+DMD | few-step | 否(分层非接力) | Wan2.1-1.3B/14B;flow map + DMD 合流形态;**唯一对 t schedule shift 做过消融的视频 DM 工作**(2-step 84.39 vs 83.44) | Both:轴 A 关键反例 + 竞品 |
| 8 | [SGMD: Score Gradient Matching](https://arxiv.org/abs/2605.30116) | 2026 | ICML 2026(待核实) | 视频 | DM | 4-step | 否 | 攻击 DMD2 fake score 追踪成本(约 3x 加速);t_list {1000,960,889,727} 零论证;代码在 LightX2V 仓库 | Both:效率竞品 + 痛点诊断可引 |
| 9 | [MagicDistillation](https://arxiv.org/abs/2503.13319) | 2025 | 待核实 | 视频 | DM | 4-step | 否 | 记录 vanilla DMD 视频训练崩溃,LoRA fake DiT + W2S 匹配修复;归因于分布支撑不重叠 | Useful:视频 DMD 失败模式证据 |
| 10 | [Alice v1](https://arxiv.org/abs/2605.08115) | 2026 | arXiv 04-27 | 视频 | consistency+DM | 50→4 | 部分(课程沿分辨率/数据轴) | rCM 式目标一步到位 50→4 且自报 VBench 91.2 超 teacher 84.0;削弱"必须分阶段"叙事 | **Risk**:需正面回应 |
| 11 | [AnyFlow](https://arxiv.org/abs/2605.13724) | 2026 | 待核实(NVlabs) | 视频 | flow-map | any-step | 否 | 批评"只为固定少数步蒸馏"的范式(即我们的固定 4-step t_list);on-policy 任意区间转移;1.3B–14B 验证 | **Risk**:固定锚点设计的最强对立面 |
| 12 | [ASD: Adversarial Self-Distillation](https://arxiv.org/abs/2511.01419) | 2025 | ICLR 2026 | 视频 | DM+GAN | 1–2 | 否 | 同基座 Wan2.1-1.3B;L_DMD + 对抗式 n/(n+1) 步自蒸馏——"相邻步数对齐"与步数接力思想相邻,须划界 | Both |
| 13 | [One-Forcing](https://arxiv.org/abs/2605.23458) | 2026 | arXiv WIP | 视频 | DM+GAN | 1(因果 AR) | 否 | **配置与我们高度相近**:Wan2.1-1.3B、GAN 权重 0.03、判别器用 transformer 层特征({21,29} vs 我们 15/22/29)、critic 每 iter/生成器每 5 iter;原句"trajectory-style CD 弱动态、DMD 系模糊帧" | Both:必须显式引用划界 |
| 14 | [Hyper-SD](https://arxiv.org/abs/2404.13686) | 2024 | NeurIPS 2024 | 图像 | consistency(+DM) | 8→4→2→1 | **是(接力)** | "resume the model weights from the previous stage"的图像域接力模板;1-step 阶段含 DMD 式 score distillation | Useful:接力协议成熟证据 |
| 15 | [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142) | 2022 | CVPR 2023(Award candidate) | 图像 | trajectory | 1–4 | **是(两阶段)** | "先吸收 CFG、再 progressive 减半、每轮从前一轮 student 初始化"的正统两阶段模板 | Useful:方法论引用起点 |
| 16 | [Imagen Video](https://arxiv.org/abs/2210.02303) | 2022 | Google 技术报告 | 视频 | trajectory(PD) | 每级 8 步 | **是(PD 减半)** | 视频 progressive distillation 最早应用(级联像素模型;abstract 确认用 PD+CFG,"8 步/级"出自正文与作者推文) | Useful:视频 PD 先例 |
| 17 | [LADD](https://arxiv.org/abs/2403.12015) | 2024 | SIGGRAPH Asia 2024 | 图像 | adversarial | 1–4 | 否 | teacher latent diffusion 生成特征做判别器 + 4-step 无 CFG 部署——我们两个设计点的最直接图像域先例 | Both:判别器设计划界必引 |
| 18 | [Seaweed-APT](https://arxiv.org/abs/2501.08316) | 2025 | ICML 2025 | 图+视频 | adversarial | 1-step | 是(蒸馏初始化+对抗后训) | 纯对抗 1-step 视频;明确记录 batch 256 → mode collapse、1024 不会,视频 LR 5e-6→3e-6 求稳——轴 C 视频侧最近的优化归因证据(GAN 式非 DMD) | Useful:轴 C 关键坐标 |
| 19 | [CDM: Continuous-Time Distribution Matching](https://arxiv.org/abs/2605.06376) | 2026 | 待核实 | 图像 | DM | 1–4 | 否 | 直接批评 DMD 固定离散锚点稀疏监督,用连续随机长度 schedule 取消 t_list,且声称无需 GAN;仅图像 | **Risk**:固定锚点范式的反方论文 |
| 20 | [Align Your Steps](https://arxiv.org/abs/2404.14507) | 2024 | ICML 2024 | 图+视频 | schedule | ~10–40 NFE | 否 | schedule 当一等变量的代表作(含 SVD 视频实验);但只作用于未蒸馏 teacher 的 solver 采样 | Useful:锚点设计引用基座 |

未入表但在正文引用的补充条目(机制相关、非本领域核心):f-distill([2502.15681](https://arxiv.org/abs/2502.15681),f-divergence 统一框架,reverse-KL 是特例)、SiD([2404.04057](https://arxiv.org/abs/2404.04057),ICML 2024)、Few-Step SiD([2505.12674](https://arxiv.org/abs/2505.12674),multi-step DM 目标的"均匀混合"定义 + Zero-CFG/Anti-CFG)、SiD-DiT([2509.25127](https://arxiv.org/abs/2509.25127),DM 目标对 flow-matching 基座开箱可用)、SenseFlow([2506.00523](https://arxiv.org/abs/2506.00523),ICLR 2026,大规模 flow 模型上 DMD 收敛困难 + timestep 重要性再分配)、SANA-Sprint([2503.09641](https://arxiv.org/abs/2503.09641),ICCV 2025)、NitroFusion([2412.02030](https://arxiv.org/abs/2412.02030),CVPR 2025,判别器头池)、Diffusion2GAN([2405.05967](https://arxiv.org/abs/2405.05967),ECCV 2024,扩散骨干判别器)、SF-V([2406.04324](https://arxiv.org/abs/2406.04324))、OSV([2409.11367](https://arxiv.org/abs/2409.11367))、V-PAE([2508.21019](https://arxiv.org/abs/2508.21019),AAAI 2026,判别器复用生成器参数)、APT2([2506.09350](https://arxiv.org/abs/2506.09350),NeurIPS 2025)、Flash-DMD([2511.20549](https://arxiv.org/abs/2511.20549),高噪 DMD/低噪 GAN 分工)、MSD([2410.23274](https://arxiv.org/abs/2410.23274),损失课程接力)、Hierarchical Distillation([2511.08930](https://arxiv.org/abs/2511.08930))、ProReflow([2503.04824](https://arxiv.org/abs/2503.04824),轨迹族 progressive)、PeRFlow([2405.07510](https://arxiv.org/abs/2405.07510),NeurIPS 2024)、Shortcut Models([2410.12557](https://arxiv.org/abs/2410.12557),ICLR 2025 Oral)、MeanFlow([2505.13447](https://arxiv.org/abs/2505.13447),NeurIPS 2025)、Align Your Flow([2506.14603](https://arxiv.org/abs/2506.14603))、pi-Flow([2510.14974](https://arxiv.org/abs/2510.14974),ICLR 2026,批评 DMD quality-diversity 权衡)、CTM([2310.02279](https://arxiv.org/abs/2310.02279),ICLR 2024)、TCD([2402.19159](https://arxiv.org/abs/2402.19159),确认无视频实验)、AnimateLCM([2402.00769](https://arxiv.org/abs/2402.00769),SIGGRAPH Asia 2024 TC)、T2V-Turbo-v2([2410.05677](https://arxiv.org/abs/2410.05677),ICLR 2025)、TDD([2409.01347](https://arxiv.org/abs/2409.01347),CM 目标 timestep 精选)、OSS([2503.21774](https://arxiv.org/abs/2503.21774))、GITS([2405.11326](https://arxiv.org/abs/2405.11326),ICML 2024)、S4S([2502.17423](https://arxiv.org/abs/2502.17423))、DDSS([2202.05830](https://arxiv.org/abs/2202.05830))、Optimized Time Steps([2402.17376](https://arxiv.org/abs/2402.17376),CVPR 2024)、INDIS([2603.17671](https://arxiv.org/abs/2603.17671))、Salt([2604.03118](https://arxiv.org/abs/2604.03118),ECCV 2026 待核实,多步组合漂移)、Adaptive Video Distillation([2603.21864](https://arxiv.org/abs/2603.21864),少步视频退化三模式)、FlashMol([2605.07020](https://arxiv.org/abs/2605.07020),分子域 DMD 的 t schedule 设计,轴 A 跨域反例)、Lip Forcing([2606.11180](https://arxiv.org/abs/2606.11180),视频 DMD 单锚点落点消融)、Embedding Loss([2604.22379](https://arxiv.org/abs/2604.22379),batch 梯度方差理论)、LiveTalk([2512.23576](https://arxiv.org/abs/2512.23576),视频 DMD 崩溃-修复配方,归因数据/初始化)、2D score DM([2412.05899](https://arxiv.org/abs/2412.05899),早期视频 DMD+GAN 先例)。

### 2.3 known-18 清单的补充与更正(只补新信息)

本次对支撑关键判断的条目逐一打开原文核实,更正/补充如下(其余条目无新信息):

- **DMD2**(2405.14867):venue 确认为 **NeurIPS 2024 Oral**(nips.cc 官方页核实)。精读卡见 3.4。
- **SDXL-Lightning**(2402.13929):**无任何会议接收证据**(dblp 仅收录 CoRR),应引用为 ByteDance 技术报告,原清单"arXiv 2024"无误但注意不要写成会议论文。精读卡见 3.3。
- **PCM**(2405.18407):确认 **NeurIPS 2024**;确有视频实验(摘要明文"state-of-the-art few-step text-to-video generator");其"phased"是轨迹分段自洽,无步数接力。
- **MCM**(2406.06890):确认 **NeurIPS 2024**;其动机(公开视频数据画质差→借图像判别器)与我们在 OpenVid-1M 上加 GAN 的动机同构,可引。
- **T2V-Turbo**(2405.18750):确认 **NeurIPS 2024**;确立了"4-step VBench total 超商业模型 + 人评 4-step 优于 teacher 50-step DDIM"的评测叙事模板。
- **TDM**(2503.06674):视频端为 CogVideoX-2B,官方 GitHub 确认 4 NFE 对 100 NFE 教师 25x 加速"无性能退化";其 multi-step 目标用**均匀** t_i = T/K·i,无锚点形状研究。
- **DOLLAR**(2412.15689):VBench 82.57 超 teacher 与 Gen-3/Kling;abs 未报 FVD;venue 仍未见正式接收记录。
- **AnimateDiff-Lightning**(2403.12706):确认逐级压步且 student 收敛后充当下一阶段 teacher(接力);但蒸馏对象仅 motion module,abs 页零量化指标(纯定性报告在 2026 年已不可接受,反面教材)。
- **ADM/DMDX**(2507.18569):补充——其 TTUR 消融显示 fake score 更新 1→8 次仅边际收益(CLIP 35.2557→35.3299)却 2.53x 训练时间;视频端为 CogVideoX-2b/5b 8 步。

> **回填更正(2026-07-06,来自 T2 验收与 planner 代码核实)**:(1) 本报告 2.2 表第 13 行与 6.1 节将 One-Forcing 描述为"transformer 层 {21,29} 特征、与我们配置最接近"——T2 精读并经 planner 联网核实,其判别器宿主为 **fake score backbone**(register-token attention 头),非生成器;相同点(0.03、TTUR 5:1、real-data、无正则)仍成立。(2) 本报告 4.3 节轴 B 收窄表述中"生成器 backbone 特征判别器"一语不准确——planner 核实 FastGen 代码(`fastgen/methods/distribution_matching/dmd2.py`):我们的判别器实为**冻结 teacher backbone 中间层特征 + 可训 multiscale MLP 头**(LADD / teacher-feature 谱系),轴 B 表述应改为"轻量冻结-teacher-特征判别器"。(3) ADM/DMDX venue 经 T2 核实为 ICCV 2025 Highlight。

### 2.4 评估协议专题(回答 Q4)

结论先行:**迁移成本最低且审稿可接受的最小量化协议 = "VBench 6 维子集 + CD-FVD 做日常消融;VBench full 一次做主表;T2VHE 式人评收尾;Dynamic Degree 与多样性指标主动报"**。评测惯例已在 2024–2026 从"FVD+CLIPSIM"(MCM)迁移到"VBench total + 人评 vs teacher"(T2V-Turbo 起);2026 年 Wan 系论文一律以 VBench-T2V 总分为主战场,均值区间 83–85。

1. **主协议:VBench full**([2311.17982](https://arxiv.org/abs/2311.17982),CVPR 2024 Highlight)。标准套件为 946 prompts(社区广泛引用值,官方 prompt 文件未逐条清点,待核实)、每 prompt 5 个视频(temporal flickering 维 25 个),报 Total/Quality/Semantic。不跑 full VBench 就无法与竞品同表——**同设定可直接对标的数字坐标**:CoDMD(Wan2.1-1.3B,50→4)Total 84.46 / DMD 83.38 / rCM 82.81 / teacher(50×2 NFE)83.69;GPD(同基座 48→6)84.04 / teacher 83.92。我们的 4-step 结果需落在 83–85 区间才有竞争力。
2. **低成本消融子协议**(checkpoint 选型、8-step vs 4-step、t_list 消融):VBench 官方 repo 支持对自采视频只评 6 个质量维度(subject/background consistency、motion smoothness、dynamic degree、aesthetic quality、imaging quality);分布指标用 **CD-FVD**(content-debiased,[2404.12391](https://arxiv.org/abs/2404.12391),CVPR 2024:I3D-FVD 偏帧质、奖励静态视频,恰好掩盖蒸馏最常见的运动退化);样本 <2k clips 时加报 **JEDi**([2410.05203](https://arxiv.org/abs/2410.05203),V-JEPA+MMD,16% 样本即收敛)。参考集可用 OpenVid-1M 验证子集(SwiftVideo 先例,与我们训练数据同源)。
3. **人评**:side-by-side 成对二选一(可选 tie)、A/B 随机化,student 4-step vs teacher 50-step,维度取视觉质量/运动质量/文本对齐三问;协议设计引 **T2VHE**([2406.08845](https://arxiv.org/abs/2406.08845),NeurIPS 2024,动态评测模块可降约 50% 成本);prompt 从 EvalCrafter 700 集([2310.11440](https://arxiv.org/abs/2310.11440),CVPR 2024)或 VBench 套件抽 ~100 条。人评规模先例:DMD2 用 128 PartiPrompts × 5 人;CoDMD 用 25 人盲测 pairwise(胜 DMD 81.8%、对 teacher 51.3%)。
4. **必须主动报的防御性指标**:(a) Dynamic Degree / Optical Flow 类运动指标——Phased DMD 实证 DMD2 在 Wan2.2 上 OF 10.26→3.23、DD 79.55→65.45,CoDMD 实证 DMD 的 DD 仅 71.11,这是对 DMD 系的公开攻击线;(b) 多样性指标(DINO 特征 cos 相似度 / LPIPS,Phased DMD Table 3 先例;或 CoDMD 未报多样性这一点本身也可作差异化)——rCM/pi-Flow/Data-Forcing 全部以 diversity 攻击 DMD 系。(c) 可选哨兵:FVMD([2407.16124](https://arxiv.org/abs/2407.16124),运动一致性)。
5. **警示**:Phased DMD 自曝其实验中 VBench 分数与人评趋势**相反**(VBench 上 DMD2 最好、base 最差,人评相反)——不要只堆 VBench,人评不可省;VBench-2.0([2503.21755](https://arxiv.org/abs/2503.21755))评测成本高,列为可选扩展,不进最小协议。
6. 速度指标:teacher 50-step(CFG 双前向,等效 100 NFE)165.24s vs 4-step student 6.59–6.63s ≈ 25x(内部记录,引用前需重读远端 metrics.csv);与 TDM 的"25x 无退化"、CoDMD 的"约 25x"表述量级一致,可同表。

### 2.5 Wan 生态相关工作(登记,交 T3 深查)

按任务书规则:只登记 title/link/一句话状态,不精读、不下结论。这些条目与轴 B/竞争定位直接相关,是 T3 对抗核实重点。

| 条目 | 链接 | 一句话状态 |
|---|---|---|
| CausVid | [2412.07772](https://arxiv.org/abs/2412.07772) | CVPR 2025;DMD 适配因果自回归视频生成的源头 |
| Self-Forcing | [2506.08009](https://arxiv.org/abs/2506.08009) | NeurIPS 2025 Spotlight(据官方 GitHub,待核);自生成 rollout + 视频级损失;续作 Self-Forcing++([2510.02283](https://arxiv.org/abs/2510.02283)) |
| Causal Forcing / ++ | [2602.02214](https://arxiv.org/abs/2602.02214) / [2605.15141](https://arxiv.org/abs/2605.15141) | ICML 2026(据 GitHub,待核);AR teacher ODE 初始化 + Self-Forcing 式 DMD 两段流程 |
| Causal-rCM | [2606.25473](https://arxiv.org/abs/2606.25473) | rCM 向流式自回归蒸馏的开源配方 |
| AccVideo | [2503.19462](https://arxiv.org/abs/2503.19462) | 合成轨迹数据集 + 少步引导 + 对抗;常用对比基线 |
| lightx2v Wan2.1 StepDistill-CfgDistill | [HF 模型页](https://huggingface.co/lightx2v/Wan2.1-Distill-Models) / [step_distill 文档](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html) | 社区部署事实标准:4-step 无 CFG,t_list=[1000,750,500,250]+shift 5;基于 Self-Forcing 的 DMD 变体;**检索摘要提示其配方可能为两段式(先初始化适配再 DMD)——轴 B 的条件性风险,T3 必查** |
| FastVideo / FastWan(-QAD) | [haoailab 博客](https://haoailab.com/blogs/fastwan-qad/) | 3-step 蒸馏 Wan2.1/2.2 + 量化产品线 |
| NVIDIA FastGen 官方博客 | [developer.nvidia.com](https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering/) | 我们所用框架的官方公开成果:Wan2.1-T2V-**14B** 用 DMD2 蒸成 few-step(64×H100,16h,demo 为 2-step);**博客未提渐进步数压缩或中间 checkpoint 初始化**——即官方公开线未覆盖我们的 1.3B 50→8→4 分阶段协议,但对外表述必须说明与上游的关系 |
| 生态动向 | — | SGMD 代码挂在 LightX2V 仓库、Phased DMD 代码在 ModelTC/Wan2.2-Lightning——学术界 DMD 改进正被 lightx2v 生态吸收,T3 做对照时注意 |

---

## 3. 精读卡(5 篇)

### 3.1 Phased DMD(arXiv 2510.27684)——"分阶段 DMD"命名占用者

- **核心问题**:一步 DMD student 容量不足(多样性下降、大幅运动退化);直接展开多步 DMD 又因 backward simulation 全链梯度而显存暴涨,DMD2 的随机梯度截断会使多步"实际退化为 one-step"。目标:无 GAN、data-free、显存接近一步蒸馏的稳定 few-step DMD。
- **机制**:按 SNR 把 [0,1] 分成子区间(4-step/2-phase,对齐 Wan2.2 双 expert 架构),由高噪到低噪逐相训练**独立 expert**;phase k 用冻结的前序 expert 组成 pipeline 生成输入,当前 expert 一步映射到本相终点噪声水平;DMD 梯度的 re-noise t 用 reverse nested interval(t~(t_k,1),终点恒为 1)——消融显示显著优于 disjoint 区间,"高噪端注噪对 DMD 训练至关重要"。fake score **每 phase 从 teacher 重置**、只在子区间训练(推导了无干净 x_0 时无偏的子区间 score matching 损失)。纯 DM 目标、无 GAN、无真实数据;fake:generator 更新比 5:1;生成器用 LoRA(rank 64)。
- **推理形态**:few-step **MoE**——各 expert 按 SNR 子区间依次接管采样步(4 步 2 套生成器权重),无 CFG。
- **实验证据**:Qwen-Image-20B、Wan2.2-T2V/I2V-A14B;t_list {1000,938,833,625}(shift=5,与我们同源);Wan2.2-T2V 上 DMD2:Optical Flow 3.23/Dynamic Degree 65.45%/FVD 763.1,Phased DMD:9.30/82.27%/700.9(base OF 10.26);多样性 DINOv3(Wan2.1-14B):base 0.708/DMD2 0.826/Phased 0.782。作者自指 VBench 与人评趋势相反。64 GPU,batch 64,fake 全参 lr 4e-7,generator LoRA lr 5e-5。
- **对我们的启发**:(a) 两篇独立工作都发现阶段切换时 fake 端重置有益(它每 phase 从 teacher 重置 fake score;我们 8→4 时重置 fake score/判别器)——可互为设计动机支撑;(b) 其"DMD2 梯度截断使多步退化为一步"论点可用来解释为什么 50→8→4 接力优于直接 50→4;(c) reverse nested interval 结论直接可用:我们 4-step 训练的 shifted t 采样应确保高噪端覆盖充分;(d) 它对 DMD2 动态性的实证攻击(OF 3.23)意味着我们必须主动报运动指标。
- **划界(可写进论文)**:它沿 SNR 轴分相、产出多 expert MoE、固定 4 步;我们沿步数轴接力(50→8→4)、单一部署 student、零推理额外开销;它每相 fake 从 teacher 重置,我们生成器跨阶段继承 + 辅助网络重置——两种正交的"分阶段"。
- **不能 claim**:"首次分阶段/phase-wise DMD"、"progressive distribution matching"命名、4-step t_list 形状本身(社区 shift=5 标准配置)、"DMD2 backward simulation 截断退化为 one-step"这一观察。**方法命名必须避开 phased/progressive DMD 字样**。

### 3.2 GPD(arXiv 2602.01814)——轴 B 的视频侧最近邻

- **核心问题**:一次性让 student 跨大步要拟合 teacher 的高曲率轨迹,是视频 few-step 蒸馏质量退化的根源;方案是渐进增大步长 + 在线生成目标。Wan2.1-T2V-1.3B,48→6 步。
- **机制**:纯**轨迹回归**(velocity L2),非 DM。K 个阶段,prediction horizon 每阶段线性 +1;**每阶段 student 从上一阶段 checkpoint 初始化**(Algorithm 1 明文);目标在线拼装:frozen 上一阶段 student 走 k−1 步 + teacher(CFG 从 6.0 线性退火到 1.5)精炼最后 1 步;最后阶段加 latent 频域高频 L2 损失。无 GAN、无 fake score、无真实视频(仅 OpenSora 文本 prompt)。每阶段仅 150 iter,4×A100,batch 4,lr 1e-6。
- **实验证据**:VBench 480p 6-step total 84.04(超 teacher 48-step 83.92);5-step 掉到 83.60(甜点在 6 步);对比 CausVid 3步 83.65、AccVideo 5步 83.28、PeRFlow 6步 82.54;训练成本仅 0.550 GPU-days。主表**无 DMD/DMD2 原方法对比**;**没有 progressive vs 一次性蒸馏的直接消融**。无人评、无 FVD;t schedule 未公开。
- **对我们的启发**:(a) 它是"步数接力在 Wan2.1-1.3B 上有效"的独立佐证,轴 B 的直接引用支撑;(b) 其 CFG 退火提示我们可消融 4-step 阶段的 teacher guidance scale(我们固定 CFG=5);(c) 它没做 progressive vs one-shot 消融——**我们补上"50→4 直蒸 vs 50→8→4"对照就是对它的明确增量**;(d) 频域高频损失是与判别器正交的廉价细节补偿,可备选。
- **划界**:目标函数族(轨迹回归 vs DMD2 分布匹配)、监督来源(无真实数据 vs OpenVid-1M+判别器)、阶段粒度(horizon+1 细粒度课程、每阶段 150 iter vs 两次独立收敛的完整蒸馏)、阶段衔接(整体继续训 vs 只继承生成器 + 重置优化器/fake score/判别器)、schedule 透明度(未公开 vs 明确报告)。
- **不能 claim**:"首个在 Wan2.1(-1.3B) 上做 progressive few-step 蒸馏"、"progressive 多阶段视频蒸馏"本身、"免 CFG 推理/把 CFG 蒸进 student"、"轨迹回归路线在 6 步量级必然质量差"(它 6 步反超 teacher 且成本极低——我们的论证要落在 ≤4 步、多样性、真实数据信号上)。

### 3.3 SDXL-Lightning(arXiv 2402.13929)——阶段接力协议的图像域原型

- **核心问题**:经典 PD 的 MSE 目标在 8 步以下模糊(student 容量不足以匹配 teacher 概率流);纯对抗(SDXL-Turbo)不保 mode coverage。要在 1024px 上做 1/2/4/8 步并给出"样本质量 vs mode coverage"的可控折中。
- **机制(阶段协议)**:128→32 用 MSE(该阶段、且仅该阶段用 CFG=6);随后对抗蒸馏按 32→8→4→2→1 逐级压步;**student 权重跨阶段延续**("Once the student model converges, it is used as the teacher model and the distillation process repeats"),**判别器每阶段重置**("We re-initialize the discriminator at each stage",从预训练 SDXL UNet encoder+midblock 拷贝初始化)。判别器以 x_t 为条件是保概率流的关键;每阶段内先条件判别器(保 flow)、再无条件判别器(放松 mode coverage、修 Janus 伪影);每阶段先 LoRA(rank 64)后 merge 全参续训。1/2 步稳定技巧:训练锚点多于推理锚点({250,500,750,1000})、预测重加噪到多个 t* 判别(权重 5:1:1:1)、1 步切 x0-prediction。
- **实验证据**:COCO 10K,FID-Whole/FID-Patch/CLIP;4-step 22.30/33.52/26.07 vs SDXL 32 步 18.49/35.89/26.48(patch FID 反超 teacher);**无正式人评**("by human preference"仅是断言);无锚点形状敏感性消融。batch 512,64×A100。
- **对我们的启发**:(a) "逐级压步 + student 权重延续 + 判别器每阶段重置"协议与我们几乎逐条对应,是阶段协议的最强图像先例,必引并划归同一家族;(b) 其判别器每阶段从预训练 encoder 重新初始化与我们 fake score 从 teacher 初始化同理;(c) 它报告的阶段间失败模式(对抗+保 flow 约束下的 Janus 语义伪影;跨阶段误差累积→skip-level teacher)提示我们监控 8→4 阶段的语义级伪影(视频中或表现为主体重复/肢体错乱/物理崩坏——与我们第一轮 4-from-8 的"物理规则崩坏"观察吻合,可引用其容量解释);(d) "训练锚点多于推理锚点提升稳定性"与我们 shifted t 采样可类比。
- **划界**:模态(图像 vs 视频 DiT)、目标(轨迹保持型 GAN——其 related work 明确以"需训练负分布 score 模型、动态目标影响稳定性"为由**不采用** score 蒸馏路线 vs 我们 DMD2 主目标 + 对抗辅助 0.03)、判别器(完整 UNet encoder 全量重训 vs 生成器 backbone 特征上的轻量 multiscale MLP)、阶段粒度(五级、先 LoRA 后全参、双判别器目标 vs 两级、全程全参、单一目标组合)。
- **不能 claim**:首创"逐级压步 + 前阶段权重延续 + 每阶段重置判别器"协议、"MSE progressive 少步模糊"的观察与容量解释、"预训练 diffusion backbone 特征做判别器"(LADD/Diffusion2GAN 亦然)、"少步蒸馏免 CFG"、"训练锚点多于推理锚点"。我们能 claim 的边界:该协议与 DMD2 目标的组合在视频域的实例化与消融。

### 3.4 DMD2(arXiv 2405.14867,NeurIPS 2024 Oral)——我们目标函数的来源

- **核心问题**:去掉 DMD 昂贵的 teacher 轨迹回归损失后稳定训练,并把方法扩展到 few-step(最多 4 步),使 SDXL 级 student 达到甚至超越 teacher。
- **机制**:DM 梯度 = teacher score(CFG)− fake score(reverse-KL/VSD 式);**TTUR**:fake score 每 generator 更新 5 次(消融:1 次不稳、10 次收敛慢、5 次最优);**GAN 分支**:判别头寄生在 fake score UNet bottleneck(卷积头),真实图像与生成图像同加噪判别,权重 1e-3(SDXL)/3e-3(ImageNet),真实数据 LAION-Aesthetic 500K;**multi-step + backward simulation**:4-step 锚点 999/749/499/249(**均匀**,无设计原则、无锚点消融),训练输入由 student 自己按推理链路模拟生成(消融:去掉后 SDXL Patch FID 20.86→24.21)。
- **实验证据**:ImageNet-64 FID 1.51(超 teacher 2.32);SDXL 4-step FID 19.32 打平 teacher 19.36(25x 少前向);消融链 3.48(无回归)→2.61(+TTUR)→1.51(+GAN);人评 128 PartiPrompts×5 人。**无任何视频实验;最多 4 步;无任何步数递减分阶段或中间 student 接力机制或讨论**。
- **对我们的启发**:(a) 我们的 fake:gen=5:1 与其完全一致,直接引用其稳定性消融即可,不必重做;(b) backward simulation 消融是我们训练时模拟推理输入分布的直接依据;(c) 其均匀锚点 + 零设计原则正是轴 A 的空隙所在;(d) GAN 权重量级(1e-3~3e-3)与我们 0.03 差一个量级,判别器位置(fake score bottleneck vs 我们生成器 backbone 15/22/29 层)是结构性差异,须在 T2 深挖。
- **划界/不能 claim**:目标函数组合(DM+TTUR+GAN)、backward simulation、TTUR 5:1、"判别器+真实数据超越 teacher"思想、"few-step 无 CFG 打平 teacher"——全部是 DMD2 的贡献,我们只是迁移。我们的增量限定在:视频模态、progressive 接力、非均匀 t_list、生成器特征判别器。

### 3.5 CoDMD(arXiv 2606.21982)——同设定 concurrent 竞品 + 评测模板

- **核心问题**:标准 DMD 的 reverse-KL 梯度是逐坐标的边缘分布匹配,对 batch 内样本间与视频帧间的关系几何(copula 成分)零约束——归因 few-step 视频 DMD 的运动失真("failed camera motion, super slow action")、布局崩坏与过饱和。
- **机制**:标准 DMD(teacher CFG=3.5 − fake score;TTUR 5:1 与我们一致;**不用 GAN**,baseline 表比的是 DMD 而非 DMD2)+ copula 正则:用已有 real/fake score 免费构造 batch 级(B×B)与 frame 级(F×F)cosine 相似度矩阵,以 KL(softmax(τ⁻¹[S_stu−ΔS]) ‖ softmax(τ⁻¹S_stu)) 匹配,τ=0.1、λ=0.1,零额外网络/数据/轨迹。**data-free**(只用文本 prompt)。**单阶段 50→4,student 直接由 teacher 初始化,全文无中间步数阶段、无 checkpoint 接力**。
- **实验证据**:Wan2.1-T2V 1.3B/14B,4-step,832×480、81 帧(与我们完全同规格);VBench Total 1.3B:CoDMD 84.46 / DMD 83.38 / rCM 82.81 / AVD 83.75 / teacher(50×2 NFE)83.69;Dynamic Degree:DMD 71.11 → CoDMD 86.11;人评 25 人盲测:胜 DMD 81.8%、胜 rCM 74.4%、对 teacher 51.3%。batch 128(1.3B),student LR 2e-6 / fake LR 4e-7,32×A100 约 1.5 天。4-step t_list 未给出;无 FVD、无多样性指标。作者含 Wan Team(Alibaba),2026-06-20 提交,典型 concurrent work。
- **对我们的启发**:(a) 评测协议模板:VBench 16 维全表 + Total/Quality/Semantic + 25 人盲测 pairwise(vs DMD/rCM/teacher),teacher NFE 记为"50×2"的写法值得沿用;(b) 其 frame 级正则对 motion 提升立竿见影(DD 71→86),是可叠加到我们 pipeline 的低成本 add-on 消融候选;(c) 它证实 few-step 视频 DMD 失败模式是社区公认问题,可引其归因作 motivation 旁证;(d) 其超参(LR 2e-6/4e-7、TTUR 5:1、CFG 3.5)是我们配置的 sanity check 坐标。
- **划界**:单阶段 vs 我们两阶段接力;无 GAN/data-free vs 我们 DMD2 判别器 + OpenVid-1M 真实数据;正则对象(关系几何)与我们的贡献点(训练调度与阶段协议)正交、可叠加。
- **不能 claim**:"首个把 Wan2.1-T2V 50-step 蒸到 4-step 的 DMD 类工作"、"首次指出 few-step 视频 DMD 的布局崩坏/过饱和/动态迟缓"、"DMD 必须依赖 GAN 或真实数据才能到 teacher 水平"(它 data-free 无 GAN 也超了 teacher Total);我们的 VBench 数字必须与 84.46 同表比较,且须按 concurrent work 惯例标注。

---

## 4. Gap 分析(逐主张裁决;"无先例"类附检索关键词与覆盖范围)

总体检索覆盖声明:8 路 sweep 共执行约 100 组检索串(2024-07 至 2026-07 实时;覆盖 arXiv 全号段至 2606、ICLR/NeurIPS/ICML/CVPR/ICCV/ECCV/SIGGRAPH (Asia)/AAAI 2024–2026 的搜索可达部分),三条轴另由独立对抗核实 agent 各自执行 7–9 组新检索并逐篇打开原文核实(轴 A 20 篇、轴 B 21 篇、轴 C 19 篇原文页)。明确未覆盖:CVPR/ICCV 2026 与各会 OpenReview 完整接收列表的逐条排查(OpenReview 多次被人机验证拦截)、闭源产品内部配方、中文社区蒸馏 LoRA 文档(留 T3)、非 arXiv 渠道(HF 模型卡/技术博客)的系统扫描。

### 4.1 主张"分阶段步数压缩 + 中间 student 接力初始化在视频上无先例"(回答 Q2)——**不成立,须收窄**

直接先例(全部原文核实):
- **图像域,接力协议成熟**:Progressive Distillation(定义即逐轮减半接力)、On Distillation of Guided Diffusion Models(CVPR 2023,两阶段模板:先吸收 CFG 再 PD 接力)、SDXL-Lightning(student 收敛后充当下一阶段 teacher + 判别器每阶段重置)、Hyper-SD(NeurIPS 2024,8→4→2→1,"resume the model weights from the previous stage")、MSD(损失课程接力)。
- **视频域**:Imagen Video(2022,级联像素模型 PD 至每级 8 步)、AnimateDiff-Lightning(2024,motion module 逐级压步接力)、**GPD(2026,Wan2.1-T2V-1.3B,48→6,明文"Initialize student v_θ^k from v_θ^{k−1}")**、SwiftVideo(OpenVid-1M,8→4→2 逐级对齐并沿用前一 checkpoint,consistency/对抗/DPO 目标)。

因此 Q2 的回答是:**"步数接力"这一层在图像与视频上都有直接先例,我们绝不能在这一层 claim 新颖性**;与我们同构的做法(不论目标函数)在视频上存在(GPD、AnimateDiff-Lightning、SwiftVideo),差异只在目标函数族与阶段协议细节。

### 4.2 轴 A(t_list schedule 设计)——**部分支持**

裁决理由(要点):敏感性"现象"已被 TMD(2601.09881)在 Wan2.1-1.3B 视频 DMD 上量化记录(2-step 有/无 shift:VBench 84.39 vs 83.44;无 shift 时 t_dmd 出现 severe mode collapse),但仅 1–2 步、单一 shift 标量、附录级;"缺少设计原则"的绝对表述被 CDM(图像域,连续 schedule 取消锚点)、FlashMol(分子域,DMD student 的 EDM ρ 形状消融)、Phased DMD(视频域,SNR 子区间原则 + nested interval 消融)、Flash-DMD(按噪声区间分配损失类型)部分抢占。但从严检视后仍有真实空隙:DMD2(均匀,无消融)、TDM(均匀)、DOLLAR(均匀)、SGMD(极端高噪密集,零论证)表明**视频 DM 蒸馏的 t_list 实践高度分裂且几乎从不被论证;没有任何工作在 4–8 步视频 DMD student 上把"高噪密集程度"当显式变量做逐锚点形状消融或给出可迁移准则**。另注意:我们的 t_list 本身 = 社区 shift=5 标准配置(Phased DMD {1000,938,833,625}、lightx2v [1000,750,500,250]+shift5 warp 后同值、Self-Forcing GitHub issue #38 用户报告关闭 warp 后质量退化——二手来源待核实),选用该形状零新颖性。

主要反例与差距:TMD(high;差:1–2 步、单标量、无形状研究)/ CDM(high;差:图像、取消而非设计锚点)/ Phased DMD(high;差:设计训练区间而非推理锚点)/ FlashMol(medium;分子域、密集方向相反)/ Flash-DMD、Lip Forcing、TDD、AYS/OSS/GITS/S4S/INDIS(medium-low;solver 侧或 CM 目标)。

对抗核实执行过的检索串(如实记录):
```
"timestep schedule" ablation "distribution matching distillation" video few-step student sensitivity
few-step video diffusion distillation "timestep shift" ablation Wan denoising schedule
arxiv 2026 timestep selection few-step distillation video diffusion schedule design principle
Self-Forcing CausVid video DMD denoising timesteps [1000 750 500 250] shift 5 schedule choice
"noise schedule" OR "timestep schedule" design principle "few-step" distilled student "high noise" dense video generation 2025 2026
arxiv timestep anchor ablation few-step DMD student quality "uniform" vs "shifted" schedule
DMD distillation ablation "shift" one-step two-step VBench video timestep shifting gamma
```
(另有 sweep-6 的 13 组 schedule 检索串,覆盖 solver 侧谱系 2022–2026 全线与 Wan shift 惯例。)

**收窄后可主张**(轴 A 最强表述):在 4–8 步渐进式 DMD2 视频蒸馏(Wan2.1-T2V-1.3B,50→8→4)设置下,首次把推理锚点 t_list 形状(均匀 vs 不同高噪密集度)当显式实验变量做系统消融,量化其对质量与 mode collapse 的影响,并给出视频 DM 目标下锚点选择的经验准则。不得声称"敏感性未被报道"(TMD 已报道)或"文献完全缺少 DM 目标 schedule 设计原则"(CDM/FlashMol/Phased DMD 在图像/分子/训练区间层面均有)。我们已有 `_step8_normalize`(均匀 t_list)消融数据点可直接纳入。

### 4.3 轴 B(progressive DMD2 for video,中间 student 接力)——**部分支持**

裁决理由(要点):对 21 篇原文逐一核实后,**未找到与主张完全同构的先例**——"每阶段都以 DMD2(reverse-KL + GAN 判别器)为目标 + 沿步数轴 50→8→4 + 4-step 生成器由 8-step student checkpoint 接力初始化 + 优化器/辅助网络重置"的公开视频配方不存在。但组合空间已被三面挤压:(1) GPD 命中"同基座 + 步数接力"但目标是轨迹回归;(2) DMD 系视频工作(CoDMD/Data-Forcing/AVD/AMD)全部单阶段直蒸;Phased DMD 沿 SNR 分相且 fake model 明文从 teacher(而非前相)重置,与步数接力正交;RMD(2603.06136)的 multi-stage 沿分辨率轴;(3) ASD 在同基座上做 DMD + 相邻步数(n/n+1)对抗自蒸馏,但单一 student 无接力;Hyper-SD/AnimateDiff-Lightning/SwiftVideo 证明"递减+接力"是成熟模板。残余风险:lightx2v 生态检索摘要出现"初始化适配 + DMD"两段式表述,若其公开配方含步数递减接力,主张进一步削弱——**已登记为待 T3 核实的条件项**。

主要反例与差距:GPD(high;差:无 DM/GAN)/ SwiftVideo(medium;差:consistency+对抗+DPO)/ ASD(medium;差:无接力)/ AnimateDiff-Lightning(medium;差:对抗目标、仅 motion module)/ Hyper-SD(medium;差:图像、consistency 主目标)/ Phased DMD(medium;差:SNR 分相、初始化策略相反)/ Alice v1(medium;差:课程沿分辨率/数据轴,DM 仅作正则)/ Data-Forcing(low;损失课程接力、步数不变)/ Imagen Video、Guided Distillation、RMD、MSD(low)。

对抗核实执行过的检索串(如实记录):
```
progressive distribution matching distillation video diffusion "8 steps" "4 steps" initialize student checkpoint
"DMD2" progressive step distillation video Wan2.1 staged 8-step 4-step initialization
"step reduction" OR "step curriculum" distribution matching distillation video generation intermediate student checkpoint 2025 2026
video diffusion distillation "progressive" DMD "reverse KL" 50 steps 8 steps 4 steps checkpoint relay
"iterative halving" OR "halve the number of sampling steps" distribution matching distillation DMD video student initialization
Magic 1-For-1 DMD2 step distillation stages 4-step video diffusion initialization
DOLLAR few-step video generation variational score distillation consistency stage-wise 8 step 4 step
"8-step" student "as the teacher" OR "initialized from" "4-step" video diffusion DMD distillation two stage
progressive distillation Wan2.1 text-to-video 2026 "distribution matching" multi-stage 50 to 8 to 4 steps
```
(另有 sweep-5 的 12 组 progressive/staged 检索串。)

**收窄后可主张**(轴 B 最强表述,直接可用):已有工作要么沿步数轴接力但用轨迹回归/consistency/纯对抗目标(Imagen Video、AnimateDiff-Lightning、Hyper-SD、GPD、SwiftVideo),要么以 DMD/DMD2 为目标但单阶段直蒸、或沿 SNR/分辨率轴分相且各相从 teacher 重置(CoDMD、Phased DMD、RMD);据我们所知,尚无工作把 DMD2 式分布匹配 + 生成器 backbone 特征判别器作为每一阶段的统一目标,沿 50→8→4 步数轴接力——用 8-step 中间 student 最优 checkpoint 仅初始化 4-step 生成器、同时重置优化器与 fake score/判别器——并在公开的 Wan2.1-T2V-1.3B 与 OpenVid-1M 上给出完整可复现配方。我们不宣称 progressive 步数蒸馏或 DMD2 视频加速本身为新。(此表述以 lightx2v 两段式配方的 T3 核实结果为条件。)

### 4.4 轴 C(优化稳定性归因)——**部分支持**

裁决理由(要点):文献侧**没有任何工作在少步视频 DMD 上做过"LR × 有效 batch × 每锚点有效更新量"的受控归因**,该空间真实存在;但主张的强因果表述("质量瓶颈**主要**在优化稳定性")与近似先例正面冲突:(1) Seaweed APT 在视频上给出 batch 256→mode collapse / 1024 不会、视频 LR 5e-6→3e-6 求稳的受控证据(GAN 式非 DMD);(2) fake score 更新率消融已被反复做过(DMD2 图 9:1/5/10;SenseFlow:5/10/20 均震荡需 IDA;Flash-DMD:1–2 次已够;ADM:1→8 边际收益不值),全部或主要在图像域;(3) 归因竞争强:在与我们相同的 Wan2.1 50→4 设定上,CoDMD/AVD 把失败归因于 reverse-KL 逐坐标目标,并在**不动优化超参**的情况下用目标级修复见效;rCM 明言 mode collapse"是目标函数根本属性而非调参问题";MagicDistillation 归因分布支撑不重叠;LiveTalk 的视频 DMD 崩溃-修复配方归因数据/初始化/调度。"每锚点有效更新量"是三因素中最新颖的一个——现有消融均以 fake score 更新率为对象,无人以生成器每锚点梯度更新量为变量。**硬前提**:我们内部证据当前 LR/batch/GPU 数三因素混淆(T0 0.3 节),不补受控实验则此轴只能降级为"可复现配方"素材,放弃"归因"措辞。

对抗核实执行过的检索串(如实记录):
```
training instability distribution matching distillation video diffusion learning rate batch size
fake score network update ratio ablation TTUR video DMD distillation
batch size sensitivity diffusion model distillation few-step collapse
learning rate few-step video diffusion distillation collapse recipe stability
"gradient accumulation" OR "effective batch" video DMD distillation mode collapse ablation
per-timestep anchor update allocation few-step distillation training budget ablation diffusion
DMD2 video Wan distillation "training recipe" hyperparameter sensitivity failure analysis 2026
importance sampling fake score learning 4-step video generator training stability
Phased DMD few-step distillation subintervals video Wan
```

**收窄后可主张**(轴 C 最强表述,前提 = 补齐受控实验):在蒸馏目标、数据与架构固定的前提下,给出少步视频 DMD 蒸馏中优化配置的首个受控归因研究——在同一开源 T2V 模型上分别隔离 LR、有效 batch、每锚点有效梯度更新量,证明它们各自独立决定训练是否坍缩与少步质量上限,并给出可复现失败-修复配方;结论不是否定目标本性归因,而是证明相当部分被归因于 reverse-KL 的少步退化在正确优化配置下可消除或显著推迟——优化稳定性是与目标设计正交且被系统性低估的一阶因素。

### 4.5 我们的谱系归属与必比先例(回答 Q5)

**归属**:distribution-matching 系(DMD2 完整配方:reverse-KL + TTUR + GAN + backward simulation)× progressive 步数接力维度(语义 c),视频模态,固定非均匀 t_list、无 CFG 部署。与 consistency/flow 族是**互补**关系(rCM/SwiftVideo/TMD 的合流趋势反而证明 DMD 项是公共组件);与 2026 年 Wan 系 DMD 变体(CoDMD/SGMD/Data-Forcing/Phased DMD)是**竞争**关系,竞争焦点在"改目标函数"(它们)vs"改训练调度/阶段协议"(我们)——两条改进轴正交,这是我们故事的立足点。

**必比先例(5 个,实验或 related work 必须正面对照)**:
1. **DMD2**(NeurIPS 2024 Oral)——我们目标函数的直接来源,一切增量都以它为基线定义;不比它就没有"我们改了什么"。
2. **CoDMD**(2606.21982)——同基座同步数同任务的 concurrent 竞品,VBench 84.46 是我们 1.3B 4-step 结果的硬坐标。
3. **rCM**(ICLR 2026)——GAN-free 竞争路线在 Wan 视频规模的代表,其对 DMD2 的 diversity/mode collapse 批评必须正面回应。
4. **Phased DMD**(2510.27684)——"分阶段 DMD"命名与思想空间占用者,不划界会被审稿人直接判撞车。
5. **GPD**(2602.01814)——同基座步数接力先例,轴 B 的 novelty 表述以与它的目标函数差异为生命线;且它缺 progressive vs one-shot 消融,是我们的直接增量点。

次级对照(视篇幅取舍):Data-Forcing、SwiftVideo、TDM、SDXL-Lightning(协议引用锚点)、One-Forcing(配置最接近的 DMD+GAN 视频工作)。

---

## 5. 对投稿叙事的可用表达(5 条,可直接使用)

1. **定位句**:"We revisit few-step video diffusion distillation from the training-schedule axis rather than the objective axis: keeping the full DMD2 recipe (reverse-KL distribution matching, TTUR, and a lightweight generator-feature discriminator) fixed at every stage, we distill Wan2.1-T2V-1.3B along a step-count relay (50→8→4), where the 4-step deployment student inherits only the generator weights of the best 8-step intermediate student while the optimizer, fake score network and discriminator are re-initialized."(与 4.3 收窄表述配套;避开 phased/progressive DMD 字样。)
2. **与两类近邻的双向划界句**:"Prior staged distillation either relays student checkpoints across step counts with trajectory-regression or adversarial objectives (SDXL-Lightning, Hyper-SD, GPD), or applies distribution matching within SNR sub-intervals producing MoE experts with per-phase re-initialization from the teacher (Phased DMD); our recipe is the missing quadrant — a single deployed student relayed across step counts under a distribution-matching objective."
3. **失败模式坐标句**:"Community evidence now attributes complementary failure modes to the two dominant families — weak dynamics for trajectory/consistency-style distillation and blurriness/oversaturation/diversity collapse for DMD-style objectives (One-Forcing; Data-Forcing; CoDMD; Phased DMD's Optical-Flow drop of DMD2 from 10.26 to 3.23) — motivating our choice of a distribution-matching main objective with a deliberately weak (0.03) discriminator and a staged curriculum, and our protocol of reporting Dynamic Degree and diversity metrics up front."
4. **progressive 必要性论证句**(需消融支撑):"Phased DMD observes that DMD2's stochastic gradient truncation effectively collapses multi-step distillation towards one-step behaviour; our 50→8→4 relay bounds the per-stage teacher-student gap instead, and we validate the necessity of the intermediate 8-step stage with a controlled 50→4 vs 50→8→4 comparison — an ablation that no existing staged video work (including GPD) reports."
5. **评估承诺句**:"We evaluate with the field's de-facto protocol (full VBench Total/Quality/Semantic with per-dimension breakdown, side-by-side human preference against the 50-step teacher following T2VHE), and additionally report motion (Dynamic Degree/FVMD) and diversity metrics that recent work shows are precisely where DMD-family students degrade — together with CD-FVD rather than I3D-FVD, whose content bias would otherwise mask motion degradation."

红线提醒(不能说的):不能说首创 staged/progressive 蒸馏、步数接力、DM+GAN 混用、免 CFG 部署、backward simulation、TTUR、t_list 形状本身;不能说"8-step 中间阶段必要"——在补出 50→4 直蒸对照之前这只是假设;不能忽略 CoDMD/rCM/GPD 而自称 SOTA。

---

## 6. 对后续调研与实验的建议

### 6.1 给 T2(组件近邻)的锚点

- **判别器设计专题**:LADD(teacher 生成特征)、Diffusion2GAN(扩散骨干 multi-scale 判别器)、DMD2(fake score bottleneck 头)、One-Forcing(**Wan2.1-1.3B、层 {21,29} 特征、GAN 权重 0.03、TTUR 同构——与我们配置最接近,T2 必须精读**)、V-PAE(判别器复用生成器参数,AAAI 2026)、NitroFusion(判别器头池 + refresh 防过拟合)、Flash-DMD(冻结 SAM 骨干 + 高低噪损失分工)、SF-V(spatial-temporal 头)。核心问题:生成器 backbone 特征(我们/One-Forcing/V-PAE)vs fake score 特征(DMD2)vs 独立骨干(ADD/OSV/Flash-DMD)三条路线的取舍与消融证据。
- **fake score 稳定化专题**:SenseFlow(IDA/ISG)、MagicDistillation(LoRA fake DiT)、SGMD(teacher stop-gradient Fisher 目标,3x 加速)、Phased DMD(子区间 score matching + 每相重置)、Few-Step SiD(multi-step DM 目标的"均匀混合"定义,对照 DMD2 backward simulation)、f-distill/SiD/SiD-DiT(目标函数理论谱系)。
- **可叠加组件候选**:CoDMD 关系正则(零额外网络)、GPD 频域高频损失与 CFG 退火、Phased DMD reverse nested interval(训练 t 高噪端覆盖)。
- **不要重复调研**:机制族谱系格局、staged 三语义分类、评估协议选型、Wan 生态存在性登记——本报告已覆盖;T2 直接从组件粒度切入。

### 6.2 给 T3(novelty 对抗核实)的锚点

- **最高优先**:lightx2v step_distill 配方是否含"步数递减 + student checkpoint 接力"(轴 B 收窄表述的条件项);NVIDIA FastGen 官方线(14B DMD2)与我们 1.3B 50→8→4 的关系表述;CausVid/Self-Forcing(++)/Causal Forcing(++)/Causal-rCM/AccVideo/FastWan 逐一核实机制与 t_list。
- **venue 核实清单**(本报告标"待核实"者):Phased DMD(OpenReview forum zzJTo7ujql 被验证墙拦截;CVF 检索无结果)、SGMD(ICML 2026?)、Salt(ECCV 2026?)、SwiftVideo、TMD、GPD、CoDMD、MagicDistillation、AnyFlow、CDM、Data-Forcing、Alice v1、f-distill、OSV、SF-V、UFOGen。
- **数字复核清单**:Seaweed APT 的 batch 消融(其图表)、DMD2 图 9、LiveTalk 第 3 节崩溃分析(PDF 超抓取限制未读全文)、TDM 视频端 VBench 数字(80.91→81.65 出自二手)、VBench 946 prompts 原始出处、Alice v1 的 VBench 91.2(自报数字异常高,需查协议)。
- **可复用检索词组合**:见 4.2/4.3/4.4 的检索串列表;Wan 生态另加 `Wan2.1 4-step LoRA distill site:huggingface.co`、`Wan2.2-Lightning ModelTC recipe`、`TurboDiffusion Kijai Wan distill` 等社区面词组。

### 6.3 实验建议(按对轴的支撑排序)

1. **50→4 直蒸 vs 50→8→4 受控对照**(轴 B 的生死消融;GPD/CoDMD 都没做;同预算同数据同 t_list,只变路径)。
2. **t_list 形状消融矩阵**(轴 A):4-step 与 8-step 各做 均匀 / shift=5 / 更高噪密集 三档;`_step8_normalize`(均匀)已是现成数据点,补齐另两档即可成表;同时对照 lightx2v 名义 [1000,750,500,250](警惕 shift 换算——名义均匀经 shift=5 后即我们的非均匀值,对比前必须换算到同一坐标系)。
3. **轴 C 三因素受控**:先修复现有混淆(LR/batch/GPU 数)——固定其二、扫其一;"每锚点有效更新量"用 8-step vs 4-step 在同 iter 数下的对照天然构造。不补此实验则轴 C 从论文主张中撤下。
4. **量化评估先行**:对现有 checkpoint(4-step 基线 0001000、8-step lr_original 0002500、第二轮 4-from-8 全部 5 个)先跑 VBench 6 维子集 + CD-FVD,把"肉眼结论"全部转成数字;主表阶段再跑 full VBench + 人评。
5. **低成本借鉴**:训练 t 采样确保高噪端覆盖(Phased DMD nested interval 结论);4-step 阶段 teacher CFG 退火消融(GPD);CoDMD 关系正则作为 add-on 消融。

---

## 附:本报告的证据边界

- 所有 2024-07 之后条目均经实时检索,高相关条目已打开 arXiv abs/HTML 原文页核实;凡未打开原文或出自二手来源者已逐处标注"待核实"。
- 精读 5 篇的关键数字直接取自原文 HTML;个别页面(OpenReview、部分 PDF)被验证墙/大小限制拦截,已在对应位置注明。
- 本报告为谱系级初判;三条轴的终裁、Wan 生态深查、venue 复核均属 T3。
