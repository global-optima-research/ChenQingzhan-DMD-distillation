# 主线 idea 候选:少步视频 DMD 的 corrective-term 受控研究(度量空间 + 配对 + GAN 边界条件)

- 生成:2026-07-11;来源:PFM/LiveEdit 精读(T3 §7)→ 5 路占位对抗检索 + 1 红队压测(6 agent,~33 万 tokens,178 次检索/原文抓取)
- 终裁:**修正版故事可立为主线,评级"高风险但每个分支结局均有退路";原始形态("R_φ 作设计准则 + where-you-measure 泛化叙事")不可立**
- 激活前置 gate:E5(零训练成本)+ E1(单个 config 训练)先出结果,再决定主线激活或降级

---

## 1. 占位裁决总表(5 路 sweep,全部"部分占据,有明确空隙")

| 组件 | 裁决 | 幸存空隙 | 关键占据者/近邻(必引划界) |
|---|---|---|---|
| (b) R_φ 型先验准则选判别器空间/层 | 部分占据 | **零占据**:R_φ 用于判别器/teacher 层选择 + 与蒸馏退化(DD/多样性)关联;PFM 仅 post-hoc、仅回归空间、明文留作 future work | PFM([2607.03524](https://arxiv.org/abs/2607.03524),R_φ 定义权);Vision-aided GAN([2112.09130](https://arxiv.org/abs/2112.09130),GAN 域 linear-probe 准则先例,**必须纳入为对照准则**);iREPA([2512.10794](https://arxiv.org/abs/2512.10794))、iFID([2603.05630](https://arxiv.org/abs/2603.05630))占据"性质→质量→选择"范式(非对抗/非蒸馏) |
| (c) 特征源受控对比 | 部分占据 | **视频域零受控**;图像域仅两两单点(SenseFlow Table 3 fake-score vs +VFM;ADM SAM vs DINOv2) | Projected GAN(特征网络消融,GAN 域);ADD(DINO 内部消融);LADD/SDXL-Lightning/NitroFusion(只有论证无对比);Teacher-Feature Drifting([2605.07327](https://arxiv.org/abs/2605.07327),teacher 层组合消融,图像/非对抗——层选择新颖性只剩"视频+对抗+准则");APT(可训骨干深度/多层头消融——"头位密度无人消融"作废) |
| (d) 感知空间回归修复 | 部分占据 | 视频 DMD 内**回归项度量空间**(latent vs teacher 层 vs 感知)无人受控对比 | DMD1(LPIPS 回归原点);Diffusion2GAN(E-LatentLPIPS);**AVD([2603.21864](https://arxiv.org/abs/2603.21864),DMD2+Wan2.1+4步 latent 自适应回归修复 oversaturation/temporal collapse——功能位已占,必进 baseline)**;T2V-Turbo(VFM reward 形式);PFM(免蒸馏形式) |
| 视频原生 VFM 判别器 | 部分占据 | InternVideo2/V-JEPA/VideoMAE 作判别器空间零占据;"同一 VFM 空间既监督又审计"的 Goodhart 论述无人系统做 | OSV([2409.11367](https://arxiv.org/abs/2409.11367),逐帧 DINOv2 判别器 + latent 上采样免解码技巧——外部路线代表/可借工具);JEDi/CD-FVD(审计空间已确立);Video GRPO reward-hacking([2511.19356](https://arxiv.org/pdf/2511.19356)) |
| (a) "度量空间设计轴"框架 | 部分占据 | 框架级无人统一(Uni-Instruct/f-distill 只统一散度形式,与空间轴正交) | Drifting([2602.04770](https://arxiv.org/abs/2602.04770) Table 3 同目标多空间消融,图像);TFD("depends heavily on the representation space"原话);CMMD/FVD-content-bias(评测侧已占"空间重要"泛论) |
| (t,ε) 配对消融 | — | **仍独家**(sweep 3 + 红队复核均未发现占据者);ADM 的 identical-t 配对是设计原则非消融 | ADM(identical-(t−Δt) 设计);Seaweed-APT(判别 t 工程化);表述须写死为"DMD2 判别器目标内 shared-(t,ε) vs independent 的受控消融 + 梯度方差机制" |

## 2. 红队终裁与致命项修复

红队评级:**勉强可立(高风险)**,13 项攻击全部可幸存,但故事必须重构:

1. **A1(致命→已修复)R_φ 理论跳跃**:R_φ 是 L2 回归"均值解"惩罚度量,对抗目标无 mean-attractor;LADD 实践反转 R_φ 排序(从 DINOv2 像素空间搬回 latent 特征反而更好)。**修复:R_φ 从"设计准则"降级为"被检验的候选 probe 之一"**,与 Vision-aided GAN 的 real/fake linear-probe separability、判别器精度轨迹、生成端梯度 SNR 并列,命题改为 probe-agnostic 的"哪个便宜 probe 能预测蒸馏结局"(probe-then-verify)——任何方向的结果都可报告。
2. **A2 PFM 自身反证**:PFM 最优感知空间下 DD 仍降(0.319<0.379)。修复:DD/多样性修复不承诺来自空间选择,DD 是**被测 outcome**;修复来源押在 (t,ε) 配对、GAN 权重、Data-Forcing 式修正项等臂上。
3. **A5/A6 预算与统计效力**:全矩阵 24 run 不可行;n=3-4 个空间做"相关性"是轶事。修复:**单因子协议 6-8 run**;放弃"相关性定律",改 case study + 机制证据链。
4. **A8/A9 双活扣(最大风险)**:特征源换臂可能无差异(StyleGAN-XL 先例,待核实);GAN-0 臂可能打平(Data-Forcing 同基座已发表"GAN 无明显收益、动态度反降")。修复:GAN-free 臂与 Data-Forcing 臂升为**正式对照臂**,研究问题升维为"**4-step 视频 DMD 中哪种 corrective term、施加在哪个空间,能恢复 DD/多样性;GAN 项在什么条件下仍有增量**"——GAN 赢/输/平三种结局都有话说。
5. **A12 特征源轴混淆**:三臂输入空间/条件/梯度路径全不同,因果表述会被否。修复:主对比限定**共享接口臂**——frozen teacher 层集 vs frozen fake-score-init 网络层集(同 latent+t 输入、同 MLP 头、匹配 FLOPs)+ 同骨干内层深度轴;外部编码器臂标 exploratory。
6. **A7 叙事内耗**:relay 与判别器科学双头并列必被问"你到底是什么论文"。修复:单一主线"诊断→杠杆→交付",**relay 降为 Experimental Setup(半页)**——它提供干净 4-step 起点与"仅继承生成器"的受控性,本身就是消融纪律的一部分。
7. **A10 组件科学门槛**:analysis-only 需 EDM/Vision-aided GAN 级系统性,我们达不到。修复:**recipe-with-analysis**——头条必须是"在 DD/多样性上优于 DMD2 默认、且与 GAN-free 正面比过的 4-step 配方",受控分析是解释层。

## 3. 幸存版故事(可立的主线)

> **题面**:少步视频 DMD 公认退化 = diversity 塌缩 + Dynamic Degree 下降;DMD2 系配方靠一个 GAN corrective term 缓解,但该项的三个设计自由度——**判别器度量空间**(frozen teacher 层集 vs fake-score 层集,层深度集合)、**real/fake 的 (t,ε) 耦合**、**GAN 权重(含 0)**——在视频 DMD 中从未被受控归因,而 GAN-free 路线(rCM/Data-Forcing/TFD)已宣称该项可弃。
> **主张一(受控归因)**:在受控的 Wan2.1-T2V-1.3B 50→8→4 relay 的 4-step 阶段,单因子改判别器接口内的度量空间与 (t,ε) 耦合,以 DD/CLIP-diversity/camera-pose diversity/VBench 为结局,用机制诊断(判别器精度轨迹、separability、生成端梯度 SNR/方差)解释效应;shared-(t,ε) vs independent 为文献空白的独家消融(定位:DMD2 判别器目标的 variance reduction)。
> **主张二(probe-then-verify)**:把 PFM 的 R_φ 与 Vision-aided GAN 的 linear-probe separability 作为两个候选便宜先验,在 teacher 各层(matched t)离线计算,检验谁能预测蒸馏结局;R_φ 预测失败本身即"回归空间准则不迁移到对抗蒸馏"的可报告结论。
> **主张三(配方交付)**:产出 4-step 配方,在 DD/多样性上优于 DMD2 默认(15/22/29 + 独立采样),并与 GAN-0 臂及 Data-Forcing 式臂正面比较,给出"GAN corrective term 何时仍有增量"的边界条件。
> relay 仅作 Setting;"Where you measure matters" 仅作章节 motif,不作普适主张。

**与 relay 主线的关系**:合并、分主次——主 = corrective-term 受控研究 + 4-step 配方;次 = relay 作 setting。**降级路径**(若 E1 显示 GAN-0 打平):主线回落为"relay 配方 + GAN corrective term 边界条件与失效分析",投中档会议或并入更大工作;measurement-space 收缩为一节。

## 4. 实验计划(按红队优先级,含 gate)

| # | 实验 | 成本 | 作用 |
|---|---|---|---|
| **E5** | 离线 probe:现有 teacher/8-step/4-step checkpoint 上,对 15/22/29 及备选层集(matched t)算 separability + 自实现 midpoint 式 off-manifold 度量 | **零训练成本,立即可做** | 产出层排序作预注册预测;PFM 未测视频空间 R_φ,我们即首次 |
| **E1** | 4-step relay 阶段 GAN 权重 {0, 0.03} | 纯 config | **Gate:决定判别器故事是否存在**;同时直接检验 Data-Forcing 结论在 relay 设置下是否复现 |
| **E2** | shared-(t,ε) vs independent 消融 + 梯度方差记录 | config/小 patch | 独家消融点,机制化表述 |
| **E3** | 共享接口特征源换臂(teacher 层集 vs fake-score 层集,匹配 FLOPs) | 中等代码量 | "where you measure"唯一干净的主对比 |
| **E6** | Data-Forcing 式 teacher-score-at-real-data 臂(其论文称一行改动) | config/小 patch | GAN-free 修复的 head-to-head 防御;若 100-300 步微调即打平,提前转向 |
| **E4** | teacher 层深度集合消融(15/22/29 vs 浅集 vs 深集) | config 层 | 与 E5 probe 排序对照,构成 probe-then-verify 闭环 |
| **E7** | 外部编码器臂(InternVideo2 或 DINO,exploratory) | 高(需解码;可借 OSV latent 上采样技巧) | 只讨论不进主表;先做显存/吞吐可行性测试 |

评测固定:VBench 6 维子集 + DD + CLIP-diversity + camera-pose diversity(与 Data-Forcing 对齐)+ 固定种子/数据顺序;跨空间审计披露(训练空间≠评测空间:JEDi/V-JEPA、CD-FVD/VideoMAE-2)。

## 5. 必引新增与红线补充(相对 T3 清单)

- **新增必引**:Vision-aided GAN(准则先例)、Projected GAN(特征网络消融)、ADD/LADD(特征源更迭谱系)、Teacher-Feature Drifting、AVD(2603.21864,回归修复功能位)、OSV(外部路线代表)、Drifting(2602.04770)、SenseFlow(已在清单,补 Table 3 两两对比角色)、T2V-Turbo/VADER/Reward Forcing(VFM 监督 reward 形式)、Embedding Loss(2604.22379)、RDM(2607.02375)、Uni-Instruct(散度轴统一,正交划界)、MoGAN(2511.21592,motion 判别器,待核实)。
- **红线补充**:不得说"首次对比特征空间"(Drifting/TFD 已做,图像);不得说"判别器层选择全文献无人论证"(TFD 层组合消融、APT 深度消融存在)→ 改"视频少步对抗蒸馏中冻结 backbone 的层位选择无受控证据,亦无可迁移准则";不得说"首次把 VFM 特征引入视频蒸馏"(T2V-Turbo reward 形式);"头架构消融"维度放弃(SF-V/POSE/Taming DiT 已做),矩阵中头结构固定。
- **时间风险(高)**:PFM 结论明文把"perceptual supervision spaces 深入刻画"挂为 future work,京东组或第三方跟进概率高;2026 年 3-7 月 AVD/TFD/RDM 三篇已从不同方向逼近。**E5+E1 应尽快出结果占位。**

## 6. 待核实(引用/动笔前)

- StyleGAN-XL"多数特征网络 FID 相近"原表(影响 E3 效应量预期)— 检索摘要级,待核实
- SenseFlow Table 3 是否同预算严格受控;其附录有无 VFM 规模消融(两来源矛盾)
- Seaweed-APT 判别器层号(16/26/36)与 timestep ensemble 细节;SF-V/OSV 判别器细节(搜索摘要级)
- MoGAN(2511.21592)全文;rCM OpenReview 状态(验证墙)
- SGMD/SwiftVideo"无感知损失"结论为中置信度(PDF 提取不完整)
- DMD1 SD-latent 实验的 LPIPS 是否先解码(附录未读)
- 2606.15553(Drifting+RAE)未打开——若其把 drifting 蒸馏与表征空间结合,进一步压缩组件 (a)
- PFM 被引复查(Semantic Scholar 429,web 旁证为零);一周后复扫 "perceptual regression video distillation"
