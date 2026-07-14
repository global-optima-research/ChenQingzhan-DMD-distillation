# T2 调研报告:DMD2 方法组件近邻盘点

- 写作日期:2026-07-06(所有"近两年/最新"判断以此为基准)
- 任务书:`research/T2_task_brief.md`;上游:`research/T0_project_analysis.md`(方法)、`research/T1_video_fewstep_distillation_landscape.md`(谱系,已验收)
- 调研方式:4 路组件检索(multi-step DM 构造 / 蒸馏内锚点 / 判别器设计 / 训练稳定性)+ 5 篇精读(One-Forcing、Seaweed-APT、Few-Step SiD、SenseFlow、ADM/DMDX,与 T1 五篇零重复)。共约 60 组检索串、40+ 篇原文页核实;SenseFlow 精读含官方代码核对。检索词明细见第 4 节。
- 结论定位:组件层初判,终裁属 T3。

> **T3 §6.1 回填更正(2026-07-14)。** 以 T3 终裁为准:(1) **`gan_use_same_t_noise=True` 是上游 WanT2V 出厂值**(全部 15 个公开实验配置均 True),本报告 4.4 planner 注记"方法默认 False、我方主动配置 True"在配方层面误导——幸存主张 D 只能表述为"论文与公开实现均无该配对的单独消融",不得暗示设计为我方提出;(2) 全部单阶段超参(t_list、0.03、层 15/22/29、CFG 5、TTUR 1:5、lr 1e-5)均为 FastGen 上游公开配置,凡本报告列为我方机制/设计处一律以"沿用上游公开配置"为准;(3) 轴 C:"batch 单因素在视频 DMD 无先例"被 Data-Forcing(batch 16 vs 128 受控,视频)击破,幸存收窄为三因素联合受控 + per-anchor 更新量主打;(4) Self-Forcing 谱系 ε 精确化:判别器更新共享 t+ε、生成器侧仅共享 t。判别器宿主 = 冻结 teacher 的更正见下方 4.4 节注记(2026-07-06,仍有效)。

---

## 1. Executive Summary

1. **我们的 GAN 配置正在成为 Wan 系社区共识,而非独有设计**。One-Forcing(2605.23458,精读)与我们逐项重合:GAN 生成端权重 0.03(它判别端也 0.03)、TTUR fake:gen=5:1、real-data 判别器、teacher CFG=5、shift=5、无判别器正则、同基座 Wan2.1-T2V-1.3B。**重要更正(2026-07-06,修正 T1 的记载)**:One-Forcing 的判别器寄生在 **fake score** backbone 第 {21,29} 层(register-token attention 头),不是生成器 backbone——我们"生成器 backbone 15/22/29 层 + multiscale MLP"仍是可辩护差异点,但差异比 T1 认为的更细。
2. **判别器主张需收窄(组件层初判:原表述不支持)**:除 One-Forcing/POSE 外,又发现三个近同构先例——ASD(2511.01419,同基座,fake score 骨干第 **12/21/29** 层多层判别头 + R1/R2)、Taming DiT(2507.13343,字面用"冻结生成器前 K 块"作判别特征,latent 视频 4-step)、AAD-1(2606.03972,Wan 权重初始化骨干第 19/29/39 层头)。**可幸存的收窄表述**:"live 生成器骨干(随蒸馏更新)+ 多层纯 MLP 头 + 同 t 同噪声配对"的具体组合无先例。
3. **"同 t 同噪声配对(gan_use_same_t_noise)未被单独消融过"成立(组件层初判:支持)**——ASD 明确独立采样噪声、AAD-1 未说明、无任何论文消融真假共享 ε;这是我们可以自己补做并写进论文的低成本独家消融(same-ε vs independent-ε)。
4. **我们判别器的两个最可能弱点**(文献定价充分):(a) **零正则 + 每 iter 更新 + 同 t 同噪声使判别任务变容易 → 判别器过强/过拟合**,Wan 家族三篇对抗蒸馏(ASD λ=600 / AAD-1 λ=20 / POSE ST-R1)全部默认带近似 R1,Diffusion2GAN 消融显示 R1 单项 -1.37 FID 为最大增益项;(b) **MLP 头无显式时空聚合、logit 粒度未设计 → 倾向奖励静态平滑**,三组独立消融同向(POSE cross-attn 头 79.56 vs conv 头 63.29;Taming DiT transformer 头 81.63 vs 卷积头 77.24;AAD-1 frame-wise logit 导致 Dynamics 1.08 的完全静态)。
5. **multi-step DM 构造的证据强度排序**:on-policy rollout/backward simulation 最强(Self-Forcing 在同基座 Wan2.1-1.3B/4-step 上受控量化:on-policy 84.31 vs teacher-forcing 82.32,+2.0 VBench;DMD2 图像 Patch FID 20.86→24.21)> TDM 区间条件化(去 K 条件 HPS 掉 1.4~2.8)> Salt 复合一致性(**锚点从 4 加密到 8 反而退化** 82.78→82.54,SC 正则修复至 83.36,同基座)> ASD 邻步自蒸馏(1-step +2.52,4-step 收益≈0)> Phased DMD 子区间 > CDM 连续 schedule(仅 +0.079 HPSv3,图像)。Few-Step SiD 的 Lemma 1(匹配各步输出的均匀混合;单一共享 fake score 即充分)为我们"一个 fake score 服务所有锚点"提供理论背书。
6. **轴 A 组件层初判:部分支持**。严格合取("4-8 步 + 通用 T2V + ≥3 种锚点形状受控消融 + 选择准则")下无反例;但 TMD(同基座!t_dmd/t_student 双 shift 消融,2-step 84.39 vs 83.44,且"VBench 测不出的 severe mode collapse")、Lip Forcing(单锚点 4 位置扫描 + 显式 tradeoff 准则)、FlashMol(分子域 ρ 4 档扫描,极端档断崖崩溃 70.78%→35.23%)构成近反例,措辞必须限定。SGMD 在 Wan2.1-14B 上默默使用 shift≈8 锚点且零论证——既是消融矩阵须覆盖 σ=8 的理由,也是"实践分裂无准则"的空缺证据。
7. **稳定性数字的可引用/须自补分界**:可引——Seaweed-APT(ICML 2025,精读)视频 batch 256→mode collapse/1024 不崩、LR 5e-6→3e-6 求稳、近似 R1(λ=100,σ=0.1)无它即崩;SenseFlow(ICLR 2026 Poster,精读)8B flow 模型上 TTUR 5/10/20 全震荡、IDA(fake←0.97·fake+0.03·student,开销 +0.6~4%)修复;ADM(ICCV 2025 Highlight,精读)TTUR 1→8 仅 CLIP +0.07 却 2.53x 时间(图像,SDXL)。**必须自补**——"视频 + reverse-KL + 特征判别器 + relay"配置类下的 TTUR 扫描、GAN on/off、EMA on/off、4-step 重置后 warmup 策略:文献全部零覆盖。
8. **对我们最有价值的三个升级候选**(与"训练调度轴"故事相容、不喧宾夺主):判别器近似 R1(σ≈0.05-0.1,λ 扫 20~600)、IDA 用于 4-step relay 重置后的 fake score 追踪窗口、Data-Forcing 式 post-training(100-300 iter,teacher score 求值点 50% 概率换真实 latent;同基座证据 camera trajectory diversity +349%)。**警示**:Data-Forcing 同基座同步数消融显示加 GAN 使 Dynamic Degree 0.50→0.375——GAN 权重扫值必须以 Dynamic Degree 为一级指标。

---

## 2. 组件层结论

### 2.1 multi-step DM 目标的构造对比(回答 Q1)

结论先行:训练输入分布的五类构造中,**on-policy rollout(backward simulation)是唯一在同基座视频上被受控量化的**,我们沿用 DMD2 默认构造有最强证据;区间化/混合/连续化各有一票,均不足以推翻现状。

| 构造 | 代表 | 定义 | 消融证据(带数字) |
|---|---|---|---|
| 模拟推理链(on-policy/backward simulation) | DMD2、Self-Forcing、AnyFlow、CDM(随机化变体) | 训练输入 = 学生自己 rollout 到锚点的中间样本 | DMD2:去掉后 SDXL Patch FID 20.86→24.21;**Self-Forcing(同基座 Wan2.1-1.3B/4-step):on-policy+DMD 84.31 vs Teacher-Forcing+DMD 82.32、Diffusion-Forcing+DMD 82.76**;损失消融 DMD 84.31 > SiD 84.07 > GAN 83.88 |
| 均匀混合 | Few-Step SiD(精读,3.3 节) | 匹配所有生成步输出的均匀混合与数据分布;Lemma 1:teacher 最优时各步最优分布相同→单一共享 fake score 充分 | 无 mismatch 消融;SDXL 4-step SiDa2 Zero-CFG FID 13.25 vs DMD2 19.32 |
| 区间/条件化 | TDM(K 条件化 + 不重叠区间)、Phased DMD(SNR 子区间,T1 已核) | 各步数/各相位的训练分布显式隔离,防止污染共享 fake score | TDM:去 K 条件化 HPS 1-step 28.90→26.11、4-step 31.31→29.39;Phased DMD:nested 区间 > disjoint |
| 复合一致性正则 | Salt(2604.03118) | 不改输入构造,惩罚"一步直达 vs 两步复合"端点差 | **同基座反例:DMD 锚点 4→8 加密反而退化(Total 82.78→82.54,过曝 + dynamic collapse),加 SC 正则 83.36** |
| 邻步自蒸馏 | ASD(2511.01419) | n 步分布对齐自身 n+1 步分布(relativistic GAN) | 1-step:78.13→80.65(仅 ASD)→83.89(+FFE);**4-step 收益≈0(84.31→84.38)** |

Video-specific 可实施借鉴(按优先级):**(1)零成本核对**:确认我们 8/4-step 训练输入确为学生 backward simulation 产物而非 noised real latent(同基座证据值 +2.0 VBench);**(2)低成本 post-training**:Data-Forcing 配方(见 2.4/6.1);**(3)条件触发**:若 8-step 阶段出现"步数多反而差/过曝/动态坍缩",加 Salt 的 SC 复合一致性正则(同基座 82.78→83.36)——该正则还让逐步算子更可复合,对 8→4 relay 初始化平滑性有理论好处,可纳入 relay 叙事;**(4)下探 2/1 步时**再用 ASD 邻步对齐,4-step 不必;**(5)不建议改**:CDM 连续 schedule(收益 +0.079 HPSv3、无视频证据),固定 t_list + 区间隔离被 TDM 消融间接支持。

### 2.2 蒸馏内部锚点处理的受控证据(回答 Q2,轴 A 组件基础)

结论先行:锚点"当变量"的受控证据只有四份且各有局限(TMD:shift 开/关两档、1-2 步;Lip Forcing:单锚点、唇同步子域;FlashMol:分子域、密集方向相反;TDD:CM 目标、曲线无数字);**"4-8 步通用 T2V 的多形状系统消融 + 选择准则"仍是空白**,且该空白在同族基座上有直接的实践分裂证据(SGMD 用 shift≈8、CoDMD 未公开锚点、DOLLAR 等距不崩 vs TMD 无 shift 即崩)。

逐篇提取(改了什么 / 差多少 / 可迁移性):

- **TMD**(2601.09881,同基座 Wan2.1-1.3B、同 480x832x81):shift 函数 t=γt′/((γ−1)t′+1) 分别施加于 t_dmd(训练加噪采样)与 t_student(推理锚点)。2-step:t_student 有 shift(γ=10)84.39 vs 无 83.44;1-step:t_dmd 有/无 shift VBench 83.24 vs 83.22 几乎不动,但无 shift 版"severe mode collapse that VBench scores cannot capture"(所有主体固定在画面左侧)。其 DMD2-v 基线用 t_dmd γ=5 / t_student γ=10——**训练侧 shift 与推理锚点 shift 是两个独立变量,最优值可以不同**。可迁移性最高。
- **Lip Forcing**(2606.11180,唇同步,2-step):固定 j0、扫第二锚点 j1∈{13,25,30,37}:FVD 单调改善 135.22→114.78,Sync-C 峰值在 25(6.95)——**分布保真与条件对齐随锚点位置反向变化**,并给显式 tradeoff 准则(取 30);另证明"训练 t 在锚点邻域窗采样、推理取窗中心"更稳(窗口化后 FVD 138.32→119.88)。
- **FlashMol**(2605.07020,分子域 DMD):EDM ρ∈{2.0,2.25,2.75,5.0} 四档形状扫描:Mol Stability 69.93/70.78(最优)/68.46/**35.23(断崖崩溃)**;并明确"8-step 学生用均匀 timestep 无法有效 DMD 优化"。**警示:形状-质量曲线不是平缓的,极端档会断崖**;但其最优方向(低噪加密)与视频(高噪加密)可能相反,准则不可照搬。
- **TDD**(2409.01347,AAAI 2025,CM 目标,图像):训练 timestep 只从"推理会经过的锚点邻域"受限采样 + 随机偏移 η;4-8 步混合锚点训练优于单 4 步锚点(曲线,无逐项数字)。可迁移思想:**8-step 阶段训练 t 偏向 8-step 与 4-step 锚点并集邻域,可能让 relay 初始化更平滑**。
- **Phased DMD nested interval**(T1 已核,定位):训练 t 支撑集应从当前锚点延伸到 t=1,不应只压在锚点邻域——我们 shifted t 采样的高噪端覆盖原则。
- **TurboDiffusion**(thu-ml,工程文档):要求"4 步 t_list 是 8 步 t_list 的子集"。**shift=5 下我们的 8 步锚点偶数位恰好就是 4 步锚点(天然嵌套)**——是 relay 初始化平滑性的免费论据,并派生"非嵌套对照"这一与 step-count relay 绑定的独家消融。

**t_list 消融矩阵建议**(4-step,t=σu/(1+(σ−1)u),u∈{1,0.75,0.5,0.25};扫 5-6 形状):
1. σ=3(低噪更密):[0.999, 0.900, 0.750, 0.500, 0]
2. σ=5(基线=现配置):[0.999, 0.9375, 0.833, 0.625, 0]
3. σ=8(=SGMD 实际配置):[0.999, 0.960, 0.889, 0.727, 0]
4. σ=12(极端高噪档,验证断崖):[0.999, 0.973, 0.923, 0.800, 0]
5. 均匀无 shift(负对照,TMD 预测 collapse):[0.999, 0.750, 0.500, 0.250, 0]
6. relay 专属非嵌套对照(如 σ=5 作用于 u∈{1,0.7,0.45,0.2}),检验嵌套性对 relay 的价值——**任何已有工作都没做**。
第二因子(TMD 证据):训练 t 采样 shift 与推理锚点 shift 解耦,至少做"锚点 σ=5 × 训练采样 σ∈{3,5,10}"一组。8-step 侧已有 `_step8_normalize`(均匀)现成数据点。

**最敏感指标**(按优先级):(1) mode collapse 征兆——跨 seed 多样性(同 prompt N seeds 的 LPIPS/DINO 特征距离)、主体质心空间分布、首帧饱和度分布(**TMD 证明 VBench overall 完全测不出 collapse**);(2) VBench 分维而非总分——Dynamic Degree、Subject Consistency、Imaging Quality(Adaptive Video Distillation 的 Instance Preservation 可加);semantic 与 quality 维可能随锚点后移反向变化(Lip Forcing 的 T2V 对应物);(3) 固定 prompt 集 FVD 作形状扫描主排序指标(Lip Forcing 中 FVD 对锚点位置严格单调)。

### 2.3 判别器设计空间(回答 Q3,本任务核心)

结论先行:四维设计空间中,证据最强的选择是——特征来源:**扩散先验骨干 > 外部感知骨干/从头训**(Diffusion2GAN:+Diffusion D 使 FID 14.72→12.04),且**骨干跟随生成器演化 > 冻结**(POSE 唯一受控:unified EMA 79.56 vs 冻结 77.36);时间维:**必须有显式时空聚合**(SF-V:FVD 180.9 vs spatial-only 514.7),DiT 时代的形态是 **video-wise attention 池化 logit**(AAD-1/POSE/Taming DiT 三方向一致);条件:timestep 条件共识最高(LADD 的噪声分布是"结构 vs 纹理"反馈旋钮),文本/输入条件收益小而稳(Diffusion2GAN +0.07);正则:**R1 家族是证据最强单项**(Diffusion2GAN -1.37 FID;APT/ASD/AAD-1/POSE 在大 transformer 上一致用近似版)。

**(i) 特征来源五条路线**:
- **A. 生成器/student 骨干**:POSE(AAAI 2026,判别骨干动态继承生成器 EMA,decay 0.995;消融:全参数 OOM / 冻结 77.36 / unified 79.56;头消融:conv 63.29 / cross-attn 78.31 / cross-attn+语义 79.56)、Taming DiT(2507.13343,冻结生成器前 K 块 + 可训 DiT block + MLP 头,latent 4-step;头消融 81.63 vs 卷积 77.24)、**我们**(live 生成器骨干 15/22/29 层 + multiscale MLP,该具体组合无完全同构先例)。
- **B. fake score 网络寄生**:DMD2(bottleneck 单头,1e-3/3e-3)、ASD(**同基座**,fake score 骨干第 12/21/29 层 cross-attn+分类头,R3GAN 式 R1&R2 λ=600/σ=0.05,1-step 78.13→83.89)、One-Forcing(精读,3.1 节)、SiDA(2410.14919,复用 score 网络 encoder,ImageNet64 一步 FID 1.110,细节待核实)。
- **C. teacher 生成特征**:LADD(每 attention block 后 token 序列 + 各层独立头;噪声分布 π(t;m,s) 消融:低噪→纹理真实但全局不连贯,高噪→细节丢失,m=1,s=1 最优)、Diffusion2GAN(ECCV 2024,**最干净的消融链**:E-LatentLPIPS 基线 14.72 → +Diffusion D 12.04 → +z 条件 11.97 → +single-sample lazy R1 10.60 → +多尺度 9.58 → +mix-and-match 9.45)、SDXL-Lightning(T1 已精读)、SF-V/NitroFusion(冻结 teacher encoder)。
- **D. 外部 VFM 骨干**:ADD(DINOv2)、MCM(逐帧 DINOv2 第 2/5/8/11 层 + 稀疏抽帧,λ_adv=1;FVD 854→703(+GAN)→526(全量)——**纯 2D 判别器必须配独立运动监督兜底,否则奖励静态**)、OSV(DINOv2 + latent 直接 upsample 免 VAE decode:FVD 232.25→171.15)、Flash-DMD(冻结 SAM;高噪只 DM、低噪只 GAN 的 timestep 解耦)、SenseFlow(DINOv2 ViT-L/14 hook [5,10,14,19,23] 层 + spectral-norm 卷积头,像素域干净 x0,精读 3.4 节)。
- **E. teacher 初始化的独立可训练判别器**:Seaweed-APT(8B 完整 DiT,第 16/26/36 层 cross-attn-only 头,干净 latent 输入 + timestep ensemble;精读 3.2 节)、AAD-1(Wan2.1 权重初始化双向骨干,第 19/29/39 层头聚合 video-wise 标量;**拓扑消融:causal+frame-wise 完全静态(Dynamics 1.08)→ bidirectional+video-wise 最优**;近似 R1&R2 λ=20/σ=0.05)。

**(ii) 时间维**:单帧 2D(MCM,需运动项兜底)< 空间+时间双头(SF-V:FVD 180.9 vs 514.7/539.2,设计空间最强单点证据;OSV 1D conv 时间头)< video-wise attention 聚合(AAD-1/POSE/Taming DiT)。latent 判别已是视频主流(省 decode、分辨率不受 VFM 限制)。**我们的缓冲**:DiT 骨干 15/22/29 层特征已经过 3D full attention,时间信息已混入(不同于 SF-V 的 UNet encoder),不至于 334 FVD 量级崩坏;但 AAD-1 证明即使骨干是 DiT,**头上的 logit 聚合方式仍显著决定 motion 质量**。

**(iii) 条件**:timestep 条件几乎普遍(例外:ASD 的头显式不接收 t,说明非必需);文本条件常见(LADD/MCM/Diffusion2GAN);x_t/轨迹条件保 flow(SDXL-Lightning);**同 t 配对是行业隐含默认,同 ε(真假共享扰动噪声)极罕见且无人消融**(ASD 明确独立采样;详见 4.2 主张 2)。ADM 的判别器 t 均匀采样(兼顾高低噪反馈)与我们"绑定生成 t"是相反选择,可列为备选消融。

**(iv) 正则与权重**:精确 R1(Diffusion2GAN single-sample lazy 间隔 16,-1.37 FID;SF-V hinge+R1)/ 近似 R1(APT 首提:||D(x)−D(x+σ 扰动)||²,λ=100、σ=0.01/0.1,"无它训练迅速崩";**Wan 家族三篇 ASD/AAD-1/POSE 全部默认携带,已是该基座事实标准**)/ 头池 refresh(NitroFusion:480 头、~1%/iter flush;消融 w/o Dynamic Pool 19.46 vs Full 18.70)/ weak-GAN 权重谱:DMD2 1e-3~3e-3 → **我们 0.03 = One-Forcing 0.03(视频侧新常态)** → MCM λ=1(GAN 为主监督)。**反面证据(同基座同步数)**:Data-Forcing 消融——加 GAN 使 Dynamic Degree 0.500→0.375、Imaging Quality 0.7452→0.7210,遂弃用 GAN。

**我们判别器的两个最可能弱点与升级方向**(综合裁决):

- **弱点 1:零判别器正则 + 每 iter 更新 + 同 t 同噪声配对 → 判别器过强/过拟合**。同 t 同 ε 消除了噪声方差、判别任务更容易;我们既无 R1(Diffusion2GAN:R1 是消融链最大单项)、无近似 R1(APT:大 transformer 无它"迅速崩";Wan 三篇全带)、无头 refresh。风险峰值在 4-step relay 阶段:判别器重置后在已相当逼真的 8-step-student 输出上重新拟合,过强判别器会把梯度推向 Data-Forcing 观察到的动态度塌缩。**升级**:优先加近似 R1(真实分支加 σ≈0.05 高斯扰动、惩罚输出差;免二阶导、兼容 FSDP),λ 在 20(AAD-1)~600(ASD)间扫;备选小比例头 refresh。**监控**:判别器 logit gap 增速(One-Forcing 给出健康参照:均值 1.53、方差 1.20 持续波动;贴零=塌缩)+ VBench Dynamic Degree。
- **弱点 2:MLP 头缺显式时空聚合,logit 粒度(逐 token/逐尺度平均 vs video-wise)未经设计 → 倾向奖励静态平滑**。三组独立消融同向(POSE +16.3;Taming DiT +4.4;AAD-1 frame-wise 即完全静态)。**升级**:把 multiscale MLP 头改为(或并联)learnable-query cross-attention 池化到 video-wise 单标量(POSE/AAD-1 配方);最低成本版:每层头前加 (b·h·w)×c×t 维 1D temporal 分支(SF-V/OSV 配方);零预算版:把逐 token logit 平均改为含时间维的加权聚合并消融 frame-wise vs video-wise。

附带:0.03 权重比 DMD2 高 10-30 倍但与 One-Forcing 一致;建议 4-step 阶段对 {0, 0.01, 0.03} 扫值并盯 Dynamic Degree(Data-Forcing 警示)。若 GAN 与 DM 梯度冲突,Flash-DMD 的"高噪只 DM、低噪只 GAN"解耦是现成缓解。

### 2.4 训练稳定性定量结论盘点(回答 Q4)

**可直接引用的数字**(工作/设置/数字/模态;绝对值不可跨协议横比):

| 议题 | 工作 | 数字 | 模态/语境 |
|---|---|---|---|
| TTUR | DMD2(T1 已核) | 1 不稳 / 10 慢 / 5 最优 | 图像 |
| TTUR | SenseFlow(精读) | 无 IDA 时 5/10/20 全部严重震荡;DMD2 式 latent 判别器在 SD3.5 上调到 20 仍崩出全黑图 | 8B flow 图像 |
| TTUR | ADM(精读) | fake 更新 1→4→8:CLIP 35.2557→35.2583→35.3299,时间 x1.00/x1.85/x2.53,"clearly unwarranted" | **图像(SDXL),Table 5 已核实** |
| TTUR+EMA | Flash-DMD | fake-score 加 EMA 后 TTUR=1-2 已足够(TTUR=1@1k iter ImageReward 0.9509,仅 DMD2 成本 2.1%) | 图像 |
| TTUR | SGMD | fake 更新率 5→1,约 3x 加速;但依赖换目标(stop-grad Fisher)且无 GAN | **视频 Wan2.1-14B** |
| batch | Seaweed-APT(精读) | 视频 batch 256 → mode collapse(跨 prompt/seed),1024 不崩;实用 batch 图像 9062/视频 2048 | 视频,1-step 纯 GAN |
| batch | Embedding Loss(2604.22379) | DMD 梯度方差存在 batch 无关底座(O(1/B) 之外),加 batch 边际递减;DMD 需 batch 336 vs +EL batch 16 更优 | 图像,理论+实验 |
| LR | Seaweed-APT | 图像 5e-6 → 视频 3e-6 "for stability";不用 weight decay、不用 grad clipping | 视频 |
| 判别器正则 | Seaweed-APT | 近似 R1:λ=100、σ=0.01(图)/0.1(视频);无它判别器 loss 归零、训练崩溃 | 视频 |
| 判别器正则 | Diffusion2GAN | single-sample lazy R1(间隔 16):FID 11.97→10.60(消融链最大单项) | 图像 |
| 判别器正则 | ASD / AAD-1 | 近似 R1&R2:λ=600,σ=0.05 / λ=20,σ=0.05 | **视频 Wan 系** |
| fake 稳定化 | SenseFlow | IDA:fake←0.97·fake+0.03·student(每次 student 更新后),开销 +0.57~3.97%/iter;w/o IDA FID 17.83 vs 13.38 | 8B flow 图像 |
| fake LoRA | MagicDistillation | LoRA 版 vanilla DMD 不稳;解法是 weak-to-strong 混合(α_weak=0.25 最优);**无 LoRA vs 全参受控对比,rank 未披露** | 视频 13B I2V |
| EMA | Seaweed-APT | EMA 0.995 + 350/300 updates 早停("质量开始退化前"取 EMA checkpoint) | 视频 |
| 优化器 | SGMD / One-Forcing | AdamW β1=0(禁一阶动量适配交替更新);One-Forcing:β1=0、LR 1e-5、EMA 0.99 | 视频 Wan 系 |
| 其他 | Practical Guide(2512.13006,存在性已确认) | 仅覆盖 sCM/MeanFlow(FLUX),无 DMD/GAN 内容;可借用一条:DiT timestep 输入归一到 [0,1] 防梯度范数塌缩 | 图像 |

**必须自己补的实验**(按对轴 C 的支撑度排序):(1) **TTUR 1/2/5 扫描**——"视频 + reverse-KL + 特征判别器 + relay 两阶段"配置类无任何数据,且 4-step 重置后最优 ratio 可能与 8-step 不同,最便宜的空白表格;(2) **fake-score EMA + TTUR=1 对照**(验证 Flash-DMD 机制是否迁移视频,省 60% fake 前反向);(3) **判别器近似 R1 on/off**(4-step 重置后判别器冷启动是否需要正则,无人测过);(4) **student 侧 EMA on/off**(视频 DMD 零消融);(5) batch 2 点对照(16 vs 64;EL 理论支持"加 batch 边际递减",反向支撑我们小 batch 可行);(6) **4-step 重置后 warmup 策略**(fake/D 先单独预热 N iter 再放开 student)——relay 特有、文献零覆盖,若有效即是自有贡献。红线:Self-Forcing 的 LR 2e-6/grad clip 10.0/β1=0 目前是二手转述,引用前必须查官方 repo configs(guandeh17/Self-Forcing)。

### 2.5 论文清单表(组件粒度,20 篇)

| # | Paper | 年 | Venue/状态 | Component | Mechanism 一句话 | Evidence | Relation | U/R |
|---|---|---|---|---|---|---|---|---|
| 1 | [One-Forcing](https://arxiv.org/abs/2605.23458) | 2026 | arXiv WIP | 判别器/稳定性 | fake score 骨干 {21,29} 层 register-token attention 头,λ_G=λ_D=0.03,无正则 | medium(无 GAN on/off 消融) | 配置最接近;精读 3.1 | Both |
| 2 | [Seaweed-APT](https://arxiv.org/abs/2501.08316) | 2025 | ICML 2025(已核) | 稳定性/判别器 | 纯 GAN post-training 的完整稳定配方(巨 batch/aR1/EMA/早停) | strong | batch/LR/aR1 数字坐标;精读 3.2 | Useful |
| 3 | [SenseFlow](https://arxiv.org/abs/2506.00523) | 2025 | ICLR 2026 Poster(已核) | 稳定性/判别器 | IDA 近端对齐 + ISG 段内接力 + VFM 判别器 | strong | fake 追踪失败的归因框架与修复;精读 3.4 | Useful |
| 4 | [ADM/DMDX](https://arxiv.org/abs/2507.18569) | 2025 | **ICCV 2025 Highlight(已核,更正 T1)** | DM变体/判别器/稳定性 | 对抗项替代 reverse-KL 本身(判 teacher vs fake 的 PF-ODE 预测) | strong | TTUR 消融来源;判别器升级的"喧宾夺主"警示;精读 3.5 | Both |
| 5 | [Few-Step SiD](https://arxiv.org/abs/2505.12674) | 2025 | arXiv,待核实 | DM变体 | 均匀混合匹配 + Zero/Anti-CFG + fake 骨干零参判别图 | strong(理论+数字) | 共享 fake score 的理论背书;降 CFG 消融依据;精读 3.3 | Useful |
| 6 | [ASD](https://arxiv.org/abs/2511.01419) | 2025 | ICLR 2026(T1 核) | 判别器/DM变体 | fake 骨干 12/21/29 层多头 + R1&R2(λ=600)+ n/(n+1) 邻步对齐 | strong | **判别器主张 1 的最强反例**;同基座 | Risk |
| 7 | [Taming DiT for Mobile Video](https://arxiv.org/abs/2507.13343) | 2025 | arXiv,待核实 | 判别器 | 冻结生成器前 K 块 + DiT block 头,latent 4-step | strong(头消融 +4.4) | "生成器骨干判别"字面先例 | Risk |
| 8 | [AAD-1](https://arxiv.org/abs/2606.03972) | 2026 | ICML 2026(GitHub 自述,待核实) | 判别器 | Wan 初始化双向骨干 19/29/39 层头,video-wise 聚合,aR1 λ=20 | strong(拓扑消融) | frame-wise→静态的关键证据 | Both |
| 9 | [POSE/V-PAE](https://arxiv.org/abs/2508.21019) | 2025 | AAAI 2026(已核) | 判别器 | 判别骨干继承生成器 EMA + cross-attn 语义头 + ST-R1 | strong(骨干机制唯一受控消融) | "跟随生成器 > 冻结"证据 | Useful |
| 10 | [NitroFusion](https://arxiv.org/abs/2412.02030) | 2024 | CVPR 2025(待核实) | 判别器/稳定性 | 480 头动态池按噪声级分舱 + ~1%/iter refresh | strong | 抗判别器过拟合的备选机制 | Useful |
| 11 | [Diffusion2GAN](https://arxiv.org/abs/2405.05967) | 2024 | ECCV 2024(待核实) | 判别器/稳定性 | teacher 骨干 + 多尺度分支 + single-sample lazy R1 | strong(最干净消融链) | 判别器各设计项收益排序参照 | Useful |
| 12 | [SF-V](https://arxiv.org/abs/2406.04324) | 2024 | NeurIPS 2024(待核实) | 判别器 | teacher encoder + spatial/temporal 双头 | strong(FVD 180.9 vs 514.7) | 时间头价值的最强数字 | Useful |
| 13 | [Self-Forcing](https://arxiv.org/abs/2506.08009) | 2025 | NeurIPS 2025(待核,T3 属地) | DM变体 | on-policy rollout 输入构造 | strong(同基座受控 +2.0) | backward simulation 的视频侧证据;细节归 T3 | Useful |
| 14 | [TMD](https://arxiv.org/abs/2601.09881) | 2026 | arXiv,待核实 | 锚点/DM变体 | t_dmd 与 t_student 双 shift 消融;3D conv 判别器 | strong | 轴 A 最近反例 + 消融范式模板;同基座 | Both |
| 15 | [Salt](https://arxiv.org/abs/2604.03118) | 2026 | ECCV 2026(arXiv 标注,待核) | DM变体 | semigroup 复合一致性正则 | strong(4→8 锚点退化反例) | 8-step 阶段的条件触发补丁;relay 平滑性论据 | Useful |
| 16 | [Data-Forcing](https://arxiv.org/abs/2606.18478) | 2026 | arXiv,待核实 | DM变体/判别器 | teacher score 求值点 50% 换真实 latent;消融后移除 GAN | strong | 多样性修复配方 + **GAN 负收益警示(同基座同步数)** | Both |
| 17 | [FlashMol](https://arxiv.org/abs/2605.07020) | 2026 | arXiv,待核实 | 锚点 | EDM ρ 四档形状扫描 + 准则 | strong | 形状断崖警示;方向不可照搬 | Useful |
| 18 | [Lip Forcing](https://arxiv.org/abs/2606.11180) | 2026 | arXiv,待核实 | 锚点 | 单锚点 4 位置扫描 + tradeoff 准则 + 锚点邻域窗训练 | strong | 消融范式 + 双指标反向变化警示 | Useful |
| 19 | [Flash-DMD](https://arxiv.org/abs/2511.20549) | 2025 | arXiv,待核实 | 稳定性/判别器 | fake-EMA 换低 TTUR;高噪 DM/低噪 GAN 解耦 | medium | TTUR 降档机制候选 | Useful |
| 20 | [SGMD](https://arxiv.org/abs/2605.30116) | 2026 | ICML 2026(待核实) | 稳定性/锚点 | stop-grad Fisher 目标使 fake 更新率 5→1;shift≈8 锚点零论证 | strong | 视频侧 TTUR 数据点;σ=8 档存在性证据 | Both |

次级条目(正文引用,不占表):TDD([2409.01347](https://arxiv.org/abs/2409.01347),AAAI 2025)、CDM([2605.06376](https://arxiv.org/abs/2605.06376))、AnyFlow([2605.13724](https://arxiv.org/abs/2605.13724),全文不可达,数字待核实)、MCM/LADD/OSV/MagicDistillation/Phased DMD(T1 清单已有,本次补组件级证据)、FSF-DMD([2605.19256](https://arxiv.org/abs/2605.19256),删除 fake score 用生成器自身 endpoint 伪速度替代,FID 3.85 vs DMD2 4.18,图像)、Embedding Loss([2604.22379](https://arxiv.org/abs/2604.22379))、SiDA([2410.14919](https://arxiv.org/abs/2410.14919),仅 abstract)、Why Are DMD Students Lazy([2606.02237](https://arxiv.org/abs/2606.02237),copying 行为归因,仅摘要核实,weak)、Practical Guide([2512.13006](https://arxiv.org/abs/2512.13006),仅 sCM/MeanFlow)、TurboDiffusion([GitHub](https://github.com/thu-ml/TurboDiffusion),锚点嵌套性工程需求)。

---

## 3. 精读卡(5 篇,与 T1 零重复)

### 3.1 One-Forcing(arXiv 2605.23458,WIP)——配置最接近的 DMD+GAN 视频工作

- **核心问题**:因果 AR 视频 1-step 的稳定性;诊断"score-only fake model 缺少显式拒绝'整体可分辨于真实视频'样本的机制",用 DMD + 真实数据 noised-latent GAN 辅助修复模糊帧。
- **机制(判别器精确设计,本次核实)**:判别器寄生在 **fake score** transformer backbone(原文"reusing the trainable fake-score transformer backbone as a noised latent discriminator"),第 {21,29} 层各挂 1 个可学习 register token,对该层 latent tokens 做轻量 attention(1536 dim/2048 FFN/12 heads),两层特征 concat 过 MLP 头出标量 logit D(x_t,t,c);真假样本**同 t 加噪但噪声独立采样**(与我们同 t 同 ε 不同);non-saturating logistic,**无任何判别器正则**;λ_G=λ_D=0.03;critic(fake flow 损失 + 0.03·adv)每 iter、生成器每 5 iter(自述沿袭 DMD2 TTUR)。teacher 为 **Wan2.1-T2V-14B(跨规模)**,CFG=5;ODE-init;AdamW β1=0、LR 1e-5、EMA 0.99;framewise 约 200 步收敛(单位待核实)。
- **关键证据**:1-step VBench 83.76(Dynamic 52.76)vs Self-Forcing 77.18 / Causal-Forcing 78.39 / ASD 79.12;人评 88.4%/92.7% 偏好。消融:CD(consistency)init 使 Dynamic 52.76→23.61(支撑"consistency 弱动态");forward-KL 正则使 Dynamic→1.30(几乎静止);判别器 logit gap μ=1.53、σ=1.20 持续波动=健康,ASD 贴零=塌缩。**缺失:无 GAN on/off、无层选择、无权重、无配对方式消融**。
- **对我们的启发**:(a) 0.03 + 5:1 + 无正则在独立工作中收敛良好,可作超参外部佐证;(b) 判别器宿主(fake score vs 生成器)是真实设计自由度,训练不稳时其方案是备选;(c) 它判别端也降权 0.03——判别器过强时可借鉴;(d) logit gap 是现成的对抗健康度监控指标;(e) 任何把学生拉向保守分布的项(forward-KL 类)都杀动态度,4-step 阶段勿加;(f) **它没做 GAN on/off——我们在 4-step relay 上补这一消融即成超越它的组件级证据**。
- **划界**:任务形态(因果 1-step vs 双向 4-step relay)、判别器宿主(fake score vs 生成器 backbone)、层(单 register-token 池化 {21,29} vs 多尺度 MLP 15/22/29)、配对(仅同 t vs 同 t 同 ε)、teacher(跨规模 14B vs 同基座 relay)。相同点要坦承:TTUR、0.03、real-data、CFG=5、shift=5、无正则——**该配置正在成为 Wan 系 DMD+GAN 社区共识,我们的贡献点必须压在 relay 协议与双向多步设定**。
- **不能 claim**:"GAN 项带来 +X"(它无孤立消融,83.76 vs 77.18 混杂多因素)、同 t 同 ε 有它背书(它噪声独立)、层选择/0.03 有实证(两家都无消融)、它是同基座蒸馏(teacher 14B)。WIP 未评审,引用需标注。

### 3.2 Seaweed-APT(arXiv 2501.08316,ICML 2025)——稳定性数字坐标

- **核心问题**:蒸馏把 teacher 设为上限;先 consistency 蒸馏拿"模糊但有效"的 1-step 初始化,再完全抛弃 teacher 对真实数据纯 GAN 后训练,8B 生成器 + 8B 判别器(~16B,自称最大 GAN)。
- **机制**:判别器从**原始 diffusion 权重**初始化(定性优于从蒸馏后权重,无消融数字);36 层 DiT 第 16/26/36 层插 cross-attn-only 头(单 learnable token query),三 token concat→单标量;输入**干净 latent 不加噪**("避免引入伪影"),但 t=0 会 collapse,故用 timestep ensemble:t~shift(U(0,T),s),s=1(图)/12(视频);近似 R1 = ||D(x)−D(x+N(0,σI))||²,σ=0.01/0.1,λ=100,每步施加;RMSProp α=0.9(=Adam β1=0,β2=0.9);G/D 1:1;EMA 0.995。
- **关键证据**:**视频 batch 256 → 跨 prompt/seed mode collapse,1024 不崩**(Figure 9);实用 batch 图像 9062 / 视频 2048(梯度累积);LR 图像 5e-6 → 视频 3e-6 "for stability";无 aR1 判别器 loss 迅速归零、生成器退化为"colored plates";只用末层特征"结构失衡",16/26/36 三层显著缓解(定性);仅 350(图)/300(视频)updates 即取 EMA checkpoint("质量开始退化前")。自承失败模式:1-step 结构完整性 −38.5%(视频人评 vs 25-step)、文本对齐 −8.3%(贴近真实分布反而降对齐,因 CFG 人为抬高 teacher 对齐)。
- **对我们的启发**:(a) 多层特征头(16/26/36 跨深度)与我们 15/22/29 同设计动机,可引作跨方法一致证据;(b) t=0 collapse 支持"判别器特征应来自带噪输入 + diffusion 先验"的合理性;(c) "判别器从原始 diffusion 权重初始化更好"为我们 relay 时**重置 fake score/判别器回 teacher 权重(而非继承 8-step 阶段)**提供文献先例;(d) 其失败区(纯 GAN + 1-step)恰是我们(DM 主导 + 弱 GAN + 4-step)规避的,可作 motivation;(e) 训练后期回退的应对配方:极短训练 + EMA + 提前取 checkpoint。
- **划界/不能 claim**:它无 DMD 对比、无 hybrid 消融;其稳定性配方建立在 2048~9062 batch、千卡 H100、1-step 纯 GAN 体制上,迁移到小 batch DMD 是**我们的推测而非其结论**;batch/aR1/多层头消融均为曲线/定性图,无 FID/VBench 数字;人评数字是相对偏好差,不可与绝对指标横比。

### 3.3 Few-Step SiD(arXiv 2505.12674,preprint 待核实)——multi-step DM 目标的理论对照

- **核心问题**:把 data-free 的 SiD(Fisher 散度)扩到 few-step,并解决 CFG 的"对齐 vs 多样性"权衡。
- **机制**:生成链每步对上步输出重新加噪(stochastic renoising)+ 步间 stop-gradient(免 BPTT);训练随机采一步 k、只在第 k 步输出上施加损失;**被匹配对象是所有步输出的均匀混合**;Lemma 1:teacher score 最优时各步最优分布相同且=数据分布 → **单一共享 fake score 充分,无需 step 条件化/分相 MoE**。τ_k 从 t_init=625 线性递减(非 T=999),论文自述 t_init=T 的变体即 DMD2 的 backward simulation。Zero-CFG(teacher 不放大引导、fake 去文本条件)/Anti-CFG(fake 负 CFG),GAN 分支保留文本条件承担对齐;判别器=fake score U-Net mid-block latent 通道池化(零新增参数),λ_adv 生成端 0.001。
- **关键证据**:SDXL 4-step:SiDa2 Zero-CFG FID **13.25** vs DMD2 19.32(CLIP 0.335 vs 0.332);SD1.5 1-step SiDa FID 7.89(自称一步历史最低);Zero-CFG 换 FID、Anti-CFG 换 CLIP 的双旋钮。无锚点/t_init 消融;无视频;无 recall 类多样性定量。
- **对我们的启发**:(a) **Lemma 1 是"一个 fake score 服务所有 t_list 锚点"的直接理论背书**(对照 Phased DMD 分相与 TDM 区间隔离,谱系内张力我们可以站 SiD 一边并引 TDM 证据说明"区间不重叠"由固定 t_list 天然满足);(b) **降 CFG 消融依据**:有真实数据条件 GAN 时,teacher CFG=5 的引导放大可能是多样性损失源头,值得做"CFG 5→3.5→1 + GAN 补对齐"消融(CoDMD 用 3.5 亦是旁证);(c) 其 GAN 权重 0.001 与 DMD2 同量级,反衬我们 0.03 是高配,须扫值;(d) 其"训练链=推理链 + stop-gradient"构造比 backward simulation 更省显存,是未来降本候选(但其推理是随机重加噪,与我们确定性 ODE 不同,理论不能平移)。
- **划界/不能 claim**:Fisher 散度族 ≠ 我们 reverse-KL 族;无步数接力;无视频;venue 未核实;κ 细节在附录未逐项核实。

### 3.4 SenseFlow(arXiv 2506.00523,ICLR 2026 Poster 已核)——fake score 稳定化与 relay 的接口

- **核心问题**:vanilla DMD/DMD2 在 8B/12B flow 图像模型上不收敛:fake score 的"内层 best response"追踪脆弱,TTUR 提到 20:1 仍震荡,DMD2 式 latent 判别器崩溃出全黑图。
- **机制**:**IDA**——每次生成器更新后 fake←λ·fake+(1−λ)·student(λ=0.97/0.98,官方代码核实),把 best response 松弛为 ε-best response,开销 +0.57~3.97%/iter;**ISG**——段内接力自蒸馏:每段采 t_mid,teacher(CFG=5,单步 Euler)从锚点走到 t_mid、冻结 student 走到下一锚点拼出 target,student 直连一步做回归——**不移动锚点,而是让锚点"吸收"段内信息**;**VFM 判别器**——冻结 DINOv2 ViT-L/14 hook [5,10,14,19,23] 层 + spectral-norm 卷积头,像素域干净 x0 输入 + DiffAugment,hinge + ω(t)=α_t² 时间权重,λ_G=0.1~2.0。
- **关键证据**:消融(SD3.5-L 4-step):完整 13.38 FID-T / w/o IDA 17.83 / w/o ISG+IDA 43.84;ISG 主要改善早期稳定性(FID-T@1.5k:14.48 有 vs **138.2 无**);FLUX 必须 IDA+ISG 才收敛;xi(t) 归一化单步重构误差在 **t∈[0.8,1.0] 局部震荡最剧烈**;VFM 判别器 vs DMD2 判别器(SDXL):偏好类指标全升、FID-T 变差(15.04→18.55,质量-多样性权衡);多样性 LPIPS 基本不变。4-step student IR 超 teacher(1.1713 vs 1.1629)。
- **对我们的启发**:(a) **IDA 是 4-step relay 阶段的天然接口**:fake score 重置后要重新追上 8-step 初始化的 student,正是"内层 best response 失败"风险最大的窗口;每次 student 更新后一行参数混合(λ 从 0.97 起),开销近零——**可讲成 relay 协议的组件(阶段间重置 + 阶段内连续对齐),完全服务训练调度轴故事**;(b) 我们 3 个锚点(0.999/0.937/0.833)全落在其证明 xi(t) 最震荡的 [0.8,1.0] 区间——ISG 与锚点选择正交、与 shifted 锚点兼容(FLUX 先例),是锚点不动的补偿方案;(c) VFM 判别器不建议整体替换(81 帧可微解码代价 + 无时序建模 + FID-T 代价),最多做"抽帧 DINOv2 头 + 保留 latent 判别器"混合试点;(d) 其"早期 checkpoint 指标序列"呈现法可直接用于展示 relay 重置 vs 继承的差异。
- **划界/不能 claim**:单阶段图像蒸馏,无多阶段/接力,不构成 relay 在先工作;失败结论出自 8B/12B flow 模型,**不能断言 1.3B 在 5:1 下必然震荡**;λ=0.97/teacher CFG 等来自官方代码而非论文正文(引用标注"official code");VFM vs DMD2 判别器对比仅 SDXL 且 FID-T 变差。

### 3.5 ADM/DMDX(arXiv 2507.18569,ICCV 2025 Highlight 已核)——对抗式分布匹配与 TTUR 数字

- **核心问题**:reverse-KL 在 student/teacher 分布支撑不重叠时梯度爆炸(p_real→0 处 log 发散)或 zero-forcing mode collapse(p_fake→0 处零梯度)。
- **机制**:保留 fake score 网络(仍在线拟合 p_fake),但**用对抗项替代 reverse-KL 本身**:teacher 与 fake score 各从 x_t 解一步 PF-ODE 到 t−Δt(Δt=T/64),判别器(冻结 teacher 骨干 + 多 block 可训头,t−Δt 条件)区分这两个预测,hinge(理论对应最小化有界对称的 TVD)。**判别器全程不接触真实数据**。one-step 另有 ADP 对抗预训练(latent 0.85 + SAM pixel 0.15 混合判别器)先建立支撑重叠;多步直接 ADM。
- **关键证据**:**TTUR 消融(Table 5,图像 SDXL,已核实)**:fake 更新 1/4/8 → CLIP 35.2557/35.2583/35.3299,时间 x1.00/x1.85/x2.53,"clearly unwarranted"——用以论证病因是支撑不重叠而非 fake 估计误差(最终更新率未明文,推断 1:1,引用需注明)。视频:CogVideoX-2b 8 步 78.584→2 倍训练时长 80.764 > teacher 100 步 80.036;5b 8 步 82.067 > teacher 81.226(少 96% NFE)。生成器训练 t 用 cubic 高噪偏置、判别器 t 均匀(无消融)。
- **对我们的启发**:(a) **"支撑重叠"理论可为 relay 背书**:8-step 最优 checkpoint 初始化 4-step 后,student 输出已接近 teacher 分布、支撑重叠好——这给了我们话语:relay 初始化降低了对强 TTUR/强判别器的依赖,沿用 5:1 是稳健选择而非必需(与 DMD2 消融表面冲突的调和:前提不同);(b) 它 8 步视频要 2 倍训练时长才超 teacher 且无 4 步视频,我们步数更激进;(c) 其判别器不看真实数据——**我们 0.03 GAN 注入 OpenVid 真实分布信息的价值不被它覆盖**。
- **划界(防喧宾夺主)**:ADM 换"目标函数轴"(对抗项即分布匹配主项),我们守"训练调度轴"(reverse-KL 主项 + 弱 GAN 辅助 + relay)。**若判别器升级采用 ADM 方案,方法故事会从调度轴漂移到目标轴,不建议**;低风险可借鉴仅:hinge、判别器 timestep 条件化、冻结骨干+轻量头的参数效率。
- **不能 claim**:它最终 TTUR=1(未明文);判别器头结构/挂载位置(主文未给);EMA/grad clip(主文未提及≠没有);CogVideoX 锚点与 LR(待核实)。

---

## 4. Gap 分析(逐主张裁决;附检索词组合与覆盖范围)

总体覆盖:4 路 sweep + 5 精读共约 60 组检索串,2024-07 至 2026-07 实时;40+ 篇原文页打开核实(SenseFlow 含官方代码克隆核对;ADM/POSE/APT/SenseFlow venue 经官网/OpenReview API 确认)。明确未覆盖:中文社区开源 Wan 蒸馏 repo 的配置旗标(gan_use_same_t_noise 类;主张 2 的"未消融"结论限于论文层面)、Self-Forcing 官方 configs(LR/clip 为二手,须核)、AnyFlow 全文(HTML 404/PDF 超限,数字待核实)、AAD-1 的 ε 是否共享、SiDA 全文。

### 4.1 主张(判别器 1):"基于生成器/student backbone 中间层特征的 multiscale 判别器用于视频 latent 少步蒸馏,除 One-Forcing/V-PAE 外无其他同构先例"——**不支持,须收窄**

反例(均原文核实):**ASD**(2511.01419,同基座,fake score 骨干第 12/21/29 层 cross-attn+分类头——多层 multiscale,与我们 15/22/29 几乎逐点对应,差在骨干宿主与 R1/R2 重正则)、**Taming DiT**(2507.13343,字面"冻结生成器前 K 块"特征 + DiT block 头,latent 视频 4-step,差在冻结/单尺度)、**AAD-1**(2606.03972,Wan 初始化骨干 19/29/39 层头,差在独立可训练判别器而非寄生)。另 One-Forcing 本次核实实为 fake-score 宿主(修正 T1)。

**收窄后可幸存的表述**:"在 **live 生成器骨干**(随蒸馏更新)的多层中间特征上用**纯 multiscale MLP 头** + **同 t 同噪声配对**的具体组合无先例"——且该组合的每个差异点(宿主/头型/配对)都有对应的消融空缺可由我们补(见 6.1)。

检索词(实际执行,节选):`discriminator design diffusion distillation feature backbone ablation` / `generator backbone feature discriminator video distillation 2025 2026` / `video latent discriminator spatio-temporal head one-step distillation` / `multiscale MLP discriminator transformer features video generation` / `AAD-1 asymmetric adversarial distillation one-step autoregressive video discriminator design` / `R1 regularization video GAN distillation approximated` 等 17 组。

### 4.2 主张(判别器 2):"同 timestep 同噪声配对判别(gan_use_same_t_noise)在少步蒸馏文献中未被单独消融过"——**支持**

5 组针对性检索 + 逐篇核对未发现任何单独消融。最近相邻证据与差距:ASD 明确对真假分支**独立**采样 ε(Alg.1,已逐条核实)且消融不含配对方式;Diffusion2GAN 消融的是条件注入(z 条件 +0.07 FID)而非扰动共享;AAD-1 同 τ 但 ε 未说明;LADD/NitroFusion 消融的是判别 t 的分布/取值。两点限度:"支持"限于论文文献(开源 repo 未扫);"未被消融"≠该设计新颖或有效——同 t 是行业默认,真正未定价的是共享 ε 半边,**这构成我们可自己补做的低成本独家消融(same-ε vs independent-ε)**。

检索词:`"same noise" discriminator real fake ablation diffusion distillation noised latents` / `shared noise injection real generated samples GAN loss video few-step distillation ablation` / `"gan_use_same_t_noise" OR "same_t_noise" discriminator distillation github` / `discriminator identical noise perturbation real fake pair "ablation" one-step video generation` / `noise conditioned discriminator same timestep pairing GAN distillation`。

### 4.3 主张(轴 A 组件层):"在 4-8 步视频 DM 蒸馏内部,把推理锚点形状当显式变量做多形状系统消融并给出选择准则的工作不存在"——**部分支持**

严格合取(视频 T2V + 4-8 步 + ≥3 形状受控 + 准则)下无反例:DOLLAR/CoDMD/SGMD/CausVid 系/Adaptive Video Distillation 全部单一固定 t_list 无形状消融(SGMD 用 shift≈8 却零消融;AVD 明确只消融正则项)。但三个近反例显著压缩表述空间:**TMD**(同基座,shift 开/关两档消融 + "VBench 测不出的 collapse")、**Lip Forcing**(单锚点 4 位置系统扫描 + 显式准则,但 2-step 唇同步子域)、**FlashMol**(ρ 4 档扫描 + 准则,但分子域)。**写作红线**:必须限定为"4-8 步通用 T2V 的多形状锚点消融与准则",不能宽泛说"锚点从未被当变量"——TMD(同基座)会被审稿人立刻举出。DOLLAR 的等距不崩(CogVideoX)vs TMD 的无 shift 即崩(Wan)提示敏感性依赖基座 flow-shift 预训练,结论要绑定 Wan 系。

检索词:`timestep anchor ablation distillation student few-step video` / `denoising step list choice distilled video model ablation 2026` / `schedule shape ablation distillation video diffusion shifted uniform` / `Wan distillation "timestep shift" ablation few-step student 4-step schedule` / `"shift" 5 "t_list" OR "timestep list" 999 937 833 624 four step video distillation` / `learned timestep schedule student diffusion distillation 2025 optimize anchor points training` 等 11 组 + 10 篇全文。

### 4.4 附带更正与确认(带日期,2026-07-06)

- **更正 T1**:One-Forcing 判别器宿主为 fake score backbone(非生成器);ADM/DMDX venue 确认为 **ICCV 2025 Highlight**(T1 标记待核实)。
- 确认:SenseFlow = ICLR 2026 Poster(OpenReview API);Seaweed-APT = ICML 2025(PMLR v267);POSE(=V-PAE)= AAAI 2026;TDD = AAAI 2025;DMD2 = NeurIPS 2024 Oral(T1 已核,不变)。
- Few-Step SiD、TMD、Salt、Data-Forcing、AAD-1、SGMD、Flash-DMD、Taming DiT、FlashMol、Lip Forcing、CDM 等 venue 均待核实(留 T3 复核清单)。

> **Planner 验收注记(2026-07-06,远端代码核实,推翻本报告一处前提)**:本报告 4.1 收窄主张与 5.1 叙事句均以"我们判别器寄生在 live 生成器骨干"为前提,该前提沿自 T0 的未核实描述。planner 已读 `fastgen/methods/distribution_matching/dmd2.py`:判别特征实由**冻结 teacher** 的 forward 提取(fake 分支复用 VSD 的同一次 teacher forward;real 分支 `return_features_early` 截断),属本报告 2.3 节路线 C(LADD / teacher-feature 谱系),先例充分——**4.1 的收窄主张与 5.1 句中 "lives on the student generator's intermediate features" 一并作废,不得使用**。幸存并升级:4.2 主张经代码坐实且更精确——`gan_use_same_t_noise=True` 时 real/fake 共享同一 t 与同一 ε(方法默认 False,是 WanT2V 配置的主动选择);另核实 FastGen 已内置近似 R1(`gan_r1_reg_weight`,默认 0.0 且我们全部 run 未启用;`gan_r1_reg_alpha=0.1` 与 APT 视频档 σ 同量级)——2.3 节弱点 1 的"零正则"对我们的 run 成立,且 R1 消融只需改配置、零代码。

---

## 5. 对投稿叙事的可用表达(组件层,5 条)

1. **GAN 配置的社区共识定位句**:"Our adversarial component follows what is emerging as the community recipe for DMD-style Wan distillation — a lightweight feature-space discriminator with generator-side weight 0.03 and a 5:1 fake-to-generator update ratio, independently converged upon by concurrent work (One-Forcing) — while differing in three deliberate choices: the discriminator lives on the *student generator's* intermediate features (layers 15/22/29) rather than the fake-score network, uses multiscale MLP heads, and pairs real/fake samples under the *same timestep and the same noise*, a pairing whose effect we ablate for the first time."
2. **relay 与稳定性理论的接口句**(引 ADM):"ADM attributes reverse-KL instability to insufficient support overlap between student and teacher distributions; our step-count relay directly manufactures this overlap — the 4-step student is initialized from a converged 8-step student — which we argue reduces the burden on the fake-score tracking loop and explains why the standard 5:1 two-time-scale rule remains stable in our setting."
3. **fake 端重置的跨工作一致性句**:"Three independent lines of evidence support re-initializing auxiliary networks at stage boundaries from pretrained rather than distilled weights: Seaweed-APT's discriminator initialization finding, Phased DMD's per-phase fake-score reset from the teacher, and SenseFlow's diagnosis that the fake network's inner best-response is the brittlest part of DMD training."
4. **锚点消融的空缺句**(轴 A 收窄版):"Anchor shape for multi-step DM students is inherited, not justified: DMD2/TDM/DOLLAR use uniform spacing, community Wan recipes use shift-5, SGMD silently uses shift-8 — none ablates shape. The only controlled evidence is TMD's binary shift on/off (1-2 steps) and sub-domain single-anchor scans (Lip Forcing); we provide the first multi-shape anchor ablation with a selection rule for 4-8 step video DM distillation, and show that aggregate VBench scores fail to detect the mode collapse it induces."
5. **GAN 项的诚实定位句**(预置 Data-Forcing 攻击的回应):"Because concurrent work reports that a GAN branch can be net-negative for dynamic degree in this exact setting (Data-Forcing, Wan2.1-1.3B/4-step), we treat the discriminator not as a given but as an ablated component: we report GAN weight {0, 0.03} × approximate-R1 {on, off} with dynamic degree and cross-seed diversity as primary metrics."

红线(组件层不能说的):不能说 0.03/层选择/同 t 同噪声"经过验证"(均无消融,第三项恰是我们要补的);不能把"判别器寄生 diffusion 骨干中间层"说成我们首创(LADD/DMD2/ASD/POSE/Taming DiT 全在此谱系);不能引 APT 的 batch/LR 数字暗示适用于 DMD 小 batch 语境;不能说 TTUR 5:1 是视频最优(视频侧零受控数据,SGMD/Flash-DMD 都在压低它)。

---

## 6. 对后续实验与 T3 的建议

### 6.1 组件升级建议(按 收益/风险/实现成本 排序)

| # | 改什么 | 预期解决的失败模式 | 文献依据 | 与"训练调度轴"故事相容性 | 成本 |
|---|---|---|---|---|---|
| 1 | **判别器近似 R1**(真实分支 σ≈0.05-0.1 扰动,λ 扫 {20,100,600}) | 判别器过强→动态度塌缩、训练后期退化 | APT(无它即崩)、ASD/AAD-1/POSE(Wan 系事实标准)、Diffusion2GAN(-1.37 FID) | 高(组件保守增强) | 低 |
| 2 | **IDA 用于 4-step relay 阶段**(student 更新后 fake←0.97·fake+0.03·student) | fake score 重置后追踪失败(震荡/黑帧)、relay 冷启动 | SenseFlow(w/o IDA 17.83 vs 13.38;开销 +0.6~4%) | **极高——可写成 relay 协议的组件**(阶段间重置 + 阶段内连续对齐) | 极低 |
| 3 | **Data-Forcing 式 post-training**(4-step 最优 ckpt 上 100-300 iter,teacher score 求值点 50% 换 OpenVid 真实 latent) | 多样性坍缩、过饱和(reverse-KL 丢模式零梯度) | Data-Forcing(同基座:camera trajectory diversity +349%;须以 DMD2 ckpt 初始化) | 高(post-hoc 步骤,不改主配方) | 低 |
| 4 | **判别头时空聚合升级**(learnable-query cross-attn 池化到 video-wise logit,或每层加 1D temporal 分支) | 动态弱、静态平滑偏置 | POSE(+16.3)、AAD-1(frame-wise→静态)、Taming DiT(+4.4)、SF-V(FVD 180.9 vs 514.7) | 高 | 中 |
| 5 | **ISG 段内接力自蒸馏**(teacher CFG=5 单步 Euler 到 t_mid + 冻结 student 拼 target) | 高噪锚点(0.999/0.937/0.833 落在 xi(t) 最震荡区)监督次优、早期不稳 | SenseFlow(FID-T@1.5k:14.48 vs 138.2;与 shifted 锚点兼容) | 高(锚点不动,正交叠加) | 中(+3~6%/iter) |

条件触发项(不进主列表):8-step 阶段若"步数多反而差"→ Salt SC 正则(同基座 82.78→83.36);GAN/DM 梯度冲突 → Flash-DMD 高低噪解耦;训练成本紧张 → fake-EMA + TTUR 降档(图像证据,需验证);下探 2/1 步 → ASD 邻步对齐。**不建议**:ADM 式对抗主项替代(喧宾夺主)、VFM 判别器整体替换(解码成本 + FID-T 代价)、CDM 连续 schedule(收益小无视频证据)。

**最值得先做的一个组件实验**:4-step relay 阶段的 **GAN 权重 {0, 0.03} × 近似 R1 {on, off} 受控消融**,以 Dynamic Degree + 跨 seed 多样性为一级指标——同时回答"判别器组件的去留"(Data-Forcing 的同基座质疑)、补 One-Forcing 缺失的 GAN on/off 证据、并为弱点 1 的修复定价;单实验服务三个论文素材点。

### 6.2 给 T3 的幸存主张清单与检索词

**幸存主张(收窄版,交 T3 终裁)**:
1. 判别器组合主张(收窄):"live 生成器骨干多层特征 + 纯 multiscale MLP 头 + 同 t 同噪声配对"的组合无先例——T3 需盯 lightx2v/FastVideo 等生态 repo 的判别器实现与 2026H2 新 arXiv。
2. 同 t 同噪声配对无单独消融(支持)——T3 补扫开源 repo 配置旗标后可升级为"文献与公开实现均无"。
3. 轴 A 收窄版:"4-8 步通用 T2V 的多形状锚点消融 + 选择准则"空白——T3 需专门反查 TMD 附录(其 shift 消融是否覆盖 4-step 档)与 2026H2 新条目。
4. relay 特有空白(轴 C 支线):4-step 重置后的 TTUR/warmup/EMA 全部零文献覆盖——属"我们必须自补实验"而非"可 claim 的空白",T3 只需确认无新工作抢占。

**T3 新增检索词**:`generator backbone discriminator video distillation 2026H2` / `same noise pairing discriminator ablation` / `TMD transition matching appendix shift ablation 4-step` / `Wan distill GAN discriminator config github lightx2v FastVideo` / `IDA proximal fake score video DMD 2026` / `anchor shape ablation video few-step 2026`;venue 复核清单见 4.4。

**T3 数字复核清单**(本报告引用但需二次目检的):Seaweed-APT Figure 9(batch 消融)与 Table 1/4 人评差值;One-Forcing framewise 收敛步数单位与判别器数据集名称;AnyFlow 全部数字(全文不可达);ADM 判别器头结构(附录);Self-Forcing 官方 configs(LR/clip);AAD-1 ε 共享与 ICML 2026 状态。

---

## 附:本报告的证据边界

- 高相关条目均打开 arXiv 原文页核实;SenseFlow 另经官方代码核对(标注"official code"的超参不出自论文正文)。二手来源与未打开条目均已逐处标"待核实"。
- 精读 5 篇与 T1 的 5 篇零重复;T1 已核事实(DMD2/Phased DMD/CoDMD/GPD 机制数字)直接引用未重查。
- 本报告为组件层初判;三条主张的终裁、生态 repo 扫描、venue 复核均属 T3。
