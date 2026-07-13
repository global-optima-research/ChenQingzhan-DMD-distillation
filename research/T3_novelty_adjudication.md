# T3 调研报告:novelty 对抗核实与竞品裁决

- 生成:2026-07-07(T3 任务书 2026-07-06 定稿版执行)
- 方法:14 路联网对抗核实并行(5 主张裁决 + 6 竞品精查 + venue/数字/评估协议 3 复核),约 486 次检索/原文抓取;OpenReview 网页与 API 当日被 Cloudflare 拦截,venue 以 iclr.cc/icml.cc/CVF openaccess/AAAI OJS/arXiv comments/papercopilot 抓取数据交叉核实
- 输入:`research/T0_project_analysis.md`(1-3 节)、`research/T1_video_fewstep_distillation_landscape.md`(4/2.5/6.2 节)、`research/T2_dmd2_component_neighbors.md`(4/6.2 节,含 4.1 作废注记)
- 立场:对抗性;默认主张可被击破,活下来的才算 novelty

---

## 1. Executive Summary

1. **轴 A(t_list 形状消融首创)终裁:不成立。** 指定复核点 TMD(arXiv [2601.09881](https://arxiv.org/abs/2601.09881),**CVPR 2026 已核**,同基座 Wan2.1-1.3B)逐字核实后远超 T1/T2 记录:Appendix A.2 确认 shift 作用于学生**推理时间网格**(t_list 本体);Table 5 在 2-step 做均匀 vs γ=10 对比并报告均匀致 VBench 测不出的 mode collapse;**Table 11(App. B.6)在 3-step 与 4-step 上对比 γ=5 vs γ=10 两档密集度(4-step:84.60 vs 84.53)**;App. B.2 给出高噪曲率准则。"形状作显式变量 + 覆盖 4-step + 多于两档 + 经验准则"四要素全被同基座论文占据。t_list 降级为配方说明。
2. **轴 B(统一 DMD2 目标步数接力)终裁:部分成立,条件项已解除但被上游严重压缩。** lightx2v step_distill 深查确认:单段一次到位 data-free 纯 DMD(无 GAN、无 8-step 中间站、无接力),"两段式"疑云消解;12 组对抗检索未发现任何"每阶段统一 DMD2 目标 + 沿步数轴接力"的视频先例。**但 NVlabs/FastGen 公开仓库的 [WanT2V config_dmd2.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/experiments/WanT2V/config_dmd2.py) 已公开我方全部单阶段配方细节**(t_list=[0.999,0.937,0.833,0.624,0.0]、GAN 权重 0.03、判别器层 [15,22,29]、`gan_use_same_t_noise=True`、TTUR 1:5、CFG=5)——我方可主张的机制面**只剩 50→8→4 接力编排 + 8-step 中间阶段 + 跨阶段重置协议 + OpenVid-1M 实例化**。
3. **轴 C(三因素受控归因)终裁:部分成立。** "batch 因素在视频 DMD 无先例"被 Data-Forcing([2606.18478](https://arxiv.org/abs/2606.18478),Cosmos I2V,batch 16 vs 128 受控)击破;幸存:三因素**联合**受控设计无先例,且 **per-anchor 有效生成器更新量是三因素中唯一在图像与视频均无受控先例的变量,应作主打**。定位措辞:排除优化混淆的归因卫生学,不与目标级归因争"根因"。
4. **主张 D(同 t 同 ε 配对判别无消融)终裁:成立,并升级为"论文与公开实现均无"。** 12 仓库源码级扫描(tarball grep + Sourcegraph 全局索引):无任何消融;且实现三分裂——DMD2 官方全独立采样、FastGen 全部 15 个公开配置共享、Self-Forcing 谱系判别器更新共享 t+ε 而生成器侧仅共享 t。红线:该设计与 True 取值是上游 FastGen 出厂配置,价值仅为可自补的独家消融。
5. **主张 E(relay 重置后调度零覆盖)终裁:部分成立。** 字面"零覆盖"被 1.x-Distill([2604.04018](https://arxiv.org/abs/2604.04018),阶段重置+分阶段 Adam betas+500 iter warmup)、Phased DMD(每 phase fake 从 teacher 重置)、TMD(Stage 2 新训 fake/判别器)击破;幸存"无系统研究/消融",只能作动机句/limitation,不能作 novelty。
6. **竞品未击穿轴 B。** 全生态(lightx2v/FastWan/CausVid/Self-Forcing 系/Causal Forcing 系/Causal-rCM/Phased DMD/TurboDiffusion/AnyFlow/SGMD/CoDMD/官方闭源 turbo)无一做"步数递减接力 + 仅生成器继承 + 辅助网络重置"。但两个重压:**FastWan 在同基座 1.3B 上 3-step 直蒸成立且三全开**(权重+训练配方+数据),CoDMD 同基座 50→4 直蒸——"必须接力"的必要性动机不成立,50→4 vs 50→8→4 受控对照成为全文生死消融;**Causal-rCM 的"前阶段 student 初始化后阶段 DMD 生成器"与我方继承结构同形**(防线:它换目标 CM→DMD、GAN-free、AR 轴)。
7. **Phased DMD 经核实无正式接收**(ICLR 2026 投稿已撤回,papercopilot 抓取 OpenReview 数据;二手来源,引用前必点原文)——但其已被 lightx2v **产品化**为 Wan2.2-Lightning 全线基础技术,命名红线的现实约束力更强,phased/progressive DMD 措辞绝对不可用。
8. **评估协议:平移可行,可作次级贡献。** 15 组对抗检索未发现 standalone"少步蒸馏退化评估协议"论文,生态位未被正式占用;但组件全部已发表(Data-Forcing 事实协议与我方最小协议几乎同构),只能定位为"compute-light distillation-degradation audit"的标准化整合,绝不能包装为新 benchmark/新指标。

---

## 2. 主张裁决表

| Claim | 裁决 | 幸存表述(收窄后) | 关键证据 |
|---|---|---|---|
| 轴 A:t_list 形状系统消融首创 | **不成立** | 降级为配方说明;唯一未占据的残余问题:relay 跨阶段锚点集嵌套关系(仅 future work) | TMD Table 5/11 + App. A.2/B.2/B.6(击破) |
| 轴 B:统一 DMD2 目标 50→8→4 接力完整配方 | **部分成立** | "首次以同一 DMD2 目标贯穿步数接力每一阶段 + 仅生成器继承/全重置 + 公开基座与数据实例化";须显式声明单阶段超参全部承自 FastGen 上游 | FastGen 仓库无 staged 配方(幸存);config_dmd2.py(压缩);TurboTalk/Causal Forcing++(压缩) |
| 轴 C:LR×batch×per-anchor 三因素受控归因无先例 | **部分成立** | 三因素**联合**受控 + per-anchor 更新量主打变量;删除"优化超参消融在视频 DMD 无先例" | Data-Forcing batch 消融(击破单因素);CARV/DASH/AC-DMD(per-anchor 近邻,均不占据) |
| 主张 D:同 t 同 ε 配对判别无单独消融 | **成立**(升级) | "论文与公开实现均无消融,且实现三分裂;我们首次受控消融 paired vs independent (t,ε)";不得暗示设计为我方提出 | 12 仓库源码扫描;DMD2 官方独立 / FastGen 配置共享 / Self-Forcing 谱系 D 共享 G 半共享 |
| 主张 E:relay 重置后 TTUR/warmup/EMA 零覆盖 | **部分成立** | "无系统研究";只作动机句/limitation,必须引 Phased DMD 与 SDXL-Lightning 重置先例 | 1.x-Distill/Phased DMD/TMD(击破字面);SenseFlow 16 篇引用无 relay 语境(幸存) |

### 2.1 轴 A——不成立(2026-07-07)

**裁决理由**:TMD 消融覆盖远超 T1/T2 记录(原记录"shift 开/关两档、仅 1-2 步"错误,需回填更正):(1) App. A.2 明确 shift 作用于学生推理时间网格 0=s_0<…<s_N=1,即 t_list 本体;(2) Table 5:2-step 均匀(w/o shift)vs γ=10,均匀致 severe mode collapse 且 VBench 测不出(Fig. 9);(3) **Table 11:3-step 与 4-step 各对比 γ=5 vs γ=10(4-step:84.60 vs 84.53)——直接覆盖我方 4-step 档、多于两档**;(4) App. B.2 准则:t≈1 高噪区轨迹曲率大,故用更大 γ。周边再压缩:Adaptive Sampling Scheduler(ICLR 2026 投稿 [GLOOoWqbCV](https://openreview.net/forum?id=GLOOoWqbCV),一致性蒸馏 2-8 步 SNR 驱动锚点,图像域,疑被拒但公开在先)、[2603.17671](https://arxiv.org/abs/2603.17671)(梯度化 discretization 搜索,含 video flow matching)、CDM([2605.06376](https://arxiv.org/abs/2605.06376),连续化取消锚点)。另核实:Causal Forcing++ 4-step 用与我方**完全相同**的 t_list(shift=5 社区标配)且零消融。

**替代表述(配方定位,非 novelty)**:"我们采用社区标准 shift=5 锚点(与 Causal Forcing++/lightx2v warp 后值一致),其高噪密集形状与 TMD 报告的准则(高噪区曲率大、均匀网格致 mode collapse)一致。"唯一未被占据的残余研究问题:**step-count relay 中锚点集跨阶段嵌套关系**(4-step t_list 是否为 8-step 的子集)对权重继承质量的影响——只能写 future work,不可作 contribution。

检索(6 组新角度,避开 T1 已用词):sampling schedule search/learned few-step distilled video anchor ablation 2026;denoising_step_list ablation Wan;arXiv 2606-2607 timestep schedule ablation;uniform vs shifted appendix;中文技术社区 t_list 消融;OpenReview adaptive sampling scheduler。覆盖:arXiv 至 2607;TMD 正文+附录逐点;ICLR 2026 投稿检索;LightX2V 中英文档。未覆盖:CVPR 2026 接收名单全量;闭源工业报告。

### 2.2 轴 B——部分成立(条件解除,上游压缩)

**幸存核心**:12 组对抗检索(2601-2607 号段 + ICML/CVPR 2026 + OpenReview)未发现"每阶段统一 DMD2(reverse-KL+GAN)+ 沿步数轴接力"的视频先例;NVlabs/FastGen 仓库(README/methods 文档/configs 全目录/WanT2V 全部 15 个配置)**无任何 progressive/staged/relay 步数递减配方**(WanT2V 默认 4-step、注释 2-step,无 8-step、无 stage 命名),NVIDIA 官方博客仅单次蒸馏(14B,50→2)+ causal warm-start;故我方接力协议非上游文档复现。lightx2v 条件项解除(见 3.1 卡)。

**压缩项(必须写进论文)**:上游 [configs/experiments/WanT2V/config_dmd2.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/experiments/WanT2V/config_dmd2.py) 已公开:input_shape [16,21,60,104]、`multiscale_down_mlp_large` 判别器 + teacher 层 [15,22,29]、gan_loss_weight_gen=0.03、guidance_scale=5.0、4-step t_list、`gan_use_same_t_noise=True`、lr 1e-5、[methods 级](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/methods/config_dmd2.py) student_update_freq=5——**我方全部单阶段细节均为上游公开值,不得表述为本项目设计点**。概念空间另被 TurboTalk([2604.14580](https://arxiv.org/abs/2604.14580),DMD 得 4 步后对抗式 4→1 渐进接力,avatar 域、二段目标非 DMD)与 Causal Forcing++(一致性初始化→DMD,AR 轴)压缩;[2606.00658](https://arxiv.org/abs/2606.00658)(Wan2.2 dual-expert,噪声轴)、RMD(分辨率轴)正交。

**收窄版(可主张)**:"在公开 Wan2.1-T2V-1.3B 上,首次以同一 DMD2 目标(reverse-KL distribution matching + frozen-teacher-feature multiscale GAN heads, LADD-style)贯穿 step-count relay(50→8→4)的每一阶段:每阶段仅继承上一阶段 student 的生成器权重,优化器、fake score network 与判别器全部重新初始化,并给出基于公开数据 OpenVid-1M 的端到端实例化。"必须随附:单阶段目标与全部超参承自 NVIDIA FastGen 公开 WanT2V DMD2 配置,本工作的增量仅为接力调度、8-step 中间阶段与跨阶段重置协议。

待核实:LongCat-Video-Avatar 1.5([2605.26486](https://arxiv.org/abs/2605.26486))"advanced step distillation to 8 NFE"的具体分段方式(仅读到摘要,avatar 域);2606.00658 全文;OpenReview 在审匿名投稿漏检风险(投稿前一个月复查)。

### 2.3 轴 C——部分成立

**击破**:Data-Forcing([2606.18478](https://arxiv.org/abs/2606.18478))在少步视频 DMD(Cosmos-Predict2.5-2B I2V)内做受控 batch 消融(16 vs 128,Table 3/4)——"优化超参消融在视频 DMD 无先例"不再可用。fake-score 更新率消融在图像早有(DMD2 App. C Fig. 9 已目检确认 1/5/10;SenseFlow;ADM)。

**幸存**:(1) 三因素**联合**受控设计在 2604-2607 全号段无一例(CoDMD lr 2e-6/batch 128/freq 5 全固定;SGMD/Salt/AMD/Alice v1/One-Forcing 均无超参敏感性研究);(2) **per-anchor 有效生成器更新量**(同 iter 数下 4-step 每锚点更新数为 8-step 两倍)在图像与视频均无受控先例——最近邻 CARV([2605.21489](https://arxiv.org/abs/2605.21489),timestep 重要性采样,教师期望方差缩减)、DASH(per-timestep 损失加权,图像)、AC-DMD([2605.26108](https://arxiv.org/html/2605.26108),fake score "limited compute budget"论述,无消融)均不占据;(3) 归因竞争:无 2026 工作证明"优化配置可消除 reverse-KL 退化"(未被抢)也无"优化配置无关"(未被压),但 DFD/CoDMD/SGMD/Salt/rCM 一致把根因钉在目标/数据层——本轴只能写成 "disentangling optimization-side confounds from objective-level failure modes",不争"根因"。

硬前提不变:不补受控实验则此轴从论文主张撤下。写作注意:DFD 的 batch 消融是 I2V(Cosmos)非 Wan T2V,可注明单因素性质与域差。

### 2.4 主张 D——成立(升级为"文献与公开实现均无")

**源码级扫描(12 仓库,tarball grep)**:
- [NVlabs/FastGen](https://github.com/NVlabs/FastGen):旗标存在,methods 级默认 False,**全部 15 个公开实验配置显式 True**;全仓 grep "ablat" 零命中。Sourcegraph 全局索引该旗标仅 FastGen、[NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)(移植版,默认 False,唯一公开 recipe 设 false 且 GAN 权重 0)与一个个人仓库,均无消融。
- [tianweiy/DMD2](https://github.com/tianweiy/DMD2/blob/main/main/sd_guidance.py):real/fake 各自独立调用 compute_cls_logits,**t 与 ε 均独立采样**。
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing/blob/main/model/gan.py)/[Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)(gan.py 逐字继承)/[One-Forcing](https://github.com/Aurora-edu/One-Forcing/blob/main/model/one_forcing.py):**判别器更新硬编码共享同一 critic_noise+critic_timestep;生成器侧 GAN loss 仅共享 t、ε 独立**——同仓库内部不一致且无旗标无消融;One-Forcing 论文三项消融均不涉及。
- FastVideo:DMD2 训练完全无 GAN 项;lightx2v 训练端(DMD/D-OPSD trainer)无 GAN 无真实数据分支;Wan2.2-Lightning/AAD-1 纯推理;SAD(ASD)仓库仅 README;Taming DiT 未开源(待核实)。

**配套论文侧补强**(数字复核项 10):AAD-1([2606.03972](https://arxiv.org/abs/2606.03972))公式记号层面 real/fake 共享同一 ε 与 τ(无明文设计声明、无消融)——"共享 t+ε"不能单独作 novelty,只能作配方细节并引 AAD-1 为先例风险。

**可主张版本**:"real/fake 判别分支的 (t, ε) 配对腐蚀在少步蒸馏论文与公开实现中均无单独消融,且公开实现三分裂(DMD2 全独立 / FastGen 配置全共享 / Self-Forcing 谱系 D 共享 G 半共享);我们首次受控消融 paired vs independently sampled (t, ε)。"红线:设计与 True 取值源自上游 FastGen 公开配置。

风险:若 NVIDIA 日后发布含该旗标消融的 FastGen 技术报告,主张 D 直接被击破——投稿前须复查(当前无 arXiv 报告;检索摘要曾给出的 arXiv 2601.18111 经打开证伪为天气预报论文,系幻觉,不可引用)。

### 2.5 主张 E——部分成立

字面"零覆盖"被击破:**1.x-Distill**([2604.04018](https://arxiv.org/abs/2604.04018),图像,DMD→对抗两阶段:生成器继承、判别器新初始化、Adam betas 分阶段 (0,0.999)→(0.9,0.95)、新模块 warmup 500 iter;HTML 摘录,引用前核对 PDF)、**Phased DMD**(每 phase fake score 从 teacher 重置、5:1 恒定)、**TMD**(Stage 2 fake/判别器新训练、优化器分阶段)。幸存:**"same-objective step-count relay 重置后的 TTUR/warmup/fake-EMA 无系统研究"**——SenseFlow IDA 的 16 篇引用(Semantic Scholar API,2026-07-07)无一进入多阶段/relay 语境。另注意 SGMD(fake 更新 5→1)与 FSF-DMD([2605.19256](https://arxiv.org/abs/2605.19256),移除 fake score)正在消解 TTUR 问题本身,压缩此点长期价值。用途限定:implementation/ablation 动机句或 limitation,必须引 Phased DMD 与 SDXL-Lightning 重置先例。

---

## 3. 竞品精查卡(Wan 生态,核实日期均为 2026-07-07)

**总裁决:无竞品击穿轴 B(无一做步数接力 + 仅生成器继承 + 辅助网络重置);但 FastWan/CoDMD 的同基座直蒸成立使"接力必要性"动机不成立,必须靠 50→4 对照消融说话。**

| 竞品 | 目标族 | 步数 | 基座 | 状态 | 开源 | 轴B重叠 |
|---|---|---|---|---|---|---|
| lightx2v StepDistill-CfgDistill | DMD(无 GAN,data-free) | 4,无 CFG | Wan2.1-**14B**(无 1.3B DMD 版) | HF 2025-06-16 起 | 权重+训练代码(Self-Forcing-Plus)+文档 | 中 |
| Wan2.2-Lightning / Phased DMD | DMD(SNR 子区间 MoE) | 4,无 CFG | Wan2.2-A14B(论文另含 Wan2.1-14B) | 2025-08 起;论文无正式接收(ICLR 2026 撤稿,待核实) | LoRA 权重;**训练代码不开源** | 高(叙事/命名) |
| FastWan / FastVideo(-QAD) | DMD(DMD2 参考实现;博客无 GAN 描述) | **3** [1000,757,522] | **Wan2.1-T2V-1.3B(同基座)** | 2025-08-04;QAD 2026-06-23 | **三全开**:权重+配方+合成数据 | 中 |
| NVIDIA FastGen(上游) | DMD2(完整,含判别器/R1 旗标) | 4(1.3B 配置)/2(14B 博客) | Wan2.1 1.3B/14B/Wan2.2-5B 等 | repo 2026-01-15 创建 | 训练代码全开;无权重发布 | 高(上游) |
| CausVid | asymmetric DMD + ODE init | 3-4,均匀 [999,748,502,247] | 官方 repo 为 Wan2.1-**1.3B**(含**双向 3-step checkpoint**) | CVPR 2025 | 权重+训练代码(CC-BY-NC-SA) | 中 |
| Self-Forcing | DMD/SiD/GAN 三选一(主发布 DMD) | 4 AR chunk-wise;t_list warp 后=我方值 | Wan2.1-T2V-**1.3B** | NeurIPS 2025 Spotlight | 全开源(Apache-2.0) | 中 |
| Self-Forcing++ | DMD 长时域扩展 | 4(推断) | Wan2.1-1.3B(AR 化) | ICLR 2026 Poster(iclr.cc 已核) | 无代码无权重("soon") | 低 |
| Causal Forcing / ++ | 三阶段:AR 化→ODE/CD 蒸馏→asymmetric DMD | 4 / 1-2 frame-wise | Wan2.1-**1.3B**(+14B) | ICML 2026 Poster / ++ 仅技术报告 | 权重+全套训练代码 | 低 |
| Causal-rCM | TF-CM→SF-DMD 顺序串联(GAN-free) | 1/2/4 | Wan2.1-1.3B(causal 化) | 技术报告 2026-06-24 | 训练代码(NVlabs/rcm,Apache-2.0);student 权重待核实 | 中 |
| AccVideo | 合成轨迹+对抗(非 DMD) | 5(Hunyuan)/10(WanX-14B) | HunyuanVideo/Wan-14B,无 1.3B | preprint | 权重+推理;无训练代码 | 低 |
| Wan 官方 | 未公开(闭源 API turbo) | 未公开 | — | wan2.1-t2v-turbo→wan2.7 全闭源 | 零开源少步产物 | 低 |
| TurboDiffusion/TurboWan | rCM+量化+稀疏注意力(非 DMD) | 1-4 | 含 Wan2.1-T2V-**1.3B** | arXiv 2512.16093 | 权重+推理;训练代码 roadmap | 无 |
| AnyFlow | flow-map(非 DMD、无 GAN) | any-step 4-32 | Wan2.1 1.3B/14B(+FAR) | preprint([2605.13724](https://arxiv.org/abs/2605.13724)),代码权重 2026-05-04 | 代码+权重 | 无 |
| SGMD | DMD 目标重构(无 GAN) | 4,{1000,960,889,727} | Wan2.1-**14B** | **ICML 2026 Poster(已核)** | 代码宣称入 LightX2V,未见入口(待核实) | 低 |
| CoDMD | DMD+copula 关系正则 | 4(50→4 直蒸) | Wan2.1-T2V **1.3B**/14B | preprint 2026-06-20 | 未指明(待核实) | 中 |
| TMD | DMD2-v + flow head | few-step(2/3/4 消融) | Wan2.1 **1.3B**/14B | **CVPR 2026(CVF 已核)** | 代码/权重未确认(待核实) | 低 |
| Data-Forcing | DMD 数据级正则(post-training) | 4 | Cosmos-2B I2V(+Wan2.1-1.3B 实验) | preprint(NVIDIA 系) | — | 低 |
| Kijai/WanVideo_comfy | 纯转制(1.3B 四条线 LoRA) | 随上游 | 覆盖 1.3B | 持续更新 | LoRA | 无 |

### 3.1 lightx2v StepDistill(轴 B 最高优先条件项——**条件解除**)

- 训练代码 = [GoatWu/Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus)(yaml 逐字核实):distribution_loss: dmd,teacher CFG=4.0 烤入 real score,dfake_gen_update_ratio: 5,EMA 0.99;**无 GAN 键、data-free(仅 30k 文本 prompt)**。denoising_step_list=[1000,750,500,250] + warp + shift 5.0 → 锚点 = [1.0,0.9375,0.8333,0.625],与我方 t_list 数值相同(同一社区标准)。
- **一步到位 50(40)→4,无 8-step 中间 student、无接力、无组件重初始化**;README 明言 bidirectional DMD 不需 ODE init(generator_ckpt: ode_init.pt 被注释)。T1 记录的"两段式"两种解读均非步数接力:(a) ODE-init+DMD 是原始 causal Self-Forcing(1.3B)流程,lightx2v bidirectional 产线跳过;(b) StepDistill-CfgDistill 双后缀指 CFG 蒸馏集成于同一 DMD run。
- 产线全为 14B;**lightx2v 从未发布 bidirectional Wan2.1-1.3B 的 DMD step-distill 权重**。注意监控:lightx2v/Wan2.1-T2V-1.3B-Distill-Models(HF 2026-06-13,无模型卡,归"FewStep RL" collection,大概率 GenRL/GRPO 后训练非蒸馏,**待核实,T4 前复查**)。
- 划界:不能 claim 4-step 无 CFG Wan T2V DMD 蒸馏本身 / t_list / TTUR 5:1 / CFG 烤入。链接:[HF](https://huggingface.co/lightx2v/Wan2.1-Distill-Models) / [docs](https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html)。

### 3.2 Wan2.2-Lightning / Phased DMD(命名与叙事最大威胁)

- Phased DMD([2510.27684](https://arxiv.org/abs/2510.27684)):SNR 子区间 progressive distribution matching + 子区间 score matching + MoE experts;4 步切 2 phase;**渐进轴是 SNR 子区间(固定最终步数),不是步数递减**;无 GAN(待核实其辅助项细节);fake 每 phase 从 teacher 重置。基座 Wan2.2-A14B/Wan2.1-14B/Qwen-Image,无 1.3B。**训练代码不开源**([ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning) 已 301 至 LightX2V-Wan2.2-Lightning,纯推理仓)。
- venue:**ICLR 2026 投稿已撤回,无正式接收**(papercopilot 抓取 forum zzJTo7ujql status=Withdraw;OpenReview 直连被拦截——二手来源,**引用前必点原文**)。注意:内部 agent 曾误标 CVPR 2026,以本条为准。
- 官方声明其为 2025-09-28 起 lightx2v 全部 Wan2.2 蒸馏模型的基础技术——**命名红线产品化坐实**:phased/progressive DMD 绝不可用;审稿人必然追问我们与其多步稳定性方案(stochastic gradient truncation 批评)的对比,该论述反可作我方 8→4 接力的动机对照文献。

### 3.3 FastWan / FastVideo(同基座最强开源竞品)

- **直接命中基座**:FastWan2.1-T2V-1.3B(480P,基座 Wan-AI/Wan2.1-T2V-1.3B-Diffusers),"sparse distillation" = DMD(官方文档引 DMD2 参考实现)+ VSA 稀疏注意力联合训练;**3-step 直蒸 [1000,757,522]**,合成数据(teacher 生成 600k latents);训练代码无 GAN 项(主张 D 扫描确认)。QAD 线:量化感知 DMD 3-step(2026-06-23)。三全开 + 成本公开(1.3B:768 H200 GPU 时)。
- **对我方的杀伤**:不能 claim"首个/唯一 Wan2.1-1.3B 开源 DMD 少步配方";**不能把"1.3B 直蒸少步不可行"作接力动机**(FastWan 3 步直蒸成立)——接力只能写成质量/稳定性取舍且需消融支撑。差异点:真实数据 OpenVid-1M vs 合成 latents;完整 DMD2(含 GAN)vs 无 GAN;接力 vs 直蒸。链接:[博客](https://haoailab.com/blogs/fastvideo_post_training/) / [HF](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)。

### 3.4 NVIDIA FastGen(上游关系,对外表述见 6.2)

- 公开仓库直接包含 Wan2.1-T2V-1.3B DMD2 配置(4-step 默认,与我方 4-step 阶段逐项相同,含近似 R1 旗标 gan_r1_reg_weight 默认 0.0、注释建议 100-1000);博客只展示 14B(50→2,64×H100 16h),从未提 1.3B、从未提 progressive/staged/OpenVid-1M。无 arXiv 技术报告(官方引用格式为 GitHub @misc);无官方蒸馏权重发布(待核实)。
- **必须预防的攻击点**:博客 "warm-starting causal distillation" 段落含"轨迹法初始化→分布匹配"两阶段思想——须引用并区分:那是跨方法族的目标函数接力,我们是同一 DMD2 目标内的步数接力。TTUR 描述须与代码一致:fake score/判别器是每 5 iter 中的 4 次更新,不是"每 iter"。

### 3.5 CausVid / Self-Forcing 系(AR 谱系,含关键 artifact)

- **CausVid**(CVPR 2025,三重确认):论文基座为内部 DiT,**官方 repo 是 Wan2.1-T2V-1.3B 复现版,且顺带发布双向(非 AR)3-step DMD checkpoint(bidirectional model 1/2)**——该谱系中唯一与我方同为"双向少步 Wan1.3B 学生"的公开 artifact,related work 必须点名切割(纯 DMD + ODE init 一次到位、均匀 t 采样 [999,748,502,247]、无 GAN、无接力)。Kijai 的 CausVid LoRA 为社区转制。
- **Self-Forcing**(NeurIPS 2025 Spotlight 已核):DMD/SiD/GAN 三选一非叠加(与 DMD2 的"叠加"不同,写作须写准);**T2 数字复核完成**:lr 2.0e-6(生成器)/4.0e-7(critic),grad clip 为代码默认 max_grad_norm 10.0(不在 config,引用写"代码默认");config real_name: Wan2.1-T2V-14B 暗示 real score 侧可能用 14B(待核实)。我方 t_list 与 TTUR 5:1 与其配置精确相同——只能写"沿用社区标准"。
- **Self-Forcing++**(ICLR 2026 Poster,iclr.cc 已核):长时域外推方向,与我方正交;无代码无权重。
- **Causal Forcing**(ICML 2026 Poster,三重确认)/**++**(**更正:++ 无独立接收,仅技术报告**;ICML 2026 接收的是原论文):三阶段范式串联(AR 化→ODE/causal-CD 蒸馏→asymmetric DMD),同基座 1.3B;++ 的"首帧 4 步、后续 1-2 步"是同一模型内按位置分配步数——若我方把"先多步后少步"写得太宽会被其表面相似性攻击,须锁定"训练阶段间的步数接力、推理全序列统一 4 步"。
- **Causal-rCM**([2606.25473](https://arxiv.org/abs/2606.25473),技术报告):**本批对轴 B 威胁最大**——TF-CM 学生初始化 SF-DMD 生成器,与我方 generator-only 继承同形。防线三条:阶段间换目标(CM→DMD)vs 我方同目标纯步数变量;GAN-free vs 我方判别器配方;AR/流式轴 vs 双向步数轴。VBench:1-step/2-step frame-wise 84.63、4-step chunk-wise 84.37(对比 Self-Forcing 83.76、Causal Forcing 83.96)。student 权重是否上 HF 引用前再查。

### 3.6 其余登记(要点)

- **Wan 官方**:开源侧零少步产物(Wan2.1/2.2 全为满步基座;lightx2v 列于 README "Community Works",非官方方案);闭源 API 有 wan2.1-t2v-turbo 与 wan2.5-preview→wan2.7(2026-06-12 快照)。可用定位:"官方开源少步 checkpoint 缺位,社区蒸馏填补"——对动机段有利;不能说"官方没有任何加速产物"。
- **TurboDiffusion/TurboWan**([2512.16093](https://arxiv.org/abs/2512.16093),thu-ml):rCM 蒸馏+量化+稀疏注意力,TurboWan2.1-T2V-1.3B-480P 端到端 1.9s;速度对比不能直接拿其端到端数字(含系统加成)。T1 的"TurboDiffusion"疑问已查清:学术工作,非 lightx2v 组件。
- **AnyFlow**(NVIDIA,[2605.13724](https://arxiv.org/abs/2605.13724);T1 不可达问题已解决,PDF 本地抽取成功):flow-map any-step;**同基座硬坐标:1.3B 83.54@4NFE / 83.96@32**;Figure 1 记录 rCM 随 NFE 增大严重退化(82.81@2→75.72@32)——评估叙事可用。
- **SGMD**(ICML 2026 Poster 已核):14B、4-step {1000,960,889,727}、无 GAN、fake 更新 5→1(3x 加速);"代码入 LightX2V"宣称未落地(待核实)。不能 claim"DMD2 的 4-step 不稳定/低效是未处理的开放问题"。
- **CoDMD**([2606.21982](https://arxiv.org/abs/2606.21982),preprint):同基座 1.3B/14B 50→4 直蒸,VBench 84.46/84.87 硬坐标;GAN 使用情况未核(待核实)。
- **minWM**([2605.30263](https://arxiv.org/abs/2605.30263)):Wan2.1-1.3B 上复用 Causal Forcing 管线的全栈开源世界模型——又一"多阶段管线+DMD"实例。
- **薄卡**:SwiftVideo(AAAI-26 已核;跨步数 trajectory alignment 是训练内正则非接力;评测用 OpenVid-1M 与我方同源);2606.00658(Wan2.2 dual-expert 蒸馏×量化,8/20 步档,LightX2V 团队)。

---

## 4. Gap 分析与红线

### 4.1 能说(对外口径,7 条)

1. **定位句(轴 B 收窄版)**:同一 DMD2 目标贯穿 step-count relay(50→8→4)每一阶段、仅生成器继承 + 优化器/fake score/判别器全重置、公开 Wan2.1-T2V-1.3B + OpenVid-1M 端到端实例化——据我们所知无完整先例(检索覆盖声明见 2.2)。
2. **正交轴句**:2026 年 Wan 系 DMD 改进集中在目标函数轴(CoDMD/SGMD/Data-Forcing/Phased DMD)与 AR 化轴(CausVid→Causal-rCM 谱系),我们在训练调度轴——保持上游 DMD2 配方逐字段不动,只改阶段编排;两轴正交(triggering:全部竞品卡)。
3. **上游关系句**(全文见 6.2):基于 NVIDIA FastGen 复用其原生 DMD2 实现与官方 Wan 配置;贡献限于训练日程层。
4. **主张 D 句**:paired vs independent (t,ε) 判别腐蚀在论文与公开实现均无消融、实现三分裂,我们首次受控消融(triggering:12 仓库扫描)。
5. **轴 C 句(前提=补实验)**:三因素联合受控归因 + per-anchor 更新量主打;定位为排除优化混淆,不争根因。
6. **主张 E 动机句**:relay 重置后调度无系统研究,我们给出配方并消融(引 Phased DMD/SDXL-Lightning 重置先例)。
7. **评估句**:compute-light、seed-controlled 的少步蒸馏退化审计协议(diversity/motion/distributional drift 三线),开源 prompts/seeds/脚本(引 Data-Forcing/DOLLAR/Phased DMD/AVD 并声明是标准化整合)。

### 4.2 不能说(红线,9 条,各注触发文献)

1. **任何 phased / progressive DMD 措辞**(Phased DMD 2510.27684 + Wan2.2-Lightning 产品化)。**最重要红线**:**不能把任何单阶段 DMD2 配方要素——t_list、GAN 权重 0.03、判别器层 15/22/29、`gan_use_same_t_noise=True`、TTUR 1:5、CFG=5、近似 R1 旗标——说成我方设计**(NVlabs/FastGen 公开 WanT2V 配置逐字段一致)。
2. 不能说"首次对 t_list 形状做系统消融/给出准则"或任何轴 A 表述(TMD CVPR 2026 Table 5/11 + App. B.2)。
3. 不能说"首个/唯一 Wan2.1-1.3B DMD 少步蒸馏或开源配方"(FastWan 三全开、CausVid 双向 3-step checkpoint、Self-Forcing、TurboWan、AnyFlow、CoDMD)。
4. 不能把"1.3B 直蒸 4 步不可行/必须接力"作动机(FastWan 3-step、CoDMD 50→4 直蒸成立);接力必要性只能靠 50→4 vs 50→8→4 受控对照说话,做出来之前是假设。
5. 不能说"优化超参消融在视频 DMD 无先例"(Data-Forcing batch 16 vs 128);不能引 Seaweed-APT batch/LR 数字暗示适用于 DMD(GAN 式;且其 Table 1/4 的 Structural Integrity 全为大幅负值,引用时不能只引 +37.2% fidelity)。
6. 不能暗示 same-t-same-ε 配对设计为我方提出或"共享 t 罕见"(FastGen 出厂 True;AAD-1 公式记号已隐含;Self-Forcing 谱系 D 更新即共享)。
7. 不能宣称"多阶段训练管线 + DMD"或"上一阶段 student 初始化下一阶段"本身新(Causal Forcing 三阶段、Causal-rCM、SDXL-Lightning、Hyper-SD、GPD、TurboTalk、1.x-Distill)。
8. 不能与 CoDMD(84.46)/AnyFlow(83.54@4)/Causal-rCM(84.37-84.63)的 VBench 数字对比性 claim SOTA(我方无量化数字);Alice v1 的 91.2 协议未披露不可比(引用必须注明)。
9. 不能说"DMD 少步失败模式无人诊断"(CoDMD/SGMD/rCM/Data-Forcing/AVD)或首创退化模式分类(AVD 已占 oversaturation/temporal collapse/mode collapse 三分类);禁止一切无检索背书的"首个/首创"绝对化表述。

### 4.3 必引划界清单(26 条:一级 12 + 二级 14;另加 2 条阶段重置先例引用义务)

**一级(正面对照,每条给"本质区别"/"不能 claim")**:

| # | 工作 | 我们与它的本质区别 | 我们不能 claim 的部分 |
|---|---|---|---|
| 1 | DMD2(NeurIPS 2024 Oral) | 我们不改其目标,改阶段编排 | 目标函数与 TTUR 的一切 |
| 2 | NVIDIA FastGen(上游,GitHub @misc) | 上游只有单阶段配置,我们加接力/中间阶段/重置协议 | 全部单阶段超参与判别器设计 |
| 3 | Phased DMD(无正式接收,待核实) | SNR 子区间 MoE vs 步数接力单一 student | phased/progressive DMD 命名;分阶段 DMD 概念 |
| 4 | CoDMD(preprint,同基座) | 目标级正则 vs 调度级编排 | "DMD 4 步退化无人处理";定量优势 |
| 5 | rCM(ICLR 2026) | GAN-free 一致性竞线;其 diversity 批评须正面回应 | 忽略 mode collapse 批评自称质量无损 |
| 6 | GPD(preprint,同基座 48→6 接力) | 纯轨迹回归 vs DMD2 统一目标 | 步数接力协议本身 |
| 7 | TMD(CVPR 2026,同基座) | 其做锚点消融与 flow head 架构,我们沿用其准则 | 一切 t_list 消融首创性 |
| 8 | FastWan(同基座 3-step 直蒸,三全开) | 合成数据无 GAN 直蒸 vs 真实数据完整 DMD2 接力 | 1.3B 少步开源首创;直蒸不可行 |
| 9 | Data-Forcing(preprint) | 数据级正则修复 diversity vs 调度级;其 GAN 负收益质疑须以消融回应 | 视频 DMD 优化消融无先例(其 batch 消融) |
| 10 | CausVid(CVPR 2025,官方双向 3-step 1.3B ckpt) | ODE init 一次到位纯 DMD vs 接力 DMD2 | 双向少步 Wan1.3B artifact 首创 |
| 11 | Self-Forcing(NeurIPS 2025 Spotlight) | AR rollout 训练 vs 双向接力;t_list/TTUR 沿用其标准 | t_list、TTUR 5:1、1.3B 4 步可行性 |
| 12 | Causal-rCM(技术报告) | 阶段间换目标(CM→DMD)+AR 轴 vs 同目标步数轴 | "前阶段 student 初始化后阶段生成器"结构本身 |

**二级(related work 覆盖)**:13. lightx2v StepDistill(部署事实标准;t_list 出处);14. Wan2.2-Lightning(Phased DMD 产品化);15. SDXL-Lightning(接力协议+判别器每阶段重置原型);16. Hyper-SD(8→4→2→1 接力);17. SwiftVideo(AAAI-26;跨步数对齐概念);18. Seaweed-APT(ICML 2025;稳定性数字,引用须带负值面);19. SGMD(ICML 2026;TTUR 反向证据、shift≈8 锚点);20. AnyFlow(同基座 any-step 基线数字);21. TurboDiffusion(rCM 1.3B 权重);22. Causal Forcing(ICML 2026)与 ++(技术报告);23. Self-Forcing++(ICLR 2026);24. AccVideo(合成轨迹+对抗路线);25. One-Forcing(配置最近的 DMD+GAN,fake-score 宿主);26. AAD-1(ICML 2026;same-t-same-ε 记号先例)。
**阶段重置先例对(主张 E/轴 B 划界句内引用)**:1.x-Distill(2604.04018)、TurboTalk(2604.14580)。

### 4.4 替代窄带主张候选(任务书问题 6)

1. **paired-corruption 判别消融**(主张 D,证据最硬):实现三分裂 + 零消融,单实验即成独家素材;证据需求:same-(t,ε) vs same-t-indep-ε vs 全独立三档消融。
2. **per-anchor 有效更新量归因**(轴 C 主打):图像与视频均无受控先例;证据需求:8-step vs 4-step 同 iter 对照 + LR/batch 固定,与 CARV/DASH 显式区分"采样哪个 t"vs"每锚点更新几次"。
3. **relay 锚点集嵌套**(轴 A 残余,仅 future work 级):8-step t_list 是否嵌套 4-step 对继承质量的影响;文献未占据但我方无证据,不作 contribution。

### 4.5 载重条目——引用前必点原文清单

| 条目 | 风险 | 动作 |
|---|---|---|
| Phased DMD 撤稿状态 | papercopilot 二手抓取;内部曾出现"CVPR 2026"误标(已裁定以撤稿说为准) | 投稿前换 IP 直开 OpenReview forum zzJTo7ujql |
| TMD Table 11 / App. B.2/B.6 | 击破轴 A 的唯一载重证据 | 引用/写 related work 前人工目检 CVF PDF |
| FastGen WanT2V config | 决定"上游压缩"表述;repo 活跃(最近 push 2026-06-07) | 投稿前 diff 最新 main;并 diff 本地克隆确认旗标非版本差异 |
| 1.x-Distill 阶段配方细节 | HTML 自动摘录 | 逐字核对 PDF |
| Salt = ECCV 2026 | 仅作者 arXiv comments 声明 | ECCV 官方列表挂网后复核 |
| lightx2v Wan2.1-1.3B-Distill-Models | 无模型卡;若为 DMD 蒸馏则同基座最近邻 | T4 前复查是否补卡 |
| CoDMD 的 GAN 使用与开源 | 未核正文 | 精读方法节 |
| MagicDistillation = AAAI 2026 | AMiner 二手,官方 OJS 被拦截 | AAAI proceedings 复核 |
| Causal-rCM student 权重 | README 只有本地文件名 | 引用前查 HF |
| 2606.00658 / LongCat-Video-Avatar 全文 | 仅摘要级;含"8 NFE step distillation"表述 | 全文复核分段结构 |

---

## 5. 评估协议平移建议

**最小可行协议**(结论:全组件 pip/repo 可用,1.3B/480p/81f 规模可行):

- **主表**(teacher 50-step / 8-step / 4-step 三个最终模型,一次性):full VBench 16 维,946 prompts×5 seeds(946 出处已核实:[Vchitect/VBench prompts/README.md](https://github.com/Vchitect/VBench/blob/master/prompts/README.md) + all_dimension.txt 实测 946 行;**temporal flickering 维需 25 视频/prompt,全套约 6,230 段**,勿写成 946×5=4,730);人评按 [T2VHE](https://github.com/ztlmememe/T2VHE)(NeurIPS 2024)成对协议,启用 dynamic evaluation module(官方称省约 50% 标注成本)。
- **消融**(多组低预算):(1) VBench custom_input 的 6 个 prompt-free 维度(subject/background consistency、motion smoothness、dynamic degree、aesthetic、imaging),150-200 prompts×1 seed;(2) 分布:[CD-FVD](https://github.com/songweige/content-debiased-fvd)(CVPR 2024,pip cd-fvd,VideoMAE-2 特征),以 teacher 输出与 OpenVid 真样本为双参照,≥1k 段;样本更少的组改报 [JEDi](https://github.com/oooolga/JEDi)(ICLR 2025,约 FVD 16% 样本量收敛);(3) **多样性必报**:40 prompts×8 seeds,1−平均 pairwise cosine(DINOv2 帧特征,Data-Forcing 式),全文固定同一指标;(4) **运动必报**:VBench Dynamic Degree + UniMatch 平均光流(Phased DMD/SGMD 事实标准)。
- **可选扩展**:FVMD(workshop 级,辅助);VBench-2.0 Physics/Commonsense(仅主表);oversaturation 的 HSV 统计偏移(自建,辅助证据);I3D-FVD 附录兼报以降审稿摩擦(正文以 CD-FVD 为准,引 CVPR 2024 content-bias 论文)。

**协议贡献可行性:可行,次级贡献定位。** 未发现 standalone 退化评估协议论文(15 组检索);但 Data-Forcing 已打包近同构事实协议、DOLLAR 占用视频 Vendi 定义、AVD 占用退化三分类、Phased DMD/SGMD 确立 DD+光流——只能写成"对既有指标的标准化整合 + 消融级小预算多样性审计"(卖点:rCM 的多样性证据仍是定性样例、CoDMD 只靠人评,量化审计在头部工作确实缺位)。卖点句:"a compute-light, seed-controlled evaluation recipe that jointly audits the canonical failure modes of few-step video distillation — per-prompt diversity collapse, motion degradation, and distributional drift — at ablation-scale budgets, with released prompts, seeds, and scripts."(必引 Data-Forcing/DOLLAR/Phased DMD/AVD。)

---

## 6. 对写作与后续实验的建议

### 6.1 T1/T2 回填更正清单(由 planner 执行,均注日期 2026-07-07)

1. **T1 4.2 / T2 2.2、4.3(TMD 记录,重大)**:原"shift 开/关两档、仅 1-2 步"→ 实为 Table 5(1/2-step)+ **Table 11(3/4-step,γ=5 vs γ=10)** + App. B.2 准则 + App. A.2 确认作用于推理网格;TMD venue = CVPR 2026。轴 A 相关表述全部作废。
2. **T0 §1 / T2 4.4 planner 注记(gan_use_same_t_noise,重大)**:"方法默认 False、我方主动配置 True"在配方层面误导——上游 WanT2V 实验配置出厂即 True(全部 15 个公开实验配置均 True);不得作为设计差异点。
3. **T1/T2 全部单阶段超参表述**:t_list、0.03、层 15/22/29、CFG=5、TTUR 1:5、lr 1e-5 均为 FastGen 上游公开配置值;凡列为我方机制/配方设计的表述删除或改注"沿用上游公开配置"。
4. **T1 2.5/4.3(lightx2v 条件项)**:"两段式"疑云解除,无步数接力;lightx2v 无 1.3B DMD 蒸馏产物。
5. **T2 4.2(Self-Forcing 谱系 ε 精确化)**:D 更新共享 t+ε、G 侧仅共享 t;"共享 t"在该谱系是普遍做法。
6. **venue 批量回填**:Phased DMD = ICLR 2026 撤稿无接收(待核实);SGMD = ICML 2026 Poster;Salt = ECCV 2026(作者声明);SwiftVideo = AAAI-26;TMD = CVPR 2026;GPD/CoDMD/Data-Forcing/Alice v1/AnyFlow/CDM/Few-Step SiD/FlashMol/Lip Forcing = 无接收 preprint;f-distill = 未见接收(待核实);OSV = CVPR 2025;SF-V = NeurIPS 2024;UFOGen = CVPR 2024;AAD-1 = **ICML 2026 Poster(由"疑投稿"上调)**;Flash-DMD = ICLR 2026 撤稿;Taming DiT = ICLR 2026 被拒;Self-Forcing = NeurIPS 2025 Spotlight(确认);CausVid = CVPR 2025(确认);Causal Forcing = ICML 2026 Poster;**Causal Forcing++ = 无独立接收(仅技术报告,由"疑 ICML 2026"下调)**;Self-Forcing++ = ICLR 2026 Poster。
7. **数字批量回填**:Seaweed-APT batch 消融 = §5.4 + Figure 9(256 崩/1024 稳;最终视频 batch 2048;Table 1/4 含大幅负值项);DMD2 Fig 9 = Appendix C(确认);LiveTalk = **arXiv 2512.23576**,崩溃归因 = 条件数据质量 + ODE init 不足 + 短学习窗口(非 GAN/TTUR);TDM 80.91→81.65 确认(CogVideoX-2B,提升主要在 Semantic);VBench 946 出处落地;Alice v1 91.2 = 协议未披露不可比;One-Forcing 收敛单位 = "training steps"、**判别器数据集原文未披露**(若记录过名称须更正);AnyFlow = 2605.13724,数字已抽取(1.3B 83.54@4);ADM 判别器头 = App. B(conv+GN+SiLU,每 3 DiT block 一头,宿主为 score estimator 侧);AAD-1 ε 记号层面共享。
8. **T2 轴 C 证据库**:补 Data-Forcing batch 消融(击破单因素无先例)与 1.x-Distill/TMD 阶段重置配方先例。

### 6.2 与上游 FastGen 关系的对外表述(可直接使用)

> 我们的全部训练基于 NVIDIA FastGen(NVlabs/FastGen,Apache-2.0),复用其原生 DMD2 实现与官方 Wan2.1-T2V-1.3B 配置——包括 teacher CFG=5、生成端 GAN 权重 0.03、real/fake 共享 timestep 与噪声(gan_use_same_t_noise=True 为官方 Wan 配置出厂值)、teacher 第 15/22/29 层特征上的 multiscale MLP 判别器、student_update_freq=5 的 two-time-scale 更新,以及 4-step t_list=[0.999,0.937,0.833,0.624,0.0]。在此之上,我们的配方贡献限于训练日程层:官方仓库仅提供从 50-step teacher 一次蒸到 4-step/2-step 的单阶段配置,我们改为 50→8→4 的 step-count relay,新增 8-step 中间 student 阶段,并规定 4-step 阶段仅继承 8-step 最优 checkpoint 的生成器权重、优化器/fake score/判别器全部重新初始化;数据侧选用 OpenVid-1M(上游不绑定数据集)。

不得称"改进/修改了 FastGen 的 DMD2";不得暗示 8-step 中间配置超参为全新设计(除非与上游 4-step 配置逐项 diff 后明示差异);引用用官方 GitHub @misc(不存在可引 arXiv 报告;2601.18111 系检索幻觉已证伪)。

### 6.3 实验优先级(依据本轮裁决重排)

1. **50→4 直蒸 vs 50→8→4 受控对照**(轴 B 生死消融,FastWan/CoDMD 直蒸成立后优先级再升——没有它接力叙事只是假设;同预算同数据同 t_list)。
2. **量化评估先行**:现有 checkpoint 跑 §5 消融协议,把肉眼结论转数字;硬坐标:CoDMD 84.46、AnyFlow 83.54@4、Causal-rCM 84.37、Self-Forcing 83.76。
3. **主张 D 独家消融**:paired vs same-t-indep-ε vs 全独立三档(零代码成本,改配置即可)。
4. **轴 C 三因素受控**(per-anchor 主打;8-step vs 4-step 同 iter 数天然构造)。
5. **T2 遗留组件消融**(GAN {0,0.03} × 近似 R1 {on,off},上游旗标零代码)——回应 Data-Forcing 的 GAN 负收益质疑。
6. 轴 A 消融预算取消(TMD 已占),省出的算力转投 1-2。

### 6.4 投稿前复查清单

- 2607 号段后半月增量复扫(词:video DMD ablation / per-timestep update / step-count relay / paired noise discriminator)。
- OpenReview 换 IP 复核:Phased DMD 撤稿、Causal Forcing ICML、GLOOoWqbCV 状态。
- FastGen repo diff(最近 push 2026-06-07 后的变化;是否新增技术报告/staged 配方/旗标消融——任一出现将分别冲击轴 B/主张 D)。
- lightx2v Wan2.1-1.3B-Distill-Models 是否补模型卡;SGMD 代码是否落地 LightX2V;AC-DMD 是否出视频版。
- Semantic Scholar 引用图滞后 2-4 周,对 SenseFlow/Phased DMD 引用列表做最终增量。

---

## 7. 增补裁决(2026-07-11,用户提供两篇本地 PDF,已联网核实身份并全文精读)

### 7.1 Perceptual Flow Matching(PFM,arXiv [2607.03524](https://arxiv.org/abs/2607.03524),2026-07-03,preprint/WIP,京东 Joy Future Academy + 复旦 + 清华 + 中科大,通讯 Nan Duan)——**高,必须收录**

- **机制**:免蒸馏少步路线(第四族,与 progressive distillation / CM / DMD 并列):保留标准 flow matching 插值,但由 v 预测反解 x̂0、经冻结 VAE decoder 解码后在**预训练感知特征空间**(图像 VGG+DINOv2 最佳;视频 InternVideo2)回归真实样本。**无 teacher、无 fake score、无判别器**;从基座 fine-tune(含 Wan2.1-T2V-1.3B),推理步数自由可调、无固定 t_list、无 CFG;NFE≤2 明显退化(4 步以下未解决)。
- **对 T3 终裁的影响**:轴 A/B/C、主张 D/E 终裁**全部不变**(无锚点消融、无多阶段、无优化归因、无判别器)。冲击在**动机层**:同基座 Wan2.1-1.3B、8 步无 CFG、VBench Total 0.792(其 35 步 baseline 仅 0.774——**协议疑似非标准**,社区通行 ≈0.84,v1 无附录不可核,引用前必查)。
- **动作**:(1) **必引清单 26→27 条**,PFM 列二级,划界语:"同基座同 benchmark 的 distillation-free 替代路线,与本文 teacher 分布匹配接力正交";我们不能 claim 少步生成必须依赖蒸馏/teacher,也不能 claim 推理无 CFG 为蒸馏路线独有收益。(2) **红线 4.2 第 4 条外延**:除"直蒸不可行"外,一并禁止"少步必须蒸馏/必须 teacher"类隐含表述,动机段措辞审查一遍。(3) T1 谱系可补"免蒸馏少步(real-data 目标改造)"第四族分支(与 MeanFlow 区分:无 JVP、无一致性约束)。(4) 可用弱点:其 Dynamic Degree 0.319 < 35 步 baseline 0.379(Table 5)——免蒸馏路线同样动态度受损,支持我们评估协议把 DD/光流放一级指标。(5) 转引其 Table 3 的 DMD2 对比数字必须注明:DMD2 baseline 仅训 1000 步,弱基线设置。
- **待核实**:v1 正文引用 Appendix 但 PDF 无附录(视频数据/超参/VBench 协议全缺);无代码;发布 8 天零被引;跟踪 v2 与投稿 venue 后再决定是否升一级。

### 7.2 LiveEdit(arXiv [2606.26740](https://arxiv.org/abs/2606.26740),2026-06-25,**ECCV 2026**,THU + HKUST)——**中,登记即可,不进必引清单**

- **机制**:流式视频编辑(非 T2V 生成),基座 Wan2.1-T2V-1.3B。其 "progressive three-stage distillation pipeline" 为**范式三阶段**(编辑能力微调 MSE → 双向转因果 causal mask → 单跳 DMD 100 NFE→4 NFE 去 CFG),每阶段目标函数不同;DMD1 风格(MSE + DM 梯度),**无判别器、无步数接力**;t=[0,250,500,750];靠 Stage 2 teacher-forced AR 权重初始化绕开 Self-Forcing 的 ODE init。
- **对 T3 终裁的影响**:全部不冲击。**对命名红线是正面支持**:又一例 "progressive ... distillation" 指范式分阶段而非步数递减,佐证 progressive 一词语义漂移严重、我们坚持 step-count relay 的必要性。注意其 Stage 3 "生成器继承上一阶段权重 + real/fake score 从 teacher 重置"与我们接力交接是弱组件近邻(跨范式初始化,非 DMD 阶段间交接)。
- **动作**:登记于同基座 4-step DMD 无 CFG 实例清单(编辑域);写作时"首个多阶段蒸馏管线"类宽泛措辞的禁用范围再添一个触发文献;related work 仅在需要覆盖编辑方向时可选引用。
- 附带线索:其最近邻 EgoEdit(arXiv 2512.06065,流式第一人称视频编辑)未在 T1/T2 登记,如叙事涉编辑域再补查。

---

## 附:本报告的证据边界

- 全部 2024-07 后条目经实时检索;竞品逐个打开官方 repo/model card/论文原文,核实日期统一 2026-07-07。venue 因 OpenReview 全天被 Cloudflare 拦截,以官方会议站点(iclr.cc/icml.cc/CVF/AAAI OJS)与 papercopilot 抓取数据交叉;papercopilot 属二手,涉撤稿/被拒结论均入"引用前必点原文"清单。
- 主张 D 为源码级证据(tarball grep + Sourcegraph),强于 README 级;未覆盖私有代码、gitee、GitHub 认证搜索。
- 明确未覆盖:CVPR 2026 接收名单全量、ICLR/NeurIPS 2026 在审匿名投稿、闭源工业配方、非英文技术博客系统扫描。
