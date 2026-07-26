# Ch1 草稿:配方、加速与量化方法论(50→4 蒸馏工程线,2026-07-26 初稿)

> 状态:故事线级初稿(2026-07-26),支撑 2026-07-28 进度汇报;数字待 2026-07-27 晚 planner 验收后冻结。风格与红线同 `thesis_ch2_draft.md`:方法措辞只用"步数接力 / step-count relay / progressive step reduction"(禁用 phased/progressive DMD);全部单阶段超参承自 NVIDIA FastGen 公开 WanT2V 配置(§1.2 原文声明);checkpoint 一律 best-of-sweep 口径;全文禁写"学生超越 teacher"(高于 teacher 的单维一律按静态/锐度偏置联读,见 §1.6/§1.7)。
> 数字出处:`reports/experiment-report-wan21-t2v-dmd2-progressive.md`(历程/配置/速度)、`experiments/results/2026-07-14-e0-full-table-g1.md`(E0 全表 + G1 裁决 + dm40)、`experiments/results/2026-07-20-g2-relay-vs-direct-final.md`(冠军档)、`research/STATUS_2026-07-20.md`(协议资产)。

## 1.1 任务与本章结论

**任务**:把公开 Wan2.1-T2V-1.3B 的 50-step teacher(CFG=5,双前向 ≈100 NFE)以 DMD2 目标蒸馏为 4-step 学生(无 CFG,4 NFE),数据为 OpenVid-1M;项目原始构思为步数接力(50→8→4),即先训 8-step 中间学生、再以其最优 checkpoint 初始化 4-step 阶段。

**本章结论先行**:

1. **加速确立**:单条 480p×81 帧视频生成从 165.24s 降至 6.59–6.63s,≈25×,与 NFE 比(≈100:4)一致(§1.4)。
2. **方法论转折是本章主线**:2026-07-14 的存量 checkpoint 量化(E0)证明肉眼选档系统性不可靠——两条训练线的"肉眼最佳档"都被量化推翻;全文由此改为 best-of-sweep + 早停选点,"先量化、后主张"成为后续一切对照实验(第 2、3 章)的前置纪律(§1.5)。
3. **评估协议有效性有正面证据**:均匀 t_list 消融(W4)在本基座复现教科书级 mode collapse——一致性维全表最高而成像/多样性/动态全表最低,证明协议中的 DD/多样性维能抓到 consistency 类指标掩盖的坍缩(§1.6)。
4. 接力必要性的受控裁决在第 2 章(结论:匹配预算与配方下无质量收益、多样性劣于直蒸);判别器机制审计在第 3 章。本章交付的是它们共同依赖的配方事实与量化方法论。

## 1.2 基座、上游与步数接力配方

**上游关系(对外表述,承 T3 §6.2,原文可直接使用)**:

> 我们的全部训练基于 NVIDIA FastGen(NVlabs/FastGen,Apache-2.0),复用其原生 DMD2 实现与官方 Wan2.1-T2V-1.3B 配置——包括 teacher CFG=5、生成端 GAN 权重 0.03、real/fake 共享 timestep 与噪声(`gan_use_same_t_noise=True` 为官方 Wan 配置出厂值)、teacher 第 15/22/29 层特征上的 multiscale MLP 判别器、`student_update_freq=5` 的 two-time-scale 更新,以及 4-step `t_list=[0.999, 0.937, 0.833, 0.624, 0.0]`。在此之上,我们的配方贡献限于训练日程层:官方仓库仅提供从 50-step teacher 一次蒸到 4-step/2-step 的单阶段配置,我们改为 50→8→4 的 step-count relay,新增 8-step 中间 student 阶段,并规定 4-step 阶段仅继承 8-step 最优 checkpoint 的生成器权重、优化器/fake score/判别器全部重新初始化;数据侧选用 OpenVid-1M(上游不绑定数据集)。

判别器表述全文统一为:**冻结 teacher backbone 第 15/22/29 层特征 + 可训练 multiscale MLP 头**(2026-07-06 远端代码核实);不称"改进/修改了 FastGen 的 DMD2",不把任何单阶段配方要素(t_list、GAN 权重 0.03、判别器层号、(t,ε) 配对旗标、TTUR、CFG=5)表述为我方设计。

**接力协议(我方新增的训练日程层)**:

- 阶段 1(8-step):teacher 起训,t_list 为高噪密集插值 9 锚点;取最优 checkpoint 作接力源。
- 阶段 2(4-step):仅继承阶段 1 生成器权重(`key_map={"net":"net"}`);优化器、fake score、判别器全部重新初始化;4-step t_list 同上游单阶段配置。
- checkpoint 每 500 iter 存档,统一 sweep 选点(§1.5)。

## 1.3 训练历程 W1→W7:压缩为动机叙事

2026 年 6 月的七条 run 是本项目全部方法论决定的来源。逐 run 超参以远端各 run 目录 `config.yaml` 为 ground truth;此处按"它教会了我们什么"压缩:

| run | 配置要点(config.yaml 已核) | 当时肉眼结论 | 量化后的地位(2026-07-14 E0) |
|---|---|---|---|
| W1 | 4-step 直蒸基线;LR 1.25e-6 / batch 8 / 6000 iter | 闭环通,`0001000` 最佳 | 弱配方直蒸参照;肉眼选档被推翻(@1500 在 5/6 质量维反超 @1000) |
| W2 | 8-step 首训;7 卡续训至 2530 | 明显模糊(单条 13.16s) | **远端 artifact 已消失,只作背景不作证据** |
| W3 | 8-step,`student_update_freq=2`;LR 1.25e-6 / batch 10 / 1000 iter | (流程中断未评) | 近静态+噪声型退化(DD(q150) 0.113 / imaging 0.395);短训低 LR 预期内的观察点 |
| W4 | 8-step,**均匀 t_list**;LR 1.25e-6 / batch 10 / 1500 iter | (流程中断未评) | **教科书级 mode collapse,评估协议有效性正面证据**(§1.6) |
| W5 | 8-step,LR 1e-5(上游出厂值)/ batch 12 / 2500 iter | "进入可用" | 接力初始化源(`0002500`,≈500 次 student 更新);aes 0.559 / imaging 0.657 / div 0.595 |
| W6 | 4-from-8 第一轮;LR 1e-5 / batch 12 / 6 卡 / 2500 iter | `0000500` 最好,后期物理规则崩坏 | 未入量化主线(与 W7 三因素混淆) |
| W7 | 4-from-8 第二轮;LR 5e-6 / batch 16 / 8 卡 / 2500 iter;GAN 0.03 / `gan_use_same_t_noise=True` / R1 off | "随 iter 单调改善" | 接力代表臂;**"单调改善"被量化推翻**(§1.5);第 2、3 章的 relay 对照臂 |

两条工程教训直接塑造了后续实验设计:

- **教训 1(归因纪律)**:W6→W7 同时改了 LR(1e-5→5e-6)、batch(12→16)、GPU(6→8)三个因素,"batch 提升带来改善"之类说法不可归因。此后立死规则:**每次实验只改一个主要因素**——第 2 章两条直蒸臂与第 3 章三臂审计的单变量设计都源于此(第 3 章两臂的 config.yaml 已由 planner 逐值核对,各自只有一个字段偏离 W7)。
- **教训 2(证据纪律)**:W2 的远端目录已消失,"它比 W1 模糊"只剩本地索引记载——不可复核的 artifact 一律降为背景。此后每个 run 固定"一个 config + 一个远端 run 目录 + 一条结果注记"三件套。

## 1.4 加速结果:165s → 6.6s(≈25×)

| 模型 | 采样步数 | CFG | NFE | 单条平均时延(10 prompts,2026-06-15) |
|---|---|---|---|---|
| teacher Wan2.1-T2V-1.3B | 50 | 5.0(双前向) | ≈100 | 165.24s |
| 4-step 学生(W1/W7 同构推理路径) | 4 | 无 | 4 | 6.59–6.63s |

加速比 ≈25×,与 NFE 比(100:4)一致,即时延收益几乎全部来自步数与 CFG 的删减、无额外推理开销。分辨率 480p、81 帧。**注**:速度数字已于 2026-07-26 直读远端原始文件复核(`wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/metrics.csv`):teacher 165.24s 精确一致;student 全 sweep 实读 6.591-6.656s(speedup 24.83-25.07×)——正文所引 6.59-6.63 覆盖前段档位,范围上限是否改写为 6.66 待 planner 裁。

## 1.5 方法论:肉眼选档被量化系统性推翻 → best-of-sweep + 早停

项目 6 月阶段的质量结论全部来自肉眼(10 条域内 prompt);2026-07-14 的 E0 量化(12 模型 × q150 六维 + 多样性)把两条训练线的肉眼记录同时推翻:

**背离 1:W7 的"随 iter 单调改善"不成立。**q150 上 aesthetic 随 iter 单调下降(@500 0.5768 → @2500 0.5379),subject/bg/motion 平缓或微降,仅 DD 上升;量化最优在 @500–@1000,而非肉眼选定的 @2500:

| W7 档 | aes | imaging | diversity |
|---|---|---|---|
| @500 | **0.5768** | 0.6935 | 0.5984 |
| @1000 | 0.5592 | **0.6971** | 0.6125 |
| @1500 | 0.5433 | 0.6832 | 0.6092 |
| @2000 | 0.5477 | 0.6938 | 0.6033 |
| @2500(肉眼最佳) | 0.5379 | 0.6896 | 0.6061 |

**背离 2:W1 的肉眼档也错了。**肉眼选 @1000,量化 @1500 在 5/6 质量维反超(subject 0.9695>0.9588、bg 0.9594>0.9530、motion 0.9894>0.9758、aes 0.5357>0.5103、imaging 0.6847>0.6341)。

两处背离方向相反(一条早停不足、一条选晚了),说明这不是某次观察失误而是肉眼协议本身不可靠。可能解释:肉眼在 10 条域内 prompt 上主要盯"物理规则修复",该性质不被域外 q150 六维捕捉——两者不矛盾,但对外结论一律采用量化口径。

**由此确立的选点方法论(全文执行)**:

1. checkpoint 每 500 iter 存档,同一 prompt 集全 sweep 打分,**best-of-sweep 选点,肉眼档一律弃用**;
2. 质量普遍在早期档(@500–@1000)见顶后回落——relay、直蒸、弱直蒸全同构(该现象的候选机制归因见第 3 章判别器审计),故 sweep 必须覆盖早期档,**早停是默认预期而非异常**;
3. "肉眼不可靠"反过来成为 E0 前置量化的方法动机:任何对照实验(第 2 章 G2、第 3 章三臂)启动前,现有 checkpoint 必须先过同一量化协议。

## 1.6 W4 案例:均匀 t_list 坍缩与评估协议有效性

W4(8-step,把训练锚点改为均匀 t_list)是评估协议的"阳性对照":

| 模型 | subject | bg | imaging | aes | DD(q150) | diversity |
|---|---|---|---|---|---|---|
| W4 uniform-t @1500 | **0.9745** | **0.9791** | 0.2555 | 0.3181 | 0.1867 | 0.4617 |
| teacher(参照) | 0.9661 | 0.9642 | 0.6918 | 0.5899 | 0.3000 | 0.7321 |

W4 的 subject/background consistency 为 E0 全表最高,而 imaging、aesthetic、DD、diversity 全表最低——高一致性 + 低成像 + 低动态 + 低多样性 = 坍缩为静态劣化样本。两点含义:

1. **协议有效性**:若只看 consistency 类指标(乃至以 consistency 为主要权重的聚合分),W4 会被误判为最优模型;DD 与跨 seed 多样性维抓到了被掩盖的坍缩。这在本基座独立复现了 TMD 报告的"均匀 t_list 致 VBench 总分测不出的 mode collapse"(措辞注意:我们不主张 t_list 消融的首创性,该轴为 TMD 占据;W4 仅作协议有效性证据)。
2. **联读规则(全文红线)**:学生 subject/bg consistency ≈ 或 > teacher **不是质量反超**,而是少步学生帧间变化小的静态偏置(与 W4 坍缩同向、程度轻)。全文任何 consistency 数字必须与 DD/diversity 联读,禁止单独表述为"学生一致性胜过 teacher"。

## 1.7 评估协议摘要(消融协议四件套 + 主表协议)

日常消融协议(第 2、3 章全部结论的口径;工具链 `exp/eval/`,独立 `e0eval` 环境,prompt 集 md5 固定、抽样准则入库):

1. **q150 六维**:VBench 官方 `all_dimension` 套件确定性抽样 150 条(md5 `690f2919`),custom-input 模式打 6 个质量维(subject/background consistency、motion smoothness、dynamic degree、aesthetic、imaging)。sweep 为 seed0 单点;**冠军档一律补 n=3(seed 0/1/2)置信带**。
2. **dm40 清洁 DD(动态度双口径的可引用侧)**:q150 的 DD 被套件内 still/frozen 类 prompt 混淆(teacher q150-DD 仅 0.300,服从"静止"指令反而压分;dm40 上 teacher 0.625)。dm40 = 40 条 motion 导向 prompt(20 条官方 `human_action.txt` uniform stride + 20 条 `all_dimension.txt` 经 MOTION_CUE 正则过滤并排除 STATIC_BLOCK;md5 `324d75a0`)。**口径规则:DD_clean(dm40)可引用;q150-DD 仅脚注级表内相对读**。
3. **d40×8 跨 seed 多样性**:40 prompts × 8 seeds,平均成对 LPIPS-alex(越高越多样;md5 `b4c1f9e3`)。这是本项目证明的主退化轴(teacher 0.732 → 学生 0.59–0.64),必须主动报告。
4. **RAFT 连续光流幅值(运动幅值双口径)**:dm40 域,像素/帧;**中位数为主读、均值并报**(teacher 中位 2.75 / 均值 5.16,重尾分布,单口径会失真)。动机:二值 DD 对好学生饱和到 ~1.0(天花板),无法分辨臂间运动差异。多 seed 纪律(2026-07-23 定稿):**臂间方向按逐 seed 配对报告(如 W7>E1a 4/4 seed 同向),单 seed 绝对百分比禁止单独引用**(seed 间中位可差 6 倍,初始噪声主导部分动态水平)。
5. **主表协议**:full VBench standard mode(官方 946 prompts × 5 seeds);已完成 16 维中的 12 维,缺失 4 维均为 GRiT 依赖维(color/object_class/multiple_objects/spatial_relationship,detectron2 未装,用户裁决缓议)。**12 维恰含官方 Quality Score 全部 7 个质量维,可按官方权重合成 Quality Score;Semantic 与 Total 因缺维不可算,表内显式声明**。temporal_flickering 官方协议为专属 75-prompt 子集、**25 样本/prompt、且经 static_filter 预筛**(官方原文"sample 25 videos to ensure sufficient coverage after applying the static filter",出处 Vchitect/VBench `prompts/README.md`@master,2026-07-26 取证);我们为 standard mode 5 样本、未执行 static_filter,该维与官方口径不可互比,须脚注或弃维。与文献数字(如 CoDMD 84.46)同页必须脚注协议差异,禁一切 SOTA 对比表述。
6. **通用口径规则**:q150 / dm40 / vb946 三个域的数字不跨表混引;训练健康指标(loss 曲线)不是质量证据;每实验只改一个主要因素;人评(T2VHE 式 vs teacher)在计划协议中但未执行,列为 limitation。

## 1.8 效度威胁(如实)

- 速度数字(165.24s / 6.59–6.63s)已复核(2026-07-26 直读远端 `metrics.csv`,路径见 §1.4 注;teacher 精确一致,student 上限 6.63→实读 6.656 的 0.026s 勘误待 planner 裁)。
- W2 远端 artifact 已消失,其"模糊"结论只作背景;W5 相对 W2–W4 同时改了 LR 与 batch,8-step 段内部存在归因混淆——不影响"W5 作为接力源够好"的工程判断,但 8-step 段不下配方结论。
- W6→W7 三因素混淆使 6 月阶段的任何配方归因失效;补救即第 2、3 章的单变量受控设计。
- 原计划协议含 CD-FVD(分布级指标),实际未执行;本章协议摘要按实际执行口径书写(缺口清单第 3 项)。
- q150 为域外 prompt 集;单基座、单数据集、单规模(1.3B/480p);无人评;best-of-sweep 粒度 500 iter,可能错过档间真实峰值。

## 1.9 小结

本章交付三件事:(i) 一个在公开基座 + 公开上游配方上端到端走通的 4-step 蒸馏配方,25× 加速(165.24s → 6.59–6.63s);(ii) 一次被数据强制的方法论转向——肉眼选档在两条线上被独立推翻,全文改为 best-of-sweep + 早停,并以 W4 阳性对照确认协议能抓到 consistency 掩盖的坍缩;(iii) 一套 compute-light、seed-controlled 的少步蒸馏退化审计口径(六维 + 清洁 DD + 跨 seed 多样性 + 连续光流,md5 固定、全脚本入库),它是第 2 章"接力 vs 直蒸"受控裁决与第 3 章判别器审计共同的度量基础。
