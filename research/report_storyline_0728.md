# 2026-07-28 进度汇报故事线(页级大纲,2026-07-26 初稿)

> 用法:每页 = 一句主张(可直接做页标题/讲稿第一句)+ 承载表格/图 + 数字出处文件名。数字待 2026-07-27 晚冻结;做 slides 时逐数字对出处文件复核,不引本大纲的转述。
> 总定调(用户裁定,2026-07-21):**避免全负叙事——按轴"各有所长"陈述;三臂审计写成"归因"而非"没用"**。全局红线:方法名只用"步数接力 / step-count relay";禁写任何"学生超越 teacher";checkpoint 一律 best-of-sweep 口径;q150 / dm40 / vb946 三域数字不跨表混引。
> 故事弧(六幕):问题与加速 → 肉眼被量化推翻(方法论)→ G2 受控对照(负结果 + 各有所长)→ 三臂审计四条归因 + E5 → full-VBench 四行表 → limitation 与 future work。

---

## 第一幕:问题与加速

### P1|封面 + 一句话总结
- **主张**:把 Wan2.1-T2V-1.3B 从 50 步蒸到 4 步(165s → 6.6s,≈25×),并用受控实验回答了两个问题:步数接力是否必要(否,多样性反而更差),判别器分支对哪些现象负责(四条归因)。
- **承载**:无表;一行时间线(2026-06 配方期 → 07-14 G1 量化 → 07-20 G2 裁决 → 07-22~25 三臂审计 → 07-26 收口)。
- **出处**:`reports/experiment-report-wan21-t2v-dmd2-progressive.md`(当前状态节)。

### P2|任务与加速:25× 来自步数与 CFG 的删减
- **主张**:50-step teacher(CFG=5,≈100 NFE)→ 4-step 学生(无 CFG,4 NFE),单条 480p×81 帧 165.24s → 6.59–6.66s,≈25×,与 NFE 比一致。
- **承载**:速度对照表(2 行:teacher/学生 × 步数/CFG/NFE/时延);可配 1 组同 prompt 对照帧。
- **出处**:`reports/experiment-report-wan21-t2v-dmd2-progressive.md`(W1 节;速度已于 2026-07-26 直读远端 `wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/metrics.csv` 复核:teacher 165.24 精确一致,student 全 sweep 6.591-6.656、speedup 24.83-25.07×;范围按实读勘误为 6.59–6.66,planner 裁定 2026-07-26)。
- **红线**:速度数字标内部记录出处。

### P3|方法与配方:上游复用 + 训练日程层贡献
- **主张**:全部单阶段配方(DMD2 目标、GAN 权重 0.03、判别器 = 冻结 teacher 15/22/29 层特征 + 可训 multiscale MLP 头、t_list、TTUR)承自 NVIDIA FastGen 公开 WanT2V 配置;我们的贡献在训练日程层——50→8→4 步数接力:仅继承 8-step 最优档生成器,优化器/fake score/判别器全重置。
- **承载**:接力示意图(50→8→4,标"继承什么/重置什么");上游关系一句话(T3 §6.2 原文压缩)。
- **出处**:`research/thesis_ch1_draft.md` §1.2、`research/T3_novelty_adjudication.md` §6.2。
- **红线**:禁 phased/progressive DMD;禁把单阶段超参说成我方设计。

## 第二幕:肉眼被量化推翻(方法论)

### P4|转折点:两条训练线的肉眼最佳档都被量化推翻
- **主张**:W7 的"随 iter 单调改善"被推翻(aesthetic 单调下降 0.577→0.538,量化最优 @500–@1000 而非肉眼 @2500);W1 肉眼选 @1000,量化 @1500 在 5/6 质量维反超——肉眼协议本身不可靠,全文改为 best-of-sweep + 早停。
- **承载**:W7 五档 aes/imaging/diversity 表(高亮单调下降列)+ W1 @1000 vs @1500 两行对照。
- **出处**:`experiments/results/2026-07-14-e0-full-table-g1.md`(全表 + 背离 1)。
- **备注**:讲"质量普遍 @500–1000 见顶后回落、全臂同构"——为第四幕的机制归因埋钩子。

### P5|协议有效性阳性对照:均匀 t_list 的教科书级坍缩
- **主张**:W4(均匀 t_list)consistency 全表最高(subject 0.9745/bg 0.9791)而 imaging 0.256、DD 0.187、diversity 0.462 全表最低——只看 consistency 会把坍缩模型判成最优;我们的 DD/多样性维抓到了它(在本基座复现 TMD 的观察)。
- **承载**:W4 vs teacher 两行六列表(consistency 与其余维反向高亮)。
- **出处**:`experiments/results/2026-07-14-e0-full-table-g1.md`(消融数据点节)。
- **红线**:不 claim t_list 消融首创(TMD 占据);顺势立联读规则——学生高 consistency ≠ 质量优势,必须与 DD/diversity 联读。

### P6|评估协议一页:四件套 + 主表协议
- **主张**:compute-light、seed-controlled 的退化审计口径——q150 六维(冠军档 n=3)/ dm40 清洁 DD(可引用侧;q150-DD 受 still-prompt 混淆仅脚注,teacher 两域 DD 0.300 vs 0.625)/ d40×8 LPIPS 多样性 / RAFT 连续光流(中位主读 + 均值并报,救二值 DD 的天花板);主表 = full VBench 946×5 standard mode。
- **承载**:协议四件套一览表(prompt 集、md5、用途、口径规则)。
- **出处**:`research/thesis_ch1_draft.md` §1.7、`research/STATUS_2026-07-20.md` §2。
- **红线**:三域数字不跨表混引;训练健康指标不是质量证据。

## 第三幕:G2 受控对照(负结果 + 各有所长)

### P7|G2 设计:该基座上首个受控 relay-vs-direct 对照
- **主张**:接力臂(W5 2500 + W7 2500)vs 两条直蒸臂(E1a = W7 配方、E1b = W5 配方,各 5000 iter)——匹配预算/数据/t_list/评估协议,双臂 bracket 令"直蒸没调好"的质疑失效;全臂 best-of-sweep;FastWan/CoDMD 已证直蒸可行但均未做受控比较。
- **承载**:三臂设计示意(时间轴 + 不变量清单)。
- **出处**:`research/thesis_ch2_draft.md` §2.1。
- **红线**:不宣称直蒸首创;贡献表述 = "该基座上首个受控 relay-vs-direct 对照"。

### P8|G2 主表:匹配预算下,接力无质量收益、多样性劣于直蒸
- **主张**:质量判平或直蒸略优(imaging E1a 0.717 为学生最高;aes n=3 带重叠判平);多样性两条直蒸臂(0.635/0.628)双双高于接力两档(0.598/0.613),独立同向——负结果如实报。
- **承载**:G2 冠军档对照表(teacher/E1a/W7×2/E1b/W1,aes/imaging/DD_clean/diversity 四列)。
- **出处**:`experiments/results/2026-07-20-g2-relay-vs-direct-final.md`;n=3 置信带见 `research/thesis_ch2_draft.md` 发现 1。
- **红线**:E1a imaging 0.717 > teacher 0.692 禁写"超越 teacher"(锐度/静态偏置口径);E1a 的 DD_clean 0.75 低于 W7/E1b 的 0.95–1.0 须如实提。

### P9|各轴图景:退化 = 多样性坍缩,而非动态度坍缩
- **主张**:全部学生 DD_clean 0.75–1.0 ≥ teacher 0.625 且 motion-smooth 0.97+(真运动非抖动)——未复现文献担心的少步动态坍缩(划界:域不同,只说"未复现");真正的主退化轴是跨 seed 多样性:teacher 0.732 → 学生 0.59–0.64。
- **承载**:DD_clean + diversity 双列条形图(teacher vs 各学生)。
- **出处**:`experiments/results/2026-07-14-e0-full-table-g1.md`(dm40 结论节)、`experiments/results/2026-07-20-g2-relay-vs-direct-final.md`。
- **红线**:"未复现"≠"反驳"(Data-Forcing 是 Cosmos I2V)。

### P10|发现 5:接力唯一稳定的实测差异是运动幅值(mixed finding)
- **主张**:RAFT 光流上接力臂运动幅值系统性更高——W7 > E1a 在 4/4 seed 同向(中位均值 3.36 vs 1.81,≈1.9×),W7 > teacher 4/4;评价随准则而异(贴近 teacher 分布 vs 最大化动态),本文不选边;二值 DD 完全掩盖该差异。
- **承载**:flow 多 seed 配对表(s0–s3 × W7/E1a + teacher 参照)。
- **出处**:`experiments/results/2026-07-23-flow-multiseed-e1b946-e2a-eval.md` §1/§4、`research/thesis_ch2_draft.md` 发现 5。
- **红线**:臂间方向可引(4/4 同向),**单 seed 绝对百分比(+61%/−22%)禁止单独引用**;E1a 相对 teacher 表述为"不高于(3/4 同向)"。

## 第四幕:三臂审计四条归因 + E5

### P11|三臂设计:两次单变量手术
- **主张**:在 W7(配对 GAN)配方上各改一个字段——E2a 仅 `gan_loss_weight_gen 0.03→0`,E2b 仅 `gan_use_same_t_noise True→False`;其余(LR/batch/iter/W5 初始化)全同;config.yaml 已逐值核对,E2a checkpoint 无判别器分片(自洽)。
- **承载**:三臂差异表(P3 接力图上标注手术位置)。
- **出处**:`experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md`、`experiments/results/acceptance-log.md` #11。
- **红线**:0.03 与 same_t_noise=True 均为上游出厂值——消融的是上游默认设计。

### P12|四条归因(本页是审计叙事的中心)
- **主张**:①运动幅值 = relay 初始化 × GAN 分支交互(E2a 关 GAN 衰减回 teacher 级 2.1–2.7;E2b 开 GAN 重建 2.40→4.71;E1a 直蒸开 GAN 仍低——单独任一均不足);②质量维与 GAN 反向(E2a 随 iter 改善 0.591→0.611/0.701→0.723,GAN-on 臂随 iter 退化)——"早峰后滑"获候选机制归因;③(t,ε) 配对惯例不敏感(同档差 0.01 级、全维走势同构)——Claim D 首个受控消融的如实答案;④多样性对判别器不敏感(三臂 0.586–0.613 均低于 E1a 0.635)——坍缩归蒸馏本身。
- **承载**:两张并排小表:E2a/E2b 五档 sweep(aes/imaging/flow 三列即可)+ 三臂 diversity 区间条。四条归因做四个要点框。
- **出处**:`experiments/results/2026-07-24-e2a-fulltable-ch3.md`、`experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md`。
- **红线**:禁写"GAN 必然抬运动"(E1a 反例);E2a aes 0.613 > teacher 0.590 禁写"超越 teacher";归因≠"判别器没用"。

### P13|E5 层×t 探针:判别信号集中于中低噪端(机制背景)
- **主张**:t ≤ 0.937 时 real-vs-生成全层线性完全可分(AUC=1.0 饱和),t=0.999 全层近随机(0.28–0.52)——悬崖式 t 依赖;上游选层 {15,22,29} 合理但探针无法裁"唯一最优";诚实对照:teacher 自身生成物可分性同构,故可分性≠蒸馏退化度量。
- **承载**:层×t AUC 热图(悬崖可视化)+ FD 随深度曲线。
- **出处**:`research/E5_probe_results.md`(修正口径,权威)。
- **红线**:非 headline 贡献,"to our knowledge" 级;弃用早期"均值 AUC 0.88–0.92"读法。

## 第五幕:full-VBench 四行表

### P14|full-VBench 12 维四行表:按轴各有所长
- **主张**:E1a(G2 加冕)赢一致性/平滑/闪烁类,W7 赢动态度(0.911)与语义动作类(human_action 0.794/scene 0.292),E1b 居中,E2a(审计臂,GAN=0,仅 @2000 单档)静态画质最高(aes 0.6482/imaging 0.6924)且动态不塌(0.80)——与 q150 域三臂结论跨域同向;多样性上界仍属 teacher(0.732,q150 域)。
- **承载**:12 维 × 4 模型主表(高亮各列第一)。
- **出处**:`experiments/results/2026-07-26-e2a-vb946-fourth-row.md`。
- **红线**:加冕叙事按 G2 预注册结果 = E1a@1000(非 cherry-pick);E2a 行标注"审计臂(GAN=0),仅 @2000 单档";Quality Score 已按官方权重合成(出处 Vchitect/VBench `scripts/constant.py`+`scripts/cal_final_score.py`@master:min-max 归一化、DD 权重 0.5、分母 6.5):**E2a@2000 85.50 / W7@1000 84.47 / E1b@500 83.62 / E1a@1000 82.80**——E1a(加冕臂)因 DD 维(0.581,权重 0.5)合成分最低,页面须并列解释权重结构且重申加冕依据是 G2 预注册协议(q150 六维+多样性)而非该合成分;Semantic/Total 因 GRiT 4 维缺失不可算(表脚注声明);**警惕:W7 的 84.47 与 CoDMD 文献值 84.46 数字巧合近同,但前者为 7 维 Quality Score、后者为 16 维 Total,绝对禁止并列比较**;temporal_flickering 官方为专属子集 25 样本/prompt + static_filter 预筛(出处 `prompts/README.md`@master),我们 5 样本未筛,脚注;若同页出现 CoDMD 84.46 等文献数字,必须脚注协议差异、禁 SOTA 对比;dynamic_degree 两域数字(q150 0.567 / vb946 0.800)不混引。

## 第六幕:limitation 与 future work

### P15|Limitations(如实)与 future work
- **主张**:诚实列缺口,每条都有留档的复现路径。
- **承载**:两列清单。Limitations:无人评(T2VHE 式计划未执行);GRiT 4 维缺失(detectron2,故 Semantic/Total 不可算);E2c(R1 臂)受单卡 32G 显存限制未测(确定性 OOM 于 R1 第二次判别器前向);sweep 质量维多为 seed0 单点(冠军档 n=3);单基座/单数据/单规模;flow 对 seed 敏感(单 seed 百分比弃用);GAN 消融为 0.03 单权重点。Future work:GRiT 四维补齐 → Semantic/Total;E2c 于 80G 节点复现(校准值 0.75 与配置已留档 `reports/e5/e2c_r1_calib.json`);人评;直蒸血统 GAN=0 对照臂(裁读 2 的跨血统检验);W5 扩表(8-step 部署档)。
- **出处**:`experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md`(E2c 节)、`experiments/results/acceptance-log.md` #11 队列更新。

### P16|结论页:贡献清单(按 T3 可说口径)
- **主张**:①25× 加速的 4-step 学生 + 完整开源链路(上游复用声明);②该基座上首个受控 relay-vs-direct 对照——负结果如实报:匹配预算下接力无质量收益、多样性劣于直蒸,唯一实测差异是运动幅值(mixed finding);③判别器审计四条归因(含 Claim D:(t,ε) 配对惯例首个受控消融,答案 = 不敏感);④compute-light 退化审计协议(肉眼不可靠 → best-of-sweep;退化 = 多样性坍缩而非动态坍缩;W4 阳性对照)。
- **承载**:四点贡献框;每点右侧标"证据页码"(P8/P10/P12/P4-P6)。
- **出处**:`research/T3_novelty_adjudication.md` §4.1(能说 7 条的压缩)。
- **红线**:每条"首个/首次"都带"据我们所知 + 检索覆盖"限定;不 claim 直蒸首创、不 claim 退化模式分类首创。

---

## 预答问题(备问卡,5 条)

1. **"负结果为什么值得讲?"**——G2 是该基座上第一个受控 relay-vs-direct 对照(GPD/CoDMD/FastWan 都没做);且不是全负:接力臂有可测的运动幅值效应(4/4 seed),审计把它归因到 relay 初始化 × GAN 交互。出处:`thesis_ch2_draft.md` §2.1、发现 5。
2. **"和 CoDMD 84.46 差多少?"**——协议不可比(我们 12/16 维、5 样本 flickering、无 GRiT 维),只作文献坐标,不做 SOTA 对比(T3 红线 8)。出处:`2026-07-22-vb946-scoring-launch.md` 协议脚注。
3. **"GAN 关了质量更高,为什么不直接建议关 GAN?"**——单权重点(0.03)、单血统(relay)、且关 GAN 后运动幅值衰减回 teacher 级——按轴取舍,不是免费午餐;我们给的是归因不是配方建议。出处:`2026-07-25-e2b-fulltable-ch3-threearm.md`。
4. **"为什么没有人评?"**——时间窗内优先受控消融;人评列 limitation 与 future work(计划 T2VHE 式 vs teacher)。出处:`acceptance-log.md` #11。
5. **"E2c 为什么没测?"**——确定性 OOM(32G 节点,R1 第二次判别器前向),非配方失败;校准值/配置留档可在 80G 节点一键复现。出处:`2026-07-25-e2b-fulltable-ch3-threearm.md` E2c 节。
6. **"为什么加冕的 E1a 合成分最低(82.80)?"**——Quality Score 的权重结构使然:dynamic_degree 以 0.5 权重计入,E1a 的低动态特性(vb946 DD 0.5806)拖低合成分,而它赢的一致性/平滑/闪烁维经归一化后区分度小。加冕依据是预注册的 G2 消融协议(q150 六维 + 跨 seed 多样性),QS 为补充参考——这正是"按轴各有所长、单一合成分掩盖轴间取舍"的又一实例。另注:W7 的 QS 84.47 与 CoDMD 文献值 84.46 系数字巧合(7 维 Quality vs 16 维 Total,量纲不同),两数字永不同页并列。出处:`2026-07-26-e2a-vb946-fourth-row.md` QS 节。

## 时间提示

16 页按 20 分钟汇报配:第一幕 3 min / 第二幕 4 min / 第三幕 5 min / 第四幕 5 min / 第五、六幕 3 min。若压缩到 12 页:P3 并入 P2,P5 并入 P4,P9 并入 P8,P13 并入 P12(保四条归因完整)。
