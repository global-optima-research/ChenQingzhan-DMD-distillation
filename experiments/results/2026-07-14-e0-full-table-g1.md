# 2026-07-14 — E0 全表(12 模型)+ G1 校准裁决

来源:远端 `experiment/E0_quant.md`;原始 json `reports/e0/scores/`(每模型 q150 6 维 + d40×8seed LPIPS,全部 paired=40)。
协议:VBench 6 维 custom_input,q150(官方 all_dimension 确定性抽样 150,seed0);多样性 = d40×8seed 平均成对 LPIPS-alex(越高越多样)。

## 全表

| model | subj | bg | motion | DD | aes | imaging | div |
|---|---|---|---|---|---|---|---|
| teacher 50-step | 0.9661 | 0.9642 | **0.9903** | 0.3000 | **0.5899** | 0.6918 | **0.7321** |
| W7 relay 4-step @500 | **0.9732** | 0.9585 | 0.9839 | 0.6133 | 0.5768 | 0.6935 | 0.5984 |
| W7 relay 4-step @1000 | 0.9696 | 0.9552 | 0.9784 | 0.7867 | 0.5592 | **0.6971** | 0.6125 |
| W7 relay 4-step @1500 | 0.9666 | 0.9533 | 0.9763 | 0.8267 | 0.5433 | 0.6832 | 0.6092 |
| W7 relay 4-step @2000 | 0.9690 | 0.9546 | 0.9792 | 0.7533 | 0.5477 | 0.6938 | 0.6033 |
| W7 relay 4-step @2500(肉眼最佳) | 0.9668 | 0.9558 | 0.9786 | 0.7867 | 0.5379 | 0.6896 | 0.6061 |
| W1 direct weak @1000(肉眼最佳) | 0.9588 | 0.9530 | 0.9758 | 0.8800 | 0.5103 | 0.6341 | 0.6492 |
| W1 direct weak @1500 | 0.9695 | 0.9594 | 0.9894 | 0.5600 | 0.5357 | 0.6847 | 0.6326 |
| W1 direct weak @2000 | 0.9580 | 0.9561 | 0.9853 | 0.7400 | 0.5059 | 0.6342 | 0.6245 |
| W5 8-step source @2500 | 0.9705 | 0.9610 | 0.9831 | 0.7533 | 0.5586 | 0.6566 | 0.5949 |
| W3 8-step freq=2 @1000 | 0.9611 | 0.9549 | 0.9900 | 0.1133 | 0.4322 | 0.3948 | 0.7058 |
| W4 8-step uniform-t @1500 | 0.9745 | 0.9791 | 0.9887 | 0.1867 | 0.3181 | 0.2555 | 0.4617 |

## G1 裁决(三条,如实)

**结论:肉眼排序 "teacher ≥ W7 > W1" 只在部分维成立;两处与肉眼记录实质性背离;W7 相对弱直蒸 W1 的优势远小于肉眼所述。→ 触发 Ch1/Ch2 措辞再锚定(计划 G1 预案),但不阻塞 E1a。**

### 背离 1(重要):肉眼选定的"最佳 checkpoint"不可靠,两条线都错
- W7:肉眼选 @2500 且称"随 iter 单调提升"。量化:aesthetic 随 iter **单调下降**(0.577→0.538),imaging/subj/motion 平或微降,只有 DD 上升;质量最优在 **@500-@1000**。
- W1:肉眼选 @1000。量化:@1500 在 5/6 质量维碾压 @1000(subj 0.9695>0.9588、motion 0.9894>0.9758、aes 0.5357>0.5103、imaging 0.6847>0.6341),W1 最优是 **@1500**。
- 含义:全文 checkpoint 选择必须 best-of-sweep,肉眼档一律弃用。这本身验证了 E0 前置量化的必要性,是方法章的正面素材。

### 背离 2(影响 Ch2 定调):W7 relay 不明显强于 W1 弱直蒸
- best-of-sweep 对比:W7@500 aesthetic 0.577 领先;但 W1@1500 在 subj/bg/motion/imaging 追平或反超 W7 各档,diversity 0.633 还略高于 W7(~0.60)。
- 即:在 q150 6 维 + 多样性上,relay 4-step 与**弱配方**直蒸 4-step 质量接近,W7 的边际优势主要是"早期 checkpoint 的 aesthetic"。
- 防线/定调:W1 是弱配方(lr1.25e-6/b8),Ch2 真正对照是 matched-recipe 的 E1a/E1b(在跑)。W1 竟能追平,是**预注册中性结局(G2:relay 在此规模非必要)更可能成立**的早期信号——计划已预留该框架,Ch3(判别器审计 + claim D)独立扛 novelty,故事不塌。

### 排序确成立的部分
- teacher 在 aesthetic(0.590)/motion(0.990)/bg(0.964)/diversity(0.732)为最优,符合"teacher 上界"预期。
- imaging 上 W1 明显掉队(0.634 档),W7/W5 更接近 teacher。
- 注意:学生 subj/bg consistency ≈ 或 > teacher,**不是质量反超**,是少步学生帧间变化小的静态偏置(与 W4 坍缩同向、程度轻),不得表述为"学生一致性胜过 teacher"。

## 消融数据点(axis A/C 观察项,非竞争 run)
- **W4(uniform t_list)= 教科书级 mode collapse**:consistency 最高(subj 0.9745/bg 0.9791)但 imaging 0.256、aesthetic 0.318、DD 0.187、diversity 0.462 全表最低。高一致性 + 低成像/低动态/低多样性 = 坍缩静态。**独立复现 TMD"均匀 t_list 致 VBench 总分测不出的 mode collapse"于我方基座**——是评估协议有效性的正面证据(DD/diversity 抓到了 consistency 掩盖的坍缩)。
- **W3(freq=2,1000iter)**:DD 0.113、imaging 0.395、aesthetic 0.432 极低但 diversity 0.706 高 → 近静态+噪声型退化,短训低 LR 预期内。
- 两者仅作 axis-A(schedule)/axis-C(更新量)观察数据点,不进质量竞争。

## DD 方法学(须记入 Ch3,因 DD 是其一级指标)
q150 的 DD 被 still/frozen prompt 指令混淆(teacher DD 仅 0.30)。当前 DD 只能表内相对读、且须与 imaging/aesthetic/diversity 联读。**干净 DD 需一个 motion 导向 prompt 子集**——建议 E1a 启动后用 gap GPU 补 ~40 条 motion prompt 重采关键档(teacher/W7-best/E1a/E1b/E2 各臂),成本低,不占关键路径。

## 用户 G1 决定(2026-07-14,已执行)
1. **Ch1 口径**:采纳——纯 best-of-sweep 量化叙事,删"单调提升/肉眼档",肉眼不可靠写成量化前置动机。
2. **Ch2 口径**:中性结局**降为领先假设**(non-conclusion)——草稿写"预期匹配预算下 relay 与直蒸质量相当,由 E1a/E1b 检验",不预设结论;G2 定稿。
3. **DD 清洁子集**:批准,已建 `dm40`(md5 `324d75a0`)。

## dm40 清洁 DD 子集(决定 3)
- **抽样准则(进论文评估协议章)**:40 条 = 20 条 VBench 官方 `human_action.txt`(纯人体动作,motion by construction,uniform stride)+ 20 条 `all_dimension.txt` 经 MOTION_CUE 正则(pan/tilt/zoom/locomotion/dynamics 动词)过滤且排除 STATIC_BLOCK(still/frozen/motionless/static/...)后 uniform stride;两源均 VBench 官方,静态混淆按构造(A)+黑名单(B)双重排除。脚本 `exp/eval/make_motion_set.py`,md5 记入 `prompts/MD5SUMS`。
- **复用**:E0 重打分与 E2 三臂共用同一 dm40。
- **重打范围**:teacher + W7 全 sweep(best-of-sweep 未定,全跑)+ W1 @1000/1500/2000 + W5 @2500,1 seed(与 q150 同口径)。
- **DD 层级**:dm40 DD = 可引用 DD(aggregator 新 `DD_clean` 列);q150-DD 降脚注级相对读。
- **状态**:17:46 GPU6(学生)/GPU7(teacher)开跑,scorer 门控待两队列退出;清洁 DD 表 ETA ~20:45,aggregator 自动并入 E0_quant.md。E1b 全程在 0-5 卡不受影响。

## dm40 清洁 DD 结果(2026-07-14 20:29,10/10 零失败)

| model | DD_clean(dm40) | DD(q150,脚注) | motion_smooth | aes | img |
|---|---|---|---|---|---|
| teacher 50-step | 0.625 | 0.300 | 0.9854 | 0.573 | 0.684 |
| W7@500 | 0.825 | 0.613 | 0.9778 | 0.582 | 0.675 |
| W7@1000 | **1.000** | 0.787 | 0.9721 | 0.565 | 0.680 |
| W7@1500 | 0.975 | 0.827 | 0.9705 | 0.548 | 0.668 |
| W7@2000 | 0.975 | 0.753 | 0.9727 | 0.556 | 0.676 |
| W7@2500 | 0.950 | 0.787 | 0.9733 | 0.553 | 0.668 |
| W1@1000 | 0.950 | 0.880 | 0.9709 | 0.505 | 0.607 |
| W1@1500 | 0.825 | 0.560 | 0.9868 | 0.527 | 0.664 |
| W1@2000 | 0.950 | 0.740 | 0.9820 | 0.514 | 0.622 |
| W5@2500 | 0.950 | 0.753 | 0.9776 | 0.561 | 0.623 |

### 结论(承重,须记入 Ch3 + 评估协议章)
1. **无动态度坍缩**:干净 motion prompt 上全部学生 DD 0.825-1.0 > teacher 0.625。文献担心的"DMD2 少步动态坍缩"(Data-Forcing/Phased DMD)在我方基座**评测层面不成立**。注意划界:Data-Forcing 是 Cosmos I2V,我方 Wan T2V,base/task 不同,只能说"未复现",非反驳。
2. **高 DD 是真运动非抖动**:motion_smoothness 全员 0.970-0.987,仅微低于 teacher 0.985 → 排除"过度动态=抖动"的替代解释。W7@1000 DD 1.0 且 smooth 0.972。
3. **本项目的退化是 diversity-specific,非 motion**:合 q150 多样性看——teacher LPIPS-div 0.732 > 学生 0.595-0.649(相对降 ~15-18%),但 dynamics 未降反升。**"少步 DMD2 在 Wan-T2V 上退化 = 跨 seed 多样性坍缩,而非动态度坍缩"** 是干净可辩护的具体主张,正是"compute-light 退化审计"次级贡献要的东西。
4. **relay vs 弱直蒸**:motion 集上 W7 在 aes/img 对 W1 的优势比 q150 更明显(DD 双方均高时,W7 imaging 0.67-0.68 vs W1 0.61-0.66)。仍 = 弱配方,真裁决待 E1a/E1b。
5. **teacher DD 0.625 偏低部分因 dm40 源 B 含 style/steady/slow-motion 变体**;human_action-20 子集更纯,但全员高+平滑已使结论稳健,暂不细分。

### E2 设计提醒(重要,E2 启动前必办)
dm40 DD 对好学生已饱和到 ~1.0。E2 判别器审计(GAN on/off、(t,ε) 配对)若指望 DD 显出 GAN 对动态的影响,**二值 DD 无分辨率(天花板效应)**。→ **E2 必须加连续无天花板的运动量:UniMatch/RAFT 平均光流幅值**(计划已列 UniMatch);DD 只作辅证。E2 前把光流幅值脚本加进 exp/eval。

## best-of-sweep 更新(合 q150+dm40)
- W7 量化最优 = @500(aes/img/smooth 均最优,DD 0.825 已够动态)或 @1000(DD 峰值 1.0,img 次优);**非肉眼 @2500**。Ch1 引用 W7 用 @500 或 @1000,注明 best-of-sweep 依据。
- W1 最优 = @1500(q150)/ 但 motion 集上 @1000/@2000 DD 更高;W1 作弱直蒸第三点,引 @1500 质量最优档。

## 状态与下一步
- E1b iter ~3160/5000,ETA 07-15 ~13:45 不变。
- E0 任务(A/B)交付完成:全 12 行表 + G1 裁决。
- 待用户 G1 定调(Ch1/Ch2 写作口径);canonical report 的 W 行结论改写等该定调,暂不动。
- 下一自动节点:E1b 收 → eval sweep + E0 协议打分 → 启 E1a;下一用户停点 ~07-17 G2。
