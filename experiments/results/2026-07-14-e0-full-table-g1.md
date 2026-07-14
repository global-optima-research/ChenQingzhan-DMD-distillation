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

## 状态与下一步
- E1b iter ~2740/5000,ETA 07-15 ~13:45 不变。
- E0 任务(A/B)交付完成:全 12 行表 + G1 裁决。
- 待用户 G1 定调(Ch1/Ch2 写作口径);canonical report 的 W 行结论改写等该定调,暂不动。
- 下一自动节点:E1b 收 → eval sweep + E0 协议打分 → 启 E1a;下一用户停点 ~07-17 G2。
