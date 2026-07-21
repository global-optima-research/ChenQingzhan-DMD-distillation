# 验收进度表(planner 台账)

规则:每次验收(研究报告 / 执行节点 / 数字冻结)后由 planner 当天更新本表。结论三档:✅ 通过 / ⚠️ 有条件通过(附条件)/ ❌ 退回。"独立抽查"列必填——验收不能只凭汇报,至少一项独立核实。未闭环关注项标 ⏳,闭环后改 ✔ 并注日期。

## 已验收

| # | 日期 | 节点 | 结论 | 独立抽查(planner 亲自做的) | 关注项 / 后续动作 | 记录指针 |
|---|---|---|---|---|---|---|
| 1 | 2026-07-06 | T1 文献调研报告 | ✅ | 4 个载重条目联网核实(CoDMD/GPD/Phased DMD/rCM 摘要逐项吻合);checklist 六项全过 | 载重数字复核清单转入 T3 ✔(07-07) | `research/T1_video_fewstep_distillation_landscape.md` |
| 2 | 2026-07-06 | T2 组件调研报告 | ⚠️ 有条件通过 | 远端代码核实判别器宿主 = **冻结 teacher**(推翻其 4.1 主张前提);One-Forcing 宿主更正联网确认;R1/same_t_noise 旗标代码级坐实 | 更正回填 T0/T1/T2 ✔(07-06) | `research/T2_dmd2_component_neighbors.md`(4.4 节 planner 注记) |
| 3 | 2026-07-11 | T3 终裁报告(消化 + 抽查) | ✅(spot-verified) | lightx2v 条件项、TMD 占据、上游 FastGen 公开配置三项抽查(上游 config 与我远端所读逐值一致) | ✔ **T3 §6.1 回填更正已执行(2026-07-14)**:T0 修正 gan_use_same_t_noise"主动配置"措辞;T1 加轴 A 作废 + 轴 B 上游压缩注记;T2 加出厂值 + 轴 C + ε 精确化注记;venue/数字批量更正以 T3 §6.1 为权威指针 | `research/T3_novelty_adjudication.md` |
| 4 | 2026-07-13 | 远端归档 + 三区结构上线 | ✅ | `RELAY-INIT-OK` 硬校验;5 个冲刺 config 逐一实例化验证(单变量差异逐项核对);E1a dry-run 全链路 | 归档只移不删(467G → `archive_pre_sprint_20260713/`);真正释放磁盘需删除,待用户拍板 ⏳ | 远端 commit `936bf7c`;本地 `1c42406` |
| 5 | 2026-07-13 晚 | 执行节点 1:E1b 启动 + E0 铺开 | ✅ | ssh 独立抽查:pid 190874 存活、iter 160、loss 0.48-0.56、31.54s/iter、峰值 27.86G、GPU 0-5 满载、e0 队列已产出 | ✔ VBench 冒烟通过(E0 链路已产数);⏳ loss 基线口径待 E1a 同口径对比;⏳ E2c 2000-iter 预案 07-18 晚触发线 | `experiments/results/2026-07-13-sprint-e1b-launch-and-e0-kickoff.md`;远端 `experiment/sprint_e1b_*.md` |
| 6 | 2026-07-14 | G1 首批(E0 7/12 行)+ 执行连续性 | ✅(收口见 #7) | 直接读远端 `experiment/E0_quant.md`(不经转述):teacher DD 0.30 vs 学生 0.61-0.88 反转、W7 aesthetic 0.577→0.538 单调降 均核实;pid 190874 已跑 23h/iter2640、E0 打分队列在跑——均 detached,不依赖执行 agent(回答"Fable 断供"操作问题的依据) | ✔「W7 随 iter 单调提升」措辞今日停用(已回填正式报告);🔄 **DD 作 E2 一级指标存疑**:~40 motion-prompt 清洁子集已批准(~2 GPU·h,E1a 启动后 gap GPU),E2 前就绪 → 闭环;✔ 全表已收口(见 #7) | `experiments/results/2026-07-14-e0-first-batch-g1.md`;远端 `experiment/E0_quant.md` |
| 7 | 2026-07-14 | **G1 收口:E0 全表(12 行)** | ✅ | 直接读远端全表,独立复核两处背离:(a) W7 aesthetic @500→@2500 单调降、质量最优在 @500-@1000 ✓;(b) best-of-sweep W7@500 vs W1@1500 质量接近(W7 领 aesthetic +0.041,W1 领 motion/bg/diversity)✓。**追加发现(执行漏报)**:relay 血统 W5/W7 diversity 0.59-0.61 系统性低于直蒸 W1 0.62-0.65、teacher 0.73——relay 换 diversity,Ch2 neutral/negative 加码证据。W4 uniform-t_list 坍缩独立复现 TMD,采纳为评估协议有效性证据 | Ch1 = best-of-sweep(肉眼档弃用),"肉眼档不可靠"写成量化前置动机;Ch2 = outcome-agnostic 起草、neutral 为领先假设(传递性:弱直蒸 W1 已追平 relay → 强直蒸 E1a 应 ≥ relay)、G2 由 W7-vs-E1a 定稿;学生高 consistency 须联读 DD/diversity(勿写"胜过 teacher") | `experiments/results/2026-07-14-e0-full-table-g1.md`;远端 `experiment/E0_quant.md` |

| 8 | 2026-07-20 | 执行节点:E1a/E1b 完训 + 两臂打分铺开 | ✅(表待收) | ssh 独立核查:两臂各 10 档 ckpt 齐;37 个打分进程在跑;`LAUNCHES.log` 全量核对——**E1a 07-15 17:46 首启、07-16 10:38 重启一次**(+43.7h 与报告完训时间 07-18 06:00 自洽);loss 对照两臂同带(E1a 0.21-0.45 / E1b 0.17-0.50)无发散 ✔(闭环 #5 loss 口径项) | ⏳ E1a 重启事故须在 E1a 记录中有独立条目(验收时核对);**⏳ 关键日历事实:E2a/b/c 与 E5 均未启动(LAUNCHES 零记录、无 run 目录、无 probe 产物)——今日必须处置**:方案甲 = 今晚发缩短版 E2b(1500-2000 iter,G1 已证峰值在 @500-1000,缩短有据),方案乙 = Ch3 转观察性证据 + 预注册协议;E5 两案均出论文转 future work;full VBench 今晚对 G2 加冕模型自动排 | 远端 `experiment/LAUNCHES.log`、`experiment/E0_quant.md`(32 行扩展中) |

| 9 | 2026-07-20 | **G2 定稿表(32 行零缺格)** | ✅(裁决案已呈用户) | 直读远端 `G2_table.txt`/`E0_quant.md` 逐格核对:(a) imaging 优势**强于执行所报**——E1a 连续 5 档(@1000-@3000 中五档 ≥0.7065)全部高于 relay 全 sweep 最佳 0.6971,峰值 0.7235@2000,非单点 0.020;(b) 多样性 0.628-0.635 vs relay 0.598-0.613 ✓(8-seed,最稳轴);(c) DD_clean 全臂 ≥ teacher 0.625 无坍缩 ✓,**但执行淡化了 E1a 的 DD_clean 0.75-0.78 明显低于 W7/E1b 的 0.95-1.0**——不翻裁决(E1b 直蒸臂以 0.975 DD + 更高多样性同样压 relay),但 Ch2 必须如实写;(d) aesthetic 判平 ✓(W7@500 0.5768 vs E1a@1000 0.5665,差 0.010 噪声带) | 两条写作红线随裁决下发:E1a 低动态注记;imaging 超 teacher(0.7235 vs 0.6918)大概率锐度偏置,与 consistency 静态偏置同类,**禁写"学生质量超越 teacher"**;②换 seed 重测转非阻塞附录项(GPU 1/3 即可跑,不等裁决) | `experiments/results/2026-07-20-g2-relay-vs-direct-final.md`;远端 `G2_table.txt` |

## 待验收队列(计划停点)

| 预计日期 | 节点 | 验收要点 | 谁拍板 |
|---|---|---|---|
| ✔ 07-14 | E0 全表 + LPIPS(已收口,#7) | G1 完成:best-of-sweep 定调、Ch2 outcome-agnostic | planner |
| ✔ 07-20 | E1a/E1b 完训节点(已验收,#8) | 两臂产物齐、loss 同带、重启事故记录在案 | planner |
| 07-20 下午 | **G2:32 行全表 + 两臂 best-of-sweep + dm40/多样性** | 接力 vs 直蒸裁决(论文级);同时定"最终加冕模型" | **用户** |
| 07-20 晚 | E2 处置决定(甲:今晚发缩短 E2b / 乙:Ch3 转观察性证据) | 取决于答辩确切时间(待用户回答) | **用户** |
| 07-20 晚 → 07-21 早 | full VBench(G2 加冕模型,唯一一次) | 与 CoDMD 84.46 同表的对外数字;顺利则补第二名 | planner |
| 07-21 | 数字冻结 + 全部开放关注项闭环;(若甲)E2b 收 + 对照数字入 slides/附录 | 表格完整性;E1a 重启条目核对 | planner + 用户 |
