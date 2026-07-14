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

## 待验收队列(计划停点)

| 预计日期 | 节点 | 验收要点 | 谁拍板 |
|---|---|---|---|
| 07-14 ~18:30 | E0 全表(12 行)+ LPIPS 多样性 | **G1 收口**:全表排序 + W7-best 最终口径(结合多样性)+ Ch1 里 iter-质量关系怎么写 | planner |
| 07-15 午后 | E1b 收 + E1a 启动 | E1b 全程健康、ckpt 齐;E1a 发射健康检查 | planner |
| ~07-17 晚 | E1a 收:接力 vs 直蒸同表 | **G2 裁决(论文级)**:两直蒸臂 + W7 同协议同表 | **用户** |
| 07-18~20 | E2a / E2b / E2c 逐臂收 | 每臂:健康 + 一级指标(DD/多样性)入表;E2c 首日校准记录 | planner |
| 07-20 晚 | 数字冻结 + 最终模型 full VBench | 表格完整性;开放关注项全部闭环(含 T3 回填) | planner + 用户 |
