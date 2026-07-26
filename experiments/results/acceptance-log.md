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

| 10 | 2026-07-21 | 执行节点:接续中断会话——E2a 门控 v2 换装(P0)+ vb946 盘点(P1)+ 四项文档回填(P2) | ✅ | ssh 独立抽查:v2 diff 逐行核对 = 仅三处声明改动(10 次重试 → deadline-while 至 07-22 09:00、重试间隔 60s→120s);新门控 PID 4096392 存活、marker 17:52:30 进入 waiting、旧门控(3488767)已消失;e1a 5 set × 946 全齐,w7 s0/s3 在飞(923/873 @18:33)与汇报自洽;vb946 目录无残留软链;GPU 占用者独立定位:GPU3 = qiuzhangxizi 单卡训练 `train_multicls_ref_adapter_v2`(16:15 起,与 13:06 打挂 w7-s0 的 27.9G 足迹同源),GPU0/2/4/5 = songrun 的 sglang 常驻服务(TP2×2,已跑 26h),GPU7 已释放;本地抽查:README 里程碑 / canonical 当前状态 / Ch2 合并(发现 1 n=3 版 + 发现 5)/ 新注记 四项均落盘 | ⏳ rename-watcher(PID 1948635)pgrep 显示命令串疑无换行分隔,队列退出(~18:50)后必须核 `reports/vb946/RENAME_DONE.txt` + rename_e1a/w7.log,失败则手动重跑 `vb946_rename.py`(幂等);⏳ E2a first-health:4/rank 显存 30/32G 边缘,OOM 即上报,**禁止擅改 batch/配方**;⏳ GPU3 协调 = 用户线下动作(对象 qiuzhangxizi);排期已裁(用户 07-21):E2a 优先,vb946 打分让路,full-VBench 数字 deadline 07-26 晚 | `experiments/results/2026-07-21-vb946-launch-gpu3-e2a-gate.md` |

| 11 | 2026-07-25 | **批验收:07-22~07-25 执行五日**(vb946 三学生 12 维打分、E2a/E2b 三臂审计、E5 探针、E2c 止损、flow 四 seed、双臂冠军 n=3) | ✅ | ssh 独立抽查:**E2a/E2B run 目录 config.yaml 逐值核对**——E2a 仅 `gan_loss_weight_gen 0.03→0.0`、E2b 仅 `gan_use_same_t_noise True→False`(GAN 权重保持 0.03),其余(lr 5e-6/batch 16/2500 iter/relay init)与 W7 全同,单变量纪律成立;E2a ckpt **无** discriminator 分片(GAN=0 自洽)、E2b 有;`vb946/scores` = {e1a, w7, e1b} 三学生;`e5/probe_v1.json` + `e2c_r1_calib.json` 落盘;LAUNCHES:E2c 六次重试(11:22-11:55)与 OOM 排查吻合、E2b 六次启动(07-23 抢卡竞态)留痕;8 卡全空、磁盘 924G;本地 7 份注记 + commit 46b8c8b | ⚠️ 07-22~25 期间 planner 不在环,本行为补充批验收;E2c 止损与 empty_cache 临时补丁的"用户批准"为执行侧转述,未独立见证;⏳ **Ch2「flow 继承自中间学生」与 Ch3「flow 由 GAN 分支驱动」须统一机制口径**(并安放 E1a = GAN-on 直蒸却低 flow 的事实,效应 scope 限定 relay 血统);⏳ E2a@2000(n=3 aes 0.613/img 0.723)为全场质量最优档但不在 vb946 表——是否补为第 4 模型待用户裁;红线:E2a aes 0.613 > teacher 0.590 **禁写"超越 teacher"**(偏置口径);`exp/configs/e2b_relay_indep_t_eps.py` 系未启用旧版,建议标废弃 | 终态 `2026-07-25-e2b-fulltable-ch3-threearm.md` + 07-22/23/24 三份注记 |

| 12 | 2026-07-26 | **初稿验收:Ch1 / Ch3 / storyline 三件(写作 agent 首任务,commit 2214994)** | ✅ | 三文件通读 + 逐数字抽查:E2a/E2b 五档表与 07-24/25 注记逐格一致;W1/W4/W5/W7/teacher 锚点与远端 G2 全表一致(planner 07-22 直读存照);E5 采用修正口径(「0.88-0.92/L7 略优」饱和假象弃用声明在位、teachergen 诚实对照、选层「合理但非唯一」);E2c 注记 / vb946 第四行 / n=3「带缘轻微接触」表述与源逐句一致;红线 8 条逐页抽查零违例;storyline 16 页页页有表格背书,备问卡指针真实 | **四项冲突裁决(07-27 冻结时执行)**:① canonical「带不重叠」句改 07-25 定稿口径(方向一致、幅度略超带宽);② Ch2 发现 5.4 + 效度按 07-23 §4 补 teacher 4-seed(W7>teacher 4/4 硬结论 +64% 均值配对、E1a 弱化「不高于 teacher(3/4)」−12%、删两处「teacher 侧仍单 seed」);③ Data-Forcing 采划界口径,canonical 已废计划节不动;④ 多样性统一句改 0.59–0.64(含审计臂)。**缺口处置**:W1 速度 metrics.csv 远端复核 + dm40 组成与 make_motion_set.py 头部比对(执行 agent,一次 ssh);Quality Score 官方权重 + temporal_flickering 官方样本数取源核算(执行 agent,引官方 repo 出处);CD-FVD「计划项未执行」声明入 canonical;E1a aes 全 sweep 轨迹回填**可选**(planner 07-22 已直读核实:@500 0.5243/@1000 0.5665/@1500 0.5523/@2000 0.5609/@2500 0.5385/@3000 0.5219/@3500 0.5327/@4000 0.5281/@4500 0.5226/@5000 0.5196,源=远端 E0_quant.md;只可作「GAN-on 直蒸臂同样 @1000 见顶后总体下行(非严格单调)」的 pattern 句,不可作归因证据——直蒸下滑亦可含过训成分,隔离变量证据只有 E2a) | 三交付文件 + 本行 |

| 13 | 2026-07-26 | **数字冻结(G-freeze)** | ✅ 冻结生效 | Quality Score 四值独立重算:逐维 (raw−Min)/(Max−Min) 手工复核,四模型加权和 5.38221/5.43505/5.49026/5.55718 ÷ 6.5 → 82.80/83.62/84.47/85.50,与执行所报分毫不差(权重出处 Vchitect/VBench `scripts/constant.py` + `cal_final_score.py`,URL 在注记);A1 metrics.csv 复核审毕(teacher 165.24 精确一致,student 全 sweep 6.591–6.656);A2 dm40 组成零修改通过;B4 flickering 官方原文坐实(25 样本 + static filter) | **A1 裁定:范围改 6.59–6.66(全 sweep 实读),≈25× 标题不变——planner 亲落全部 9 处触点**;84.47 vs CoDMD 84.46 数字巧合:禁同页并列,storyline 增备问卡 #6(加冕臂 QS 最低 = DD 权重结构,加冕依据是预注册 G2 协议不受影响);**冻结时点 = 本行 commit,此后对外数字不再变更,变更须开新验收行重裁** | `2026-07-26-freeze-verification.md`(planner 裁定附言) |

## 待验收队列(计划停点)

| 预计日期 | 节点 | 验收要点 | 谁拍板 |
|---|---|---|---|
| ✔ 07-14 | E0 全表 + LPIPS(已收口,#7) | G1 完成:best-of-sweep 定调、Ch2 outcome-agnostic | planner |
| ✔ 07-20 | E1a/E1b 完训节点(已验收,#8) | 两臂产物齐、loss 同带、重启事故记录在案 | planner |
| 07-20 下午 | **G2:32 行全表 + 两臂 best-of-sweep + dm40/多样性** | 接力 vs 直蒸裁决(论文级);同时定"最终加冕模型" | **用户** |
| 07-20 晚 | E2 处置决定(甲:今晚发缩短 E2b / 乙:Ch3 转观察性证据) | 取决于答辩确切时间(待用户回答) | **用户** |
| 07-20 晚 → 07-21 早 | full VBench(G2 加冕模型,唯一一次) | 与 CoDMD 84.46 同表的对外数字;顺利则补第二名 | planner |
| 07-21 | 数字冻结 + 全部开放关注项闭环;(若甲)E2b 收 + 对照数字入 slides/附录 | 表格完整性;E1a 重启条目核对 | planner + 用户 |

> 队列更新(2026-07-21,#10 验收时):07-20 下午 G2 已由用户裁决(#9);07-20 晚 E2 处置在新时间表(07-28/07-31)下改为 Plan A 并已获批(E2a GAN=0 四卡门控自启);full VBench 生成 in-flight、打分让路 E2a(用户 07-21 裁)。新停点:07-22 早 E2a 起训健康验收(若 09:00 门控超时 → 用户协调 GPU3);~07-24 E2a 完训验收 + vb946 打分上卡(16 维 × 2 模型,协议差异逐维脚注),数字硬 deadline 07-26 晚;07-26/27 数字冻结,E2b/E2c 缩短版与 E5 视剩余窗口取舍。

> 队列更新(2026-07-25,#11 批验收时):实验期收口,转写作冲刺。新停点:07-25 晚 用户裁「vb946 是否补 E2a@2000 第 4 模型」(裁定后归还 8 卡并向共享方致谢);07-26 Ch3 初稿验收(三臂表 + E5 机制段 + E2c 限制注记 + Ch2/Ch3 flow 口径统一,planner);07-27 晚 全稿 + 数字冻结(planner + 用户);07-28 汇报。可选项(07-28 后):GRiT 四维 detectron2、E2c 80G 复现、W5 扩表。

> 队列更新(2026-07-26,#12 验收时):初稿三件通过。冻结日程:07-27 白天 = 五处冻结修订(canonical ①④ + CD-FVD 声明、Ch2 ②,写作 agent)+ 远端两核(metrics.csv、dm40 组成,执行 agent)+ Quality Score/flickering 官方取源核算(执行 agent);07-27 晚 = planner 终审 + 数字冻结(全稿数字不再变更);07-28 汇报(slides 由 storyline 直出,逐数字对源复核,重心页 P12/P14)。

> 队列更新(2026-07-26 晚,#13 冻结生效):冻结提前一天完成(原计划 07-27 晚)。余程:07-27 = slides 制作(由 storyline 直出,逐数字对源;重心页 P12/P14;含备问卡 6 张)+ 对冻结数字排练;07-28 = 汇报。07-28 后再议:GRiT 四维 → Semantic/Total、E2c 80G 复现、人评、直蒸血统 GAN=0 臂、W5 扩表、磁盘清理。8 卡若尚未归还,即刻归还并向 songrun / qiuzhangxizi 致谢。
