# Wan2.1-T2V-1.3B DMD2 渐进式蒸馏实验记录

Last updated: 2026-07-20

本文件是当前主线(WanT2V step-count relay `50 -> 8 -> 4`)的**唯一正式实验记录**:只做目录、关键数据与结论。逐 run 详表进 `experiments/results/`,远端 artifact 为准。历史线(Wan2.2 TI2V 5B / WanI2V,2026-04/05)见 `03-dmd-distillation/HANDOFF.md`(已冻结)。研究侧证据分级与 novelty 红线见 `research/T0_project_analysis.md` 与 T1/T2 报告。

## 索引

| ID | 状态 | 实验(远端 run 目录名后缀) | 关键结论 | 详表 |
|---|---|---|---|---|
| W1 | 已完成 | 4-step 基线 `..._global_8`(LR 1.25e-6 / batch 8 / 6000 iter) | 训练闭环通;肉眼最佳 ckpt `0001000`;10-prompt 速度 sweep:teacher 165.24s vs student 6.59-6.63s ≈ 25x(**已复核 2026-07-26,直读远端 metrics.csv**:teacher 165.24 精确一致;student 全 sweep 实读 6.591-6.656、speedup 24.83-25.07x——上限 6.63→6.656 有 0.026s 出入,是否改写为 6.59-6.66 待 planner 裁) | 远端 `wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/metrics.csv` |
| W2 | 已完成(artifact 不可复核) | 8-step 首训 `..._step8`(7 卡续训至 2530) | `0002500` 平均 13.16s、明显模糊差于 W1 `0001000`——仅本地 2026-06-17 索引记载,**远端目录已消失,不得作为已证明证据** | `reports/2026-06-17-wan-dmd2-openvid-progress.md` |
| W3 | 已量化 | 8-step 消融 `..._step8_freq`(`student_update_freq=2`,LR 1.25e-6 / batch 10 / 1000 iter) | 近静态+噪声型退化(DD_q150 0.113 / imaging 0.395);短训低 LR 预期内,仅作轴 C 观察点 | 2026-07-14 E0 注记 |
| W4 | 已量化 | 8-step 消融 `..._step8_normalize`(均匀 t_list,LR 1.25e-6 / batch 10 / 1500 iter) | **教科书级 mode collapse**:consistency 全表最高(0.975/0.979)但 imaging 0.256/div 0.462 全表最低——在本基座独立复现 TMD"均匀 t_list 致坍缩",是评估协议有效性的正面证据 | 同上 |
| W5 | 已量化 | 8-step `..._step8_lr_original`(LR 1e-5 / batch 12 / 2500 iter) | `0002500` = relay 初始化源;量化:aes 0.559/img 0.657/div 0.595,8 步档部署备选 | 同上 |
| W6 | 已完成 | 4-from-8 第一轮 `..._step4_from_step8_2p5k`(LR 1e-5 / batch 12 / 6 卡) | 肉眼:早期 `0000500` 最好、后期物理规则崩坏 | 待 P0 量化 |
| W7 | 已完成+已量化 | 4-from-8 第二轮 `..._step4_from_step8_8node`(LR 5e-6 / batch 16 / 8 卡) | 量化推翻肉眼记录:aesthetic 随 iter **单调下降**,best-of-sweep 在 @500-@1000(非肉眼 @2500);与 W6 三因素混淆仍在 | E0 全量化,见 2026-07-14 注记 |
| E1a | 已完成+已量化 | 直蒸对照 A 臂 `sprint_e1a_direct50to4_lr5e6_b16`(= W7 配方,50→4,5000 iter,teacher 起训) | **G2 冠军档 @1000:imaging 0.717(超 relay 与 teacher)、div 0.635(超 relay 两档)**;训练一次 iter-500 保存 hang(近满盘瞬态),重启后完整跑完 | 2026-07-20 G2 注记 |
| E1b | 已完成+已量化 | 直蒸对照 B 臂 `sprint_e1b_direct50to4_lr1e5_b12`(= W5 阶段配方,50→4,5000 iter) | 冠军档 @500:img 0.695/div 0.628;aes 偏低(0.532)显示激进 LR 上限——bracket 有效 | 同上 |

旁线(未评估,论文 baseline 候选):`wan_fdistill`(2026-06-09~12)、`wan_mf`(2026-06-12);ladd/causvid 等 smoke config 存在未跑全。

## 当前状态(每阶段更新)

- 阶段(2026-07-21):E0 量化 + E1a/E1b 直蒸对照完成,**G2 已裁决**(用户 2026-07-20);n=3 置信带与 RAFT 连续光流回填**已完成并并入 Ch2 草稿**(发现 5:运动幅值为接力唯一实测差异,W7 +61% vs E1a −22% 相对 teacher,mixed finding,总裁定不变);full VBench(946×5)生成 in-flight;**E2a(判别器审计第一臂,Plan A)已获批**,服务器端门控待发。里程碑:汇报 07-28、最终论文 07-31。
- **当前最佳(量化口径,best-of-sweep)**:直蒸 E1a `0001000`(imaging 0.717/div 0.635/DD_clean 0.75/aes 0.567)。relay 代表档 = W7 `0001000`。checkpoint 选择一律 best-of-sweep,肉眼档弃用(G1 裁定)。
- **G2 结论(Ch2 定调)**:匹配配方与预算下,step-count relay 质量不优于直蒸、多样性劣于直蒸(两直蒸臂 div 0.628-0.635 > relay 0.598-0.613,独立同向)——负结果如实报;卖点 = 本基座首个受控 relay-vs-direct 对照(GPD/CoDMD/FastWan 均未做)。
- **退化定性(评估协议贡献)**:少步退化 = 跨 seed 多样性坍缩(teacher 0.732 → 学生 0.59-0.64,含审计臂),**非**动态度坍缩(dm40 干净 DD:学生 0.75-1.0 ≥ teacher 0.625;smooth 0.97+ 排除抖动)。DD 口径:dm40 DD_clean 可引用,q150-DD 降脚注相对读。
- 竞品坐标:CoDMD 84.46 等 full-VBench 数字只作文献坐标,协议差异必须脚注(禁 SOTA 对比,T3 红线 8)。full VBench 两模型 E1a@1000 + W7@1000(946×5)生成 2026-07-21 in-flight:e1a 5/5 seed 齐(各 946),w7 s1/s2/s4 齐、s0/s3 收尾中;flat 目录已产出(各 4720 条,946×5 去重后)。**打分 12/16 维已完成(2026-07-22 17:18)**:缺的 4 维全为 GRiT 系(color/object_class/multiple_objects/spatial_relationship,缺 detectron2,墙内源码编译风险高)——用户裁决**缓议**,先按 12 维推进写作;亮点:dynamic_degree 官方域大分化(w7 0.911 vs e1a 0.581,独立复证发现 5 方向)。详见 `experiments/results/2026-07-22-vb946-scoring-launch.md`。
- 晚间三线结果(2026-07-23 晨全部落定):① e1a@2000 div = 0.6225(@1000 0.635→@2000 0.622,仍高于 relay 两档);② **flow 多 seed 定稿:W7>E1a 在 4/4 seed 全同向(中位均值 3.36 vs 1.81,≈1.9×),发现 5 升级为多 seed 稳健**,但 seed 方差大(单 seed 百分比弃用,Ch2 已改写);③ **E1b@500 full-VBench 12 维齐**——三臂主表落成:E1a 赢一致性/平滑/闪烁类,W7 赢动态与语义动作类,E1b 居中(见 `experiments/results/2026-07-23-flow-multiseed-e1b946-e2a-eval.md`)。
- **2026-07-25 晨:Ch3 三臂对照定稿**(E2a=GAN0 / W7=配对 GAN / E2b=独立 (t,ε) GAN,全部 5 档 sweep + 六维 + 多样性 + flow):①运动幅值由 GAN 分支驱动且与配对约定无关(E2b flow 2.40→4.71 单调爬升,E2a 滞留 teacher 级);②质量维与 GAN 反向,"早峰后滑"获候选机制归因;③ **Claim D 裁定:配对 vs 独立不敏感**(同档几乎无差,首个受控消融即审计贡献);④多样性对判别器不敏感(坍缩归蒸馏本身)。E2a 冠军档 n=3 定稿(aes 0.613/imaging 0.723;与 GAN-on 臂逐 seed 配对 3/3 同向,均值带带缘轻微接触——方向一致、幅度略超带宽)。**E2c(R1 臂)确证受硬件限制取消**(4/8 卡形态均确定性 OOM 于 R1 额外前向,补丁无效;校准值 0.75 与配置留档供 80G 节点复现,审计矩阵该维标"未测");E2b 冠军档 @500 q150 n=3 补漏中。详见 `experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md`。
- 2026-07-23 日间落定:**E2a 训练完整成功**(12:12,2500/2500,5 档 checkpoint 齐;stall marker 为完成后误报,监护 pgrep 教训已入册);eval burst 自动跑(dm40+flow 支线已完,q150/d40 支线 ~18:30 前齐);**teacher 4-seed flow 定稿**(W7>teacher 4/4 硬结论;E1a<teacher 仅 3/4,弱化为"不高于");**E5 探针出首版**(`exp/eval/e5_probe.py` → `reports/e5/probe_v1.json`:全层 AUC 0.88-0.92、head 层 {15,22,29} 合理非最优、t 依赖极强(低 t AUC=1.0 / 高 t ≈随机)、teachergen 对照示可分性主要为生成域共性——非 headline 贡献);**E2b 已于 16:18 起训**(`gan_use_same_t_noise=False`,GAN 0.03,卡组 2/3/5/7;此前 GPU0 三次抢卡竞态,容错门控切卡后首攻即中)——first-health 过:iter1 正常,显存 31.1G/32.6G(95%,GAN 分支 +3.8G,边缘站住),稳态估 55-60s/iter,完成约 07-25 早间。GRiT 4 维缓议。详见 `experiments/results/2026-07-23-flow-multiseed-e1b946-e2a-eval.md`。
- 进行中(2026-07-22):**E2a relay-stage GAN=0 已于 04:47 自动起训**(门控 v3 精检 125 轮后 gpu3 释放;4 卡 1/3/6/7,NCCL fail-fast + stall-guard 2600s 在岗)。first-health 通过:iter1 正常完成,显存 26.8G/32.6G(~82%),首步 ~52.7 s/iter → 500 iter ≈ 7.3h/档。详见 `experiments/results/2026-07-22-e2a-launch.md`。待做:E2 其余臂、E5 探针、full-VBench 打分(vb946 flat 目录已就绪)。
- 节点与他人共享(2026-07-21 17:48 快照:对方占 GPU 0/2/3/4/5/7,我方 1/6 在跑 vb946);磁盘 /data 95%,余 1.2T(<90% 不可达:他人占 ~18T)。

## 通用路径

```text
远端仓库:      ust_ip:/data/chenqingzhan/FastGen(branch main,commit e66f6c6 @2026-07-06)
输出根(现行): /data/chenqingzhan/FastGen/FASTGEN_OUTPUT/fastgen/wan_dmd2/
输出根(停用): /data/chenqingzhan/fastgen_output(2026-06-06 后不再写入)
run 目录:      <输出根>/wan21_t2v_dmd2_OpenVid_global_8[_step8_*|_step4_from_step8_*]/
                 每个 run 含 config.yaml(实际超参 ground truth)/ checkpoints / inference / logs
teacher 模型:  FASTGEN_OUTPUT/MODEL/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
数据:          WDS:/data/datasets/OpenVid-1M/webdataset
eval prompts:  scripts/inference/prompts/wan21_dmd2_openvid_eval_prompts.txt(10 条)+ negative_prompt.txt
远端入口脚本:  fastgen/configs/experiments/WanT2V/run_train_dmd2_step4_from_step8_2p5k.sh(训练,env 参数化)
               fastgen/configs/experiments/WanT2V/run_infer_dmd2_step4_from_step8_8node_eval10.sh(eval-10 sweep)
               fastgen/configs/experiments/WanT2V/run_infer_dmd2_step8_freq_eval5.sh(已泛化的 eval-N)
本地一行提交:  bash experiments/bin/check_remote.sh experiments/configs/wan21_check.env
               bash experiments/bin/run_remote_script.sh [--dry-run] experiments/configs/<config>.env
```

## 记录口径

- 训练健康指标(loss 曲线)≠ 收益指标;收益一律落在量化协议上:日常消融 = VBench 6 质量维 + CD-FVD(禁用 I3D-FVD)(CD-FVD 为计划项,实际未执行;协议以实际执行口径为准,见 Ch1 §1.8)+ 跨 seed 多样性(LPIPS/DINO);主表 = full VBench(Total/Quality/Semantic)+ T2VHE 式人评 vs teacher 50-step;**Dynamic Degree 与多样性必须主动报**(TMD 证明 VBench 总分测不出 mode collapse;Phased DMD 证明 DMD2 动态性坍缩是公开攻击线)。
- **每次实验只改一个主要因素**。W6→W7 已违例(LR/batch/GPU 同时变),这是 P1a 存在的原因;此后任何配方改动必须可归因。
- 每个 run 的超参以其输出目录 `config.yaml` 为准,不以报告转述为准;引用未重读的数字一律标注(如 W1 速度数字)。
- checkpoint 选择用同一 prompt 集 sweep + 早停,不默认最后一个;每 500 iter 存档。
- 方法措辞:step-count relay / progressive step reduction;**禁用 phased/progressive DMD**(命名已被 arXiv 2510.27684 占用);判别器表述 = 冻结 teacher backbone 15/22/29 层特征 + 可训 multiscale MLP 头(2026-07-06 代码核实)。
- 一个实验 = 一个 `experiments/configs/*.env` + 一个远端 run 目录 + 一条 `experiments/results/YYYY-MM-DD-*.md` 结果注记。

## W1:4-step 基线(`global_8`)

结论先行:训练/推理/checkpoint 闭环全通,速度收益确立;质量只有肉眼结论(`0001000` 最佳),`0000500` checkpoint 缺失。

```text
超参(config.yaml 已核):LR 1.25e-6(net/disc/fake 相同)/ batch_size_global 8 / max_iter 6000 /
  student_sample_steps 4 / t_list [0.999, 0.937, 0.833, 0.624, 0.0] / guidance_scale 5.0 /
  student_update_freq 5(=每 5 iter 一次 student 更新)
速度(2026-06-15,10 prompts):teacher 50-step(CFG 双前向 ≈100 NFE)平均 165.24s;
  student 4-step 无 CFG 平均 6.59-6.63s ≈ 25x —— 已复核 2026-07-26(路径:`wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/metrics.csv`;teacher 精确一致,student 全 sweep 6.591-6.656,范围上限勘误待 planner 裁)
```

## W2:8-step 首训(artifact 已不可复核)

结论先行:本地 2026-06-17 索引记载其明显模糊(13.16s/条,约 2 倍于 4-step 时长),差于 W1 `0001000`;远端 run 目录已消失,**只作背景,不作证据**。当时诊断:高噪密集插值 t_list + `student_update_freq=5` 下每锚点有效监督不足 + 7 卡续训扰动。

## W3 / W4:两个 8-step 消融(freq / normalize)——已跑未评

结论先行:这两个 run 是免费的现成数据点(W3 = 更新频率 2,W4 = 均匀 t_list),当时因肉眼流程中断从未出结论;P0 量化时必须纳入,直接服务轴 A(schedule)与轴 C(每锚点更新量)。

## W5:8-step lr_original——relay 初始化源

结论先行:恢复 LR 1e-5 + batch 12 后,肉眼判定 8-step "进入可用";`0002500`(≈500 次 student 更新)被选为两轮 4-from-8 的共同初始化。注意:与 W2/W3/W4 相比同时改了 LR 与 batch,8-step 内部同样存在归因混淆,只是不影响"它作为 relay 源足够好"这一工程判断。

```text
超参(已核):LR 1e-5 / batch 12 / 2500 iter / t_list 高噪密集插值 9 锚点 / update_freq 5
```

## W6:4-from-8 第一轮(2p5k)

结论先行:肉眼:`0000500` 最好,`0001000-0002500` 物理规则崩坏(与 SDXL-Lightning 记载的对抗蒸馏后期语义伪影模式相容)。init = W5 `0002500`,仅继承生成器权重(`key_map={"net":"net"}`),优化器/fake score/判别器重置。

```text
超参(已核):LR 1e-5 / batch 12 / 6 卡 / 2500 iter / 4-step t_list 同 W1
```

## W7:4-from-8 第二轮(8node)——当前肉眼最佳

结论先行:肉眼:可用,且 `0000500 -> 0002500` 随 iter 单调改善、物理规则修复。**但与 W6 相比 LR 减半(1e-5→5e-6)、batch 12→16、GPU 6→8 三因素同时变,"batch 提升带来改善"的说法不成立**——P1a 的受控实验是把它变成可发表结论的唯一途径。5 个 checkpoint 的 eval-10 MP4 已在远端(2026-06-25 生成),P0 可直接复用。

```text
超参(已核):LR 5e-6 / batch 16 / 8 卡 / 2500 iter / init = W5 0002500(同 W6)
GAN 侧(已核):gan_loss_weight_gen 0.03 / gan_use_same_t_noise True(同 t 同 ε)/ gan_r1_reg_weight 0.0(R1 未启用)
```

## 下一阶段主线(2026-07-07 定稿;完整裁决与候选项见 `research/experiment_plan.md`)

比较轴与验收标准先定死:一级指标 = VBench 6 质量维(重点 Dynamic Degree)+ CD-FVD + 跨 seed 多样性;每条 run 完成即评;任何"改善"必须给出对照 run 的同协议数字。

1. **P0:存量 checkpoint 量化评估(先行,零训练)**。对象:W1 `0001000`(+邻近 2 档)、W5 `0002500`、W7 全部 5 档、W3/W4 最佳档、teacher 参照。素材:W7 与 W1 已有 eval-10 MP4 可直接吃 VBench 自采视频 6 维;CD-FVD 需每档补采样本(4-step 单条 ≈6.6s,便宜)。交付:本报告索引表全部"待 P0 量化"改为数字 + 一条 `experiments/results/` 注记。**这是所有论文主张的前置。**
2. **P1a:LR × batch 解混淆(2-3 条 run)**。围绕 W7 配方:`(5e-6,16)` 已有 → 补 `(1e-5,16)` 与 `(5e-6,12)`;唯一变量原则;裁决"W6→W7 改善主因"。
3. **P1b:50→4 直蒸对照(1 条 run)**。同预算/数据/t_list,只去掉 8-step 中间阶段;轴 B(relay 必要性)的生死消融,GPD/CoDMD 都没做。
4. **P1c:GAN {0, 0.03} × 近似 R1 {on, off}(4 条短 run,纯配置零代码)**。`gan_r1_reg_weight` 已内置(alpha 0.1 与 Seaweed-APT 视频档同量级);一级指标盯 Dynamic Degree(Data-Forcing 同基座警示:加 GAN 使 DD 0.500→0.375);同时补 One-Forcing 缺失的 GAN on/off 证据。
5. **P2:t_list 形状矩阵(6 档)**。均匀(负对照,W4 现成)/ σ=3 / σ=5(基线)/ σ=8(SGMD 默默在用)/ σ=12(断崖检验)/ 非嵌套 σ=5(relay 独家对照);第二因子 = 训练 t 采样 shift 与推理锚点解耦(TMD 证据)。
6. 执行纪律:每条 run 走 `experiments/bin/run_remote_script.sh` 一行提交(先 check_remote,再 --dry-run);launch 前 GPU 预检 <100 MiB;结果注记当天落 `experiments/results/`。
7. 条件项(等 T3/P0 结果再启):IDA(relay 阶段 fake score 冷启动)、Data-Forcing 式 post-training、判别头时空聚合升级、同 ε vs 独立 ε 消融——清单与依据见 `research/experiment_plan.md` Candidates 节。

---

> 计划更替注记(2026-07-11):本文件「下一阶段主线」一节的 P0-P2 阶梯已被取代——T3 终裁(轴 A 被 TMD 占据、上游 FastGen 公开全部单阶段配方、FastWan/CoDMD 直蒸成立)与答辩时间表(~2026-07-22)确定后,当前唯一有效计划为 `research/experiment_plan.md`(11 天冲刺版:E0 量化 / E1 两臂直蒸对照 / E2 判别器审计 / E5 探针,数字冻结 2026-07-19)。

> 工作区注记(2026-07-13):远端已切换为三区结构——配置区 `FastGen/exp/`(薄启动脚本 + 每实验一个 conf/config)、记录区 `FastGen/experiment/`(INDEX.md 索引 + 逐实验详情)、日志区 `FASTGEN_OUTPUT/` 不变。存量清理:W1 六档全量 ckpt、各完结 run 的优化器/辅助网络分片、W6 全部 ckpt、wan_fdistill/wan_mf/cifar10、旧输出根共 ~467G 移入 `/data/chenqingzhan/archive_pre_sprint_20260713/`(只移不删);E0/接力所需档位保留 `net_model+pth`,接力初始化 `step8_lr_original/0002500` 已校验完好。冲刺五个 config(E1a/E1b/E2a/E2b/E2c)已建好并验证,启动方式:`bash exp/run.sh exp/confs/<X>.conf`。

> E0 量化更正(2026-07-14,回填 G1 首批):正式报告 W7 条目「`0000500 -> 0002500` 随 iter 提升」措辞今日停用。E0 q150(域外 VBench 抽样)量化显示 W7 aesthetic 随 iter 单调下降(0.577→0.538),subject/bg/motion 平缓微降,仅 DD 上升;按这 6 维,W7 量化最优在 @500-@1000,而非肉眼选的 @2500。6 月肉眼观察的「后期物理规则修复」是域内 10-prompt 性质,这 6 维捕捉不到——两者不矛盾,但对外一律用量化口径。另一警示:DD 在本表与画质负相关(W1 最高 DD、最差 aesthetic/imaging),疑测抖动而非动态,DD 作一级指标的有效性须在 E2 前重定。数字见远端 `experiment/E0_quant.md`。

> E0 全表收口(2026-07-14,G1 通过):(1) **checkpoint 选择全文改 best-of-sweep**——本批证明肉眼选档不可靠(W7 肉眼选 @2500,量化质量最优在 @500-@1000;W1 肉眼选 @1000,量化 @1500 在 5/6 质量维反超),"肉眼档不可靠"反过来成为 E0 前置量化的方法动机。(2) best-of-sweep 下 relay(W7@500)与弱配方直蒸(W1@1500)质量接近:W7 领 aesthetic(0.577 vs 0.536),W1 领 motion/bg/diversity——relay 不明显优于弱直蒸,且 relay 血统 W5/W7 diversity(0.59-0.61)系统性低于直蒸 W1(0.62-0.65)与 teacher(0.73),即 relay 换取多样性。"接力必要性"改 outcome-agnostic,由 matched-recipe E1a/E1b vs W7 在 G2 定稿(领先假设:neutral/relay 非必要)。(3) 学生 subj/bg consistency ≈ 或 > teacher 不是质量反超,是少步静态偏置(与 W4 坍缩同向),须与 DD/diversity 联读。(4) W4(uniform t_list)= 教科书 mode collapse(consistency 最高、imaging 0.256/DD 0.187/diversity 0.462 最低),本方基座独立复现 TMD"均匀锚点致 VBench 测不出坍缩",纳入评估协议有效性证据。(5) DD 在 q150 受 still-prompt 混淆(teacher 仅 0.30),仅表内相对读;~40 条 motion-prompt 清洁子集已批准补测(E2 一级指标前置)。数字见远端 `experiment/E0_quant.md`。
