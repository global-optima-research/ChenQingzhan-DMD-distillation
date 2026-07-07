# Wan2.1-T2V-1.3B DMD2 渐进式蒸馏实验记录

Last updated: 2026-07-07

本文件是当前主线(WanT2V step-count relay `50 -> 8 -> 4`)的**唯一正式实验记录**:只做目录、关键数据与结论。逐 run 详表进 `experiments/results/`,远端 artifact 为准。历史线(Wan2.2 TI2V 5B / WanI2V,2026-04/05)见 `03-dmd-distillation/HANDOFF.md`(已冻结)。研究侧证据分级与 novelty 红线见 `research/T0_project_analysis.md` 与 T1/T2 报告。

## 索引

| ID | 状态 | 实验(远端 run 目录名后缀) | 关键结论 | 详表 |
|---|---|---|---|---|
| W1 | 已完成 | 4-step 基线 `..._global_8`(LR 1.25e-6 / batch 8 / 6000 iter) | 训练闭环通;肉眼最佳 ckpt `0001000`;10-prompt 速度 sweep:teacher 165.24s vs student 6.59-6.63s ≈ 25x(内部记录,引用前重读 metrics.csv) | 远端 `reports/eval_10prompts/` |
| W2 | 已完成(artifact 不可复核) | 8-step 首训 `..._step8`(7 卡续训至 2530) | `0002500` 平均 13.16s、明显模糊差于 W1 `0001000`——仅本地 2026-06-17 索引记载,**远端目录已消失,不得作为已证明证据** | `reports/2026-06-17-wan-dmd2-openvid-progress.md` |
| W3 | 已跑未评 | 8-step 消融 `..._step8_freq`(`student_update_freq=2`,LR 1.25e-6 / batch 10 / 1000 iter) | 推理样例在远端,**结论从未整理**;是"每锚点有效更新量"轴的现成数据点 | 待 P0 量化 |
| W4 | 已跑未评 | 8-step 消融 `..._step8_normalize`(均匀 t_list,LR 1.25e-6 / batch 10 / 1500 iter) | 同上;是 t_list 形状矩阵(P2)的现成"均匀档" | 待 P0 量化 |
| W5 | 已完成 | 8-step `..._step8_lr_original`(LR 1e-5 / batch 12 / 2500 iter) | 肉眼"进入可用";`0002500` 被选为 relay 初始化源 | 待 P0 量化 |
| W6 | 已完成 | 4-from-8 第一轮 `..._step4_from_step8_2p5k`(LR 1e-5 / batch 12 / 6 卡) | 肉眼:早期 `0000500` 最好、后期物理规则崩坏 | 待 P0 量化 |
| W7 | 已完成 | 4-from-8 第二轮 `..._step4_from_step8_8node`(LR 5e-6 / batch 16 / 8 卡) | 肉眼:可用且 `0000500 -> 0002500` 随 iter 提升;**与 W6 之间 LR/batch/GPU 三因素同时变化,改善归因不成立** | eval-10 样例已在远端;待 P0 量化 |

旁线(未评估,论文 baseline 候选):`wan_fdistill`(2026-06-09~12)、`wan_mf`(2026-06-12);ladd/causvid 等 smoke config 存在未跑全。

## 当前状态(每阶段更新)

- 阶段(2026-07-07):训练全部停止(8 卡空闲自 2026-06-25);研究侧 T0-T2 已验收,T3(novelty 终裁 + Wan 生态竞品)已分发。
- 最大缺口:**全部质量结论为肉眼判断,零量化指标**;W6→W7 归因混淆未解。
- 当前最佳(肉眼口径):W7 的 `0002500`(4-step,relay 自 W5 `0002500`)。
- 竞品坐标:CoDMD(Wan 官方团队,concurrent,同基座同步数)VBench Total 84.46——我们 4-step 结果量化后需落在 83-85 区间才有竞争力。
- 下一步 = P0(现有 checkpoint 量化评估),见「下一阶段主线」。

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

- 训练健康指标(loss 曲线)≠ 收益指标;收益一律落在量化协议上:日常消融 = VBench 6 质量维 + CD-FVD(禁用 I3D-FVD)+ 跨 seed 多样性(LPIPS/DINO);主表 = full VBench(Total/Quality/Semantic)+ T2VHE 式人评 vs teacher 50-step;**Dynamic Degree 与多样性必须主动报**(TMD 证明 VBench 总分测不出 mode collapse;Phased DMD 证明 DMD2 动态性坍缩是公开攻击线)。
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
  student 4-step 无 CFG 平均 6.59-6.63s ≈ 25x —— 内部记录,引用前重读远端 metrics.csv
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
