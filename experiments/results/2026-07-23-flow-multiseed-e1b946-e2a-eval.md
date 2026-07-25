# 2026-07-23 — flow 多 seed 定稿 / E1b full-VBench 齐 / E2a eval burst 门控

## 1. dm40 flow 多 seed(昨晚 GPU5 队列,18 jobs 0 failed)

RAFT 中位幅值(像素/帧,dm40×40 条):

| seed | E1a@1000 | W7@1000 |
|---|---|---|
| s0(原) | 2.15 | 4.44 |
| s1 | 1.83 | 3.28 |
| s2 | 0.46 | 1.27 |
| s3 | 2.80 | 4.44 |
| **4-seed 均值** | **1.81** | **3.36** |

- **臂间方向 4/4 seed 全同向(W7 > E1a,≈1.9×)——发现 5 升级为多 seed 稳健**;已回填 Ch2(发现 5 第 4 条 + 效度威胁)。
- 新警示:seed 间中位差可达 6 倍(s2 两臂同低)→ 单 seed 绝对百分比不可单独引用;teacher 仍单 seed(2.75),相对差以方向+区间表述。
- teacher 多 seed flow 补测已启动(2026-07-23 09:40,GPU2,用户拍板):dm40 × seed 1/2/3 的 50-step 生成 + flow,队列 `teacher_flow_multiseed.jobs`,预计 ~16:00 出数 → teacher 侧同样升级 4-seed,相对差按逐 seed 配对报。

## 2. E1b@500 full-VBench 12 维完成(09:05,过夜全自动链路)

`done: 22 jobs, 4 failed`(4 失败=GRiT 缺 detectron2,缓议中)。三臂主表(standard mode,946×5):

| 维度 | E1a@1000 | W7@1000 | E1b@500 |
|---|---|---|---|
| subject_consistency | **0.9727** | 0.9693 | 0.9673 |
| background_consistency | **0.9579** | 0.9508 | 0.9416 |
| motion_smoothness | **0.9812** | 0.9727 | 0.9747 |
| dynamic_degree | 0.5806 | **0.9111** | 0.8806 |
| aesthetic_quality | 0.5967 | **0.6087** | 0.5802 |
| imaging_quality | 0.6687 | 0.6687 | 0.6614 |
| temporal_flickering | **0.9894** | 0.9796 | 0.9810 |
| human_action | 0.690 | **0.794** | 0.716 |
| scene | 0.2173 | **0.2922** | 0.2225 |
| appearance_style | 0.1990 | 0.1982 | 0.2003 |
| temporal_style | 0.2214 | **0.2305** | 0.2260 |
| overall_consistency | 0.2240 | **0.2386** | 0.2298 |

- 观察(报告用,不下因果结论):E1a 赢一致性/平滑/闪烁类,W7 赢动态度与语义动作类(human_action/scene 或与动态度联动:动起来才判成功);E1b 居中偏 W7 侧(高动态低美学,与其激进 LR 特征一致)。与 q150 结论"质量平/多样性直蒸优"不冲突——两域两协议不混引(Ch2 红线)。
- E1b flow(dm40 s0,过夜顺带): `e1b_0000500_flow.json` 已在 scores/ 下,聚合时并入。

## 2.5 E2a 训练完成 + stall 误报事件(12:12/12:57)

- **E2a 完整成功**:12:12:18 `Training complete.`,@2500 checkpoint 保存 SUCCESS(net/fake_score × model/optim,16G),四 rank 干净退出;**5 档 sweep(500-2500)全齐**。全程 31.4h,零真实事故。
- **stall guard 误报**:训练结束后日志永久静默 → 12:57:44 guard 达阈值空放(进程已不在)。设计缺陷:guard 不区分"完成静默"与"hang 静默",应在目标进程消失时自动退出。
- **连锁教训(与 07-20 pkill 事故同族,入册)**:eval 门控 pgrep `gan[0]` 括号只防自匹配,防不了 **stall guard argv 里的字面 `...gan0`** → 门控误判训练仍在,晚启动 45 分钟(12:13→12:58)。后续监护进程的 pgrep 模式必须同时排除所有监护者 argv(如限定 `python.*train.py` 或匹配进程名),armed 时逐一验证 pgrep 命中集。
- 影响:仅 eval burst 顺延 45 分钟,无数据损失。

## 3. E2a eval burst 已挂门控(09:21,PID 1646264;12:58:56 实际起跑)

- 等 E2a 训练退出(~12:20)→ 精检 → 三条自包含队列:GPU1 = d40×8×5 档 + LPIPS;GPU6 = q150×5 档 + 6 维打分;GPU7 = dm40×5 档 + flow。GPU3 留 E5。
- 预计 17:00 前 E2a 全 sweep 的 q150 6 维 + DD_clean + 多样性 + flow 齐 → Ch3 第一组数据。
- E5:设计稿已批,今日读 `fastgen/methods/distribution_matching/dmd2.py:250 _compute_real_feat` + `fastgen/networks/discriminators.py` 后编码。

## 4. teacher 4-seed flow 定稿(15:40)

teacher dm40 中位:s0 2.75 / s1 2.17 / s2 0.86 / s3 2.41,4-seed 均值 2.05。**逐 seed 配对终裁**:
- **W7 > teacher:4/4 全同向**(+64% 按均值)——接力高运动幅值为硬结论。
- **E1a < teacher:仅 3/4**(s3 反向 2.80>2.41),幅度收窄至 −12% → 表述弱化为"不高于 teacher(3/4 同向)"。
- seed2 三模型齐低(0.46/0.86/1.27)——初始噪声主导部分动态水平,配对设计的必要性实证。Ch2 发现 5 须按此再修订(待今晚一并改)。

## 5. E5 探针首版结果(14:59,`reports/e5/probe_v1.json`,64 clip/侧)

- 工具:`exp/eval/e5_probe.py`(特征路径逐字段对齐训练侧 `_compute_real_feat`;null-text 统一条件;global avg-pool + torch 逻辑回归 5-fold AUC + Fréchet)。
- **全层高可分**(6-t 均值 AUC 0.88-0.92);head 层 {15,22,29} 在合理区间但非最优——L7 略高,L29 一致最低("上游选层合理但非唯一"审计素材)。
- **t 依赖极强**:t=0.2 时 AUC=1.0(全侧全层),t=0.999 时 0.28-0.52(≈随机)——判别器有效工作区在中低噪端,为 E2b (t,ε) 配对消融提供机制背景。
- **诚实对照**:teachergen(teacher 50-step 生成)与 real 的可分性(0.90-0.92)不低于学生侧——可分性主要源于"生成域 vs 真实域"共有差异而非蒸馏退化;且 real=OpenVid(训练域)、fake=q150 prompt(域外),含内容域差成分。E5 解读一律带此注(红线:非 headline 贡献)。

## 6. E2b 起训波折与容错门控(15:54-16:01)

- 用户拍板起训 E2b(`gan_use_same_t_noise=False`,GAN weight 0.03 上游默认,`sprint_e2b_relay_indep_tnoise`,config import + dry-run 均过)。
- **GPU0 抢卡竞态三连**:15:54 首启在 preprocess_data OOM(他人 17.6G 短任务插入精检-分配窗口);15:57 重启同样被抢(15.2G);16:01 门控三试时 precheck 直接拒绝——GPU0 判定为抢卡热点。两次失败均非配方问题(GAN 显存构型未及受测)。
- 处置:`exp/eval/e2b_autolaunch.sh`(PID 4120818)容错门控——fallback 等 GPU 1/3/6/7(eval burst 结束即空,避开 GPU0)自动重试至 07-24 06:00;成功后自动挂 stall guard + first-health。
- 残留监护(两次失败的 stall guard/health writer)均已按 PID 单杀清理。
- 16:18 起训(2/3/5/7,门控 v2 首攻即中):iter1 正常(loss 1.43,peak 27.93GB,显存 95%)。
- **16:2x 配方性 OOM 死亡(iter 2-5 间)**:rank 自身 31.33G 用满再要 96MiB 失败——**GAN-on 的 4 卡形态确证挤不进 32G**(E2a 能跑纯因 GAN 分支关闭;峰值构型高于 iter1 的 27.9GB)。非抢卡。
- 显存杠杆盘点(诊断闭环):micro-batch 已是 1/rank(batch_size_global=16 → 框架自动 grad_accum=4,累积路已走满);FSDP 已全分片;无 activation-checkpointing 支持;`offload_module_in_decoding` 不涉训练峰值。**唯一非配方杠杆 = `trainer.fsdp_cpu_offload=True`**(参数/优化器下放 CPU,速度代价未实测,估 2-3×;2500 iter 或达 60-120h)。判别器缩容/降分辨率/减 batch 均属配方变更,按红线不擅动。
- 处置(用户 16:4x 裁决):**协商拿到全节点——8 卡训 E2b**。门控几经波折(GPU4 实际 20:20 后才交出;18:41 起两个门控实例并存过,无害):**20:35:08 8 卡起训成功**,`sprint_e2b_relay_indep_tnoise_20260723_203508.log`。first-health:iter1 45.1s,peak 26.59GB(8 卡形态余量转宽);**稳态实测 31.9s/iter**(iter20@10.6min)→ 2500 iter ≈ 22.1h,**预计 07-24 ~18:45 完成**。stall guard 在岗,残留门控已清,夜间监控至 07-24 08:30。eval burst 四队列(q150 六维/d40 LPIPS/dm40 flow/teacher flow)18:24 前全部 done、零 FAIL;E2a q150 六维与 d40 多样性数字待抽取(custom-input json 结构与 vb946 不同)。

## 7. Ch3 第一信号:E2a(GAN=0)flow 全 sweep(16:48 抽取)

dm40 RAFT 中位(seed0,与 E0 同协议):

| E2a 档 | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|
| flow 中位 | 2.14 | 2.40 | 2.30 | 2.40 | 2.74 |

- **GAN=0 臂全 sweep 贴近 teacher(2.75/s0),显著低于同配方 GAN-on 的 W7(4.44/s0)**——初步指向"W7 的高运动幅值部分来自 GAN 分支"。此为判别器审计第一个实质分化信号;定读需并入 q150 质量维(已打分,custom-input json 为 list 结构,抽取待修)与 d40×8 多样性(gpu1 队列 30/45,~18:50 完)。E2b((t,ε) 配对)结果将给第三个坐标。
