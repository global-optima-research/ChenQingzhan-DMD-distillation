# 冲刺执行任务书(2026-07-13 分发)

你是实验执行 agent,在本仓库(`/Users/chenhingchin/Desktop/ChenQingzhan-DMD-distillation`)工作,负责远端 `ust_ip` 上的冲刺实验执行。今天是 2026-07-13。本任务书自包含;本目录 CLAUDE.md 的角色设定以本任务书为准。

## 边界(先读)

- 只做本任务书列出的事;不修改 `exp/run.sh`、`exp/eval_sweep.sh` 与任何既有 python config;新变体一律新建文件、新 `log_config.name`。
- 严禁删除任何远端文件;`/data/chenqingzhan/archive_pre_sprint_20260713/` 不许碰。
- 同一时间最多一个训练 run;每次启动前先 `DRY_RUN=1` 看命令(GPU 预检内置于 run.sh)。
- 远端 sshd 会限流高频连接:命令尽量合并成少量会话;连接被拒等 20 秒重试。
- 训练崩溃:抓 log 尾部 80 行写进记录文件,**先报告,不要自动重启**。

## 环境事实

- 远端仓库 `/data/chenqingzhan/FastGen`(commit `936bf7c`);启动方式:`bash exp/run.sh exp/confs/<X>.conf`。
- 三区:配置区 `exp/`(confs + configs + 薄脚本);记录区 `experiment/`(`INDEX.md` 索引、`TEMPLATE.md` 模板、`LAUNCHES.log` 自动追加);日志区 `FASTGEN_OUTPUT/fastgen/wan_dmd2/<log_config.name>/`。
- wrapper log:`/data/chenqingzhan/logs/<EXP_ID>_<时间戳>.log`;conda:`source /data/chenqingzhan/miniconda3/etc/profile.d/conda.sh && conda activate fastgen`。
- 保留的存量 checkpoint(均只有 `net_model`+`pth`):W7(`..._step4_from_step8_8node`)0000500-0002500 五档;W5(`..._step8_lr_original`)五档;W1(`..._global_8`)0001000/0001500/0002000;W3(`..._step8_freq`)0001000;W4(`..._step8_normalize`)0001500。
- 本地一行提交备选:`bash experiments/bin/run_remote_script.sh experiments/configs/wan21_sprint.env`(改其 `CONF=` 行换实验)。

## 任务 A(今晚立刻):启动 E1b 并确认健康

1. `DRY_RUN=1 bash exp/run.sh exp/confs/E1b.conf` —— 确认预检通过、命令为 `nproc_per_node=6` + `e1b_direct_lr1e5_b12.py`。
2. 去掉 DRY_RUN 真跑;记录 pid 与 log 路径。
3. 约 3 分钟后 tail log:FSDP wrap 完成、dataloader 建立、iteration 1 指标打印、无 OOM/NaN → 发射成功。
4. 约 30 分钟后复查:iteration 持续增长;ETA 参考同配方实测 2500 iter ≈ 23h(5000 ≈ 46h);首个 checkpoint 应出现在 iter 500。
5. 在 `experiment/` 新建 `sprint_e1b_direct50to4_lr1e5_b12.md`(抄 TEMPLATE.md,填启动信息)。

## 任务 B(与 A 并行,只用 GPU 6,7):E0 工具链 + 存量模型第一批数字

目标:2026-07-14 晚前出第一批表。

1. 工具链(代码放 `FastGen/exp/eval/`,附简短 README;装在 fastgen env 或新建 env):
   - VBench 官方 repo,custom-video 模式,只跑 6 个质量维(subject consistency / background consistency / motion smoothness / **dynamic degree** / aesthetic quality / imaging quality);
   - 跨 seed 多样性:同 prompt 8 seeds 的成对 LPIPS 均值(帧均);DINO 特征距离可选;
   - 固定 prompt 集两份存 `exp/eval/`:quality ~150 条(建议从 VBench 官方 prompt 套件抽样,记录来源与抽样方式)、diversity 40 条。
2. 生成(先学生后 teacher;输出统一 `FASTGEN_OUTPUT/fastgen/wan_dmd2/reports/e0/<model_tag>/`):
   - 每 checkpoint:quality 集 1 seed + diversity 集 8 seeds;
   - 顺序:W7 五档 → W1 0001000 → W5 0002500(8 步推理,config 用 `fastgen/configs/experiments/WanT2V/config_dmd2_step8_2k.py`)→ W3 0001000(同上 config)→ W1 0001500/0002000 → W4 0001500;
   - ⚠️ **W4 陷阱**:它是"均匀 t_list"消融,现有 step8 config 的 t_list 是插值版,直接用会错。新建 `exp/configs/eval_w4_step8_uniform.py`(拷 step8_2k,t_list 换 `[0.999,0.875,0.75,0.625,0.5,0.375,0.25,0.125,0.0]`),并与该 run 目录 `config.yaml` 核对后再用;
   - teacher(50 步,≈165s/条,最慢):两份 prompt 集各生成一遍,挂夜里跑。
3. 打分汇总:`experiment/E0_quant.md`——一行一个 (model, ckpt),列 = 6 维 + DD + 多样性;原始 csv 留 `reports/e0/`。
4. 自检(必须如实报告):按肉眼记录预期 teacher ≥ W7 最佳 > W1;若排序不符,原样标注,不要藏。

## 后续队列(每个节点完成后给用户 ≤10 行总结,等确认再走下一步)

1. E1b 收(约 07-15):复制 `EVAL_e1a.conf` 为 `EVAL_e1b.conf` 改路径跑推理 sweep + E0 协议打分 → 启动 E1a(`exp/confs/E1a.conf`,8 卡,5000 iter ≈ 2 天余)。
2. E1a 收(约 07-17):两直蒸臂 + W7 同表呈现 → **停下,交用户裁决**(接力 vs 直蒸是论文级决定)→ 确认后依次 E2a → E2b → E2c(各 2500 iter ≈ 1 天,启动即换 conf)。
3. E2c 特别项:首个 20-iter log 出来后对比 `gan_loss_ar1` 与 `gan_loss_disc` 量级;若差一个数量级以上,新建 `e2c_relay_r1w<新值>.py`(不改旧文件)换 conf 重启,只调这一次。
4. 落后预案(已批准,按序启用):审计臂缩到 2000 iter → 砍 E2c → E1 评估网格缩到 {1000, 2500, 5000}。

## 汇报格式

每完成一个节点,对话里给 ≤10 行:启动/完成了什么、数字表路径、健康状况、偏差与待用户决策点。远端记录文件(`experiment/*.md`)是第一现场,结论不要只留在对话里。
