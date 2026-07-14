# 2026-07-13 — E1b 启动 + E0 生成队列开跑

## E1b(任务 A)

- run:`sprint_e1b_direct50to4_lr1e5_b12`(50→4 直蒸对照 B 臂 = W5 配方 lr1e-5/b12,5000 iter,init=teacher)
- config:远端 `exp/configs/e1b_direct_lr1e5_b12.py`;conf `exp/confs/E1b.conf`;GPU 0-5
- wrapper log:`/data/chenqingzhan/logs/sprint_e1b_direct50to4_lr1e5_b12_20260713_165934.log`;torchrun pid 190874
- run 目录:`FASTGEN_OUTPUT/fastgen/wan_dmd2/sprint_e1b_direct50to4_lr1e5_b12`
- 状态:**健康**。16:59 启动;17:01 FSDP wrap 完成、iter_start:0;17:53 到 iter 100,avg 31.5 s/iter,total_loss 0.40-0.50,峰值显存 27.9G,6 卡 100% 利用
- 日志唯一 "ERROR" 命中为 WDS shard 名排序告警(`shard-000015 is not a number`),dataloader 正常,非故障
- ETA:31.5 s/iter × 5000 ≈ 43.8h → **预计 2026-07-15 ~13:45 完成**(优于 46h 参考值)
- 远端记录:`experiment/sprint_e1b_direct50to4_lr1e5_b12.md`;INDEX 已改"进行中"

## E0(任务 B)

- prompt 集(确定性抽样自 VBench 官方 all_dimension.txt 946 条,fetched 2026-07-13):q150(md5 690f2919...)、d40 ⊂ q150(md5 b4c1f9e3...)、teacher 两半 A75/B75;方法与 md5 记录在 `exp/eval/`(`make_prompt_sets.py` + prompts/MD5SUMS)
- 生成:复用 June 泛化推理脚本(未改动任何既有脚本);W4 用新建 `exp/configs/eval_w4_step8_uniform.py`(均匀 t_list 已对 run config.yaml 核对);teacher 用新建 `exp/eval/gen_teacher.sh`(50 步 CFG5)
- 队列:GPU6/GPU7 双队列 17:54 启动(runner pid 334883/334884),日志 `FASTGEN_OUTPUT/fastgen/wan_dmd2/reports/e0/queue_gpu{6,7}_20260713.log`;全部 11 个存量 checkpoint(W7×5/W1×3/W5/W3/W4)存在性已核
- 输出:`FASTGEN_OUTPUT/fastgen/wan_dmd2/reports/e0/<tag>/...`;学生今晚 ~01:30 完成,teacher 过夜到明日 ~11:30
- 评测环境:独立 conda env `e0eval`(torch 2.11+cu128 + vbench + lpips)后台安装中;VBench 6 维冒烟(现成 W7 eval-10 十条视频)排在 GPU7 队列 ~20:00 档

## 用户指示落账(2026-07-13 晚)

1. GPU6 瞬时 0% 利用率(显存持有 20.7G)按"视频间隙/模型加载"处理,由 20:25 自动检查确认队列未卡死,不额外干预。
2. **loss 基线口径**:E1b 的 total_loss 0.40-0.56(iter ≤100,LocalStats callback)无六月历史参照,立为基线;E1a 起跑后用**完全相同口径**(同 callback、同 iter 段)对比两臂 loss 轨迹,该对比本身作为一行数据进记录。
3. **日程账(用户核定)**:E1b 07-15 13:45 收 → 收尾 burst(8 卡 ~2h:E1b 全 10 档 q150 生成+打分,d40×8seeds 只做 6 维最优 2-3 档)→ **E1a 立即接**(8 卡满载,~07-17 晚收)→ 裁决停点(用户)→ E2 三臂 07-17~20。**E2c 压冻结日(07-20)边缘:2000-iter 预案待命,07-18 晚落后超半天即直接启用,不等。**
4. 明晨两个 gate:(a) VBench 冒烟——失败则最高优先级修(卡整条 E0 打分链);(b) teacher 过夜生成收尾(~11:30)。
5. **G1 校准交付**:第一批 E0 表先贴 teacher / W7-best / W1 三行给用户(量化排序 vs 肉眼记录),决定 Ch1/Ch2 写作口径是否调整。
6. 下一个用户停点:~07-17 接力 vs 直蒸同表裁决。

## 下一步

- ~20:25 检查点:冒烟结果 + 队列进度 + E1b iter(+远端 E1b 记录补记 30min 结果)
- 明日:teacher 完成后 GPU6/7 转打分;`experiment/E0_quant.md` 第一批表(目标 07-14 晚前;teacher/W7/W1 三行先行)
- E1b 完成(07-15 13:45)→ 2h 收尾 burst → 启动 E1a
