# 2026-07-21 — vb946 全量生成启动 / gpu3 事件 / E2a 门控 v1→v2

运维注记(非质量结论)。远端第一现场:`FASTGEN_OUTPUT/fastgen/wan_dmd2/reports/vb946*`、`reports/e0/E2A_LAUNCH_marker.txt`、`exp/eval/VB946_PLAN.md`。

## 1. vb946 全量生成(E1a@1000 + W7@1000,946×5 seeds)

- 10:19 按 `exp/eval/VB946_PLAN.md` 启动:4 条 run_queue(`exp/eval/g2q/vb_gpu{1,3,6,7}.jobs`,共 11 行 job = 10 个 set + 1 行补跑),队列日志在 `reports/vb946_gpu{1,3,6,7}.log`。
- prompt 集:`vb946_s0..s4.txt` 均为官方 `vbench_all_dimension.txt` 的逐字复制(md5 `7f0f7e52` 五份一致,946 行);seed 仅通过 `trainer.seed` 区分。
- 生成 wrapper 约定:落盘到 `vb946_s{k}_hq`(HQ x265),**job 完成时由 wrapper 自行 `mv` 成 `vb946_s{k}`**——中途看到 `_hq` 目录属正常在飞状态,勿手工改名/建链。
- 17:59 盘点:e1a s0-s4 全齐(各 946);w7 s1/s2/s4 齐(各 946),s0(729/946)、s3(684/946)在飞,按 ~5.6 条/min 预计 18:40-18:50 完成。
- rename-watcher(PID 1948635)已挂:两队列退出后自动跑 `vb946_rename.py`(hardlink 成 VBench 标准 `{prompt}-{k}.mp4` flat 目录)→ `reports/vb946/{e1a_flat,w7_flat}` + `RENAME_DONE.txt`。

## 2. gpu3 / gpu7 事件

- **13:06:27 gpu3 上 w7 s0 秒挂**:`CUDA error: CUBLAS_STATUS_ALLOC_FAILED`(torchrun exitcode 1)——他人 27.9G 任务上了 gpu3,与我方 20.8G 无法共存。run_queue 容错继续,14:05 把 w7 s0 **追加**到 `vb_gpu6.jobs` 补跑(run_queue 为 while-read 增量读文件,追加行会被在跑队列消费——已验证生效)。
- **15:53 `vb946_gpu3.log` 退出 = 正常完成**(`done: 3 jobs, 1 failed`,失败即上述 s0);16:17 gpu7 队列正常完成(`2 jobs, 0 failed`),随后 gpu7 被他人进程(23G)占用。
- 17:48 GPU 快照:对方占 0/2/3/4/5/7,我方 1/6 跑 vb946 收尾。

## 3. E2a 门控 v1→v2(Plan A 已批)

- v1(`exp/eval/e2a_autolaunch.sh`,PID 3488767,15:44 挂):等 vb946 队列退出 → GPU 1/3/6/7 精检 **仅重试 10×60s** → 启动 E2a(`exp/confs/E2a_4gpu.conf`,NCCL fail-fast,stall-guard 2600s,+6min 自动写 first-health)。鉴于 gpu3/gpu7 被他人长占,10 分钟窗口必然 ABORT。
- 17:52 换 **v2**(`e2a_autolaunch_v2.sh`):复制后唯一改动 = 精检循环改为**长窗口,每 120s 重试至 2026-07-22 09:00**(`DEADLINE=$(date -d "2026-07-22 09:00" +%s)`);其余逐行不动(diff 已核)。严禁原地编辑在跑脚本(bash 按字节偏移续读)。
- 旧门控按 PID **单次** `kill 3488767`(pgrep 先确认恰为该 PID;不用 kill 循环/pkill 模式——07-20 事故教训),确认退出后 `nohup bash exp/eval/e2a_autolaunch_v2.sh &`(PID 4096392),marker 17:52:30 进入 waiting。run_queue×2、rename-watcher、两推理进程全程未动。
- 操作插曲(透明记录):17:55 曾为 rename 兼容给 `vb946_s{0,3}` 建了指向 `_hq` 的软链,17:59 发现 wrapper 完成时会自行 `mv _hq → 无后缀`(软链会令该 mv 报错)后**已删除**,距最早完成点 ~40 min,无影响。
- 预期链路:~18:45 两队列退出 → watcher 自动 rename → v2 精检(1/6 已释放,等他人让出 3/7)→ E2a 起训,4 rank 显存 30/32G 边缘(OOM 会快速失败),+6min marker 写 first-health;本地已挂 240s 轮询监控,出快照/ABORT 即汇报。

## 3.5 晚间核验(20:36 快照)

- **vb946 生成 + rename 全链路闭环**:两队列 ~18:47 前退出;w7 s0/s3 由 wrapper 正常 mv 成标准名;rename-watcher 命令串完好,18:47:20 写出 `RENAME_DONE.txt`。
- rename 核验:e1a 与 w7 均 `linked=4720 skipped=10 missing=0`,flat 各 4720 条。4720+10=4730=946×5;skipped=10 与"946 条官方 prompt 中约 2 条重复文本 × 5 seeds"自洽(重复 prompt 的 dst 同名,首个 index 已建即跳过;VBench 官方对重复 prompt 同样按同名文件处理,打分无缺失)——重复条目待 `sort | uniq -d` 坐实后在打分注记里脚注。
- E2a 门控 v2 实战确认:20:35 已精检 55 轮(每 120s),gpu7 已释放,**仅剩 GPU 3**(他人 27.9G 任务)未让出;v1 的 10 轮窗口早已耗尽,换装必要性得到实证。
- `experiment/LAUNCHES.log` 基线:末条为 2026-07-16 E1a;E2a 起训应新增一条(比对依据)。磁盘 1193G。
- 监控口径(用户 2026-07-21 晚指令):本地轮询放宽至 600s;E2a 起训后核 LAUNCHES.log 新条目 + marker first-health(显存 4/rank ~30/32G 边缘,OOM 只如实上报不改配方)并记录实测 iter/s;磁盘 <500G 即告警。
- **窗口延期(用户 2026-07-22 00:35 指令)**:门控 v2→v3(`e2a_autolaunch_v3.sh`,唯一改动 deadline 09:00→**12:00**;v2 PID 4096392 按 PID 单杀,v3 PID 1478965 nohup 挂起,00:36:57 进入精检)。v2 阶段 174 轮 try 历史备份于 `reports/e0/E2A_LAUNCH_marker_v2_history.txt`。本地监控超时同步延至 12:30。

## 4. full VBench 打分链路盘点(只报告,未启动)

- 就绪:`score_vb946_one.sh`(单维×单 flat 目录,standard mode,done-marker 幂等,HF mirror + weights-only 补丁)、`vbench` CLI 在 e0eval 环境、flat 目录将由 watcher 自动产出。
- 未备:16 维 × 2 模型的编排 jobs 文件尚无(计划书建议 per-dim 平铺;先烟测一个便宜维——部分维要拉新权重)。
- 成本(VB946_PLAN.md):打分 ~1-2 GPU·天/模型,与 E2a 抢 GPU 1/3/6/7;**排期等用户拍板**。协议注意:temporal_flickering 官方为 25 样本/prompt,我们 5 样本,报 Total 需脚注或弃该维。
