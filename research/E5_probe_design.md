# E5 离线探针设计稿(2026-07-22 晚起草;代码明日实现)

任务来源:`experiment_plan.md` E5 行——对现有 teacher/8-step/4-step ckpt 做**判别器特征层可分性**(15/22/29 vs 备选层,matched t)+ **off-manifold 度量**;零训练,1-2 卡空隙,~0.5 天代码 + few GPU·h。
Claim 红线(plan §红线 5):"to our knowledge" + 记录检索覆盖;非 headline 贡献。

## 研究问题

判别器挂在 frozen-teacher 第 15/22/29 层(代码验证 2026-07-06)——这三层对"real vs 学生生成"的线性可分性在全部层里处于什么位置?可分性如何随 t 与学生代际(8-step W5 / 4-step W7 / 直蒸 E1a)变化?学生样本在各层特征空间中的 off-manifold 程度多大?这直接支撑 Ch3 判别器审计的机制解释,并为 E2 系列的 GAN 消融结果提供解读维度。

## 输入资产(全部已存在,零生成)

| 侧 | 来源 | 数量目标 |
|---|---|---|
| real | `/data/datasets/OpenVid-1M/webdataset` 随机 shard 抽 clip → VAE encode | 256 clip |
| fake-W7 | `reports/e0/w7/0001000_student_4step/`(q150/d40 mp4)→ VAE encode | 256 |
| fake-E1a | `reports/e0/e1a/0001000_student_4step/` 同上 | 256 |
| fake-W5(8-step) | `reports/e0/` 下 W5 生成物(INDEX 核对路径) | 256 |
| teacher-gen(可选对照) | teacher 50-step 生成物(E0 q150) | 256 |

## 协议

1. **t 集**:与判别器训练所见对齐——取 4-step t_list 锚点(以 W7 run `config.yaml` 为准)+ 中点补 2 个;对每个 clip latent 以相同 ε 加噪至各 t。
2. **层集**:head 层 {15,22,29} + 备选 {3,7,11,19,25,27}(Wan1.3B DiT 共 30 blocks;明日以代码实测层数为准)。
3. **特征**:逐层输出按判别器 head 同款池化(**明日第一步**:读 `fastgen/.../dmd2.py` 判别器 `_compute_*_feat` 的 hook 与池化方式,逐字段对齐,不自创)。
4. **可分性**:每 (layer, t, student) 上 real-vs-fake 逻辑回归线性探针,5-fold CV,报 AUC(主)+acc(附)。
5. **off-manifold**:同特征上 per-layer Fréchet 距离(均值+协方差);报 layer×student 矩阵。
6. 输出:`reports/e5/probe_auc.json` + `probe_fd.json` + 汇总表;脚本 `exp/eval/e5_probe.py`(遵循 exp/eval 工具链惯例:md5/参数落 json、幂等)。

## 资源与排期

- 特征抽取 ~2-3 GPU·h(单卡,batch 小、teacher forward only);探针/FD 为 CPU。
- 排期:明日(07-23)E2a 结束(~12:00)后空隙;若 GPU0 的 E1b 链路仍在跑,用 1/3/6/7 释放的卡。

## 风险与核查清单(编码前必查)

- [ ] dmd2.py 判别器特征:层号、池化、是否 t-conditioned、text-cond 处理——全部照抄训练侧。
- [ ] VAE encode mp4 的帧数/fps/分辨率与训练 latent 规格对齐(832×480×81f);色彩范围 [-1,1]。
- [ ] real 侧 clip 采样与训练 dataloader 同预处理(crop/resize 路径)。
- [ ] t 加噪公式与训练 scheduler 一致(flow-matching 插值式)。
- [ ] 探针样本数 256/侧的 AUC 方差:bootstrap 95% CI 一并报。
