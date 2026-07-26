# 2026-07-26 — E2a@2000 入 full-VBench 主表(第四行,审计臂标注)

链路(07-25 16:45 启动,20:43 收数,全自动):5 卡并行生成 946×5 → rename(hardlink flat)→ 12 维打分;GRiT 4 维照例缺席(detectron2)。

## full-VBench 主表(standard mode,946×5;四行定稿)

| 维度 | E1a 直蒸@1000(G2 加冕) | W7 接力@1000 | E1b 直蒸B@500 | **E2a@2000(审计臂,GAN=0)** |
|---|---|---|---|---|
| subject_consistency | 0.9727 | 0.9693 | 0.9673 | 0.9753 |
| background_consistency | 0.9579 | 0.9508 | 0.9416 | 0.9581 |
| motion_smoothness | 0.9812 | 0.9727 | 0.9747 | 0.9786 |
| dynamic_degree | 0.5806 | 0.9111 | 0.8806 | 0.8000 |
| aesthetic_quality | 0.5967 | 0.6087 | 0.5802 | **0.6482** |
| imaging_quality | 0.6687 | 0.6687 | 0.6614 | **0.6924** |
| temporal_flickering | 0.9894 | 0.9796 | 0.9810 | 0.9878 |
| human_action | 0.690 | 0.794 | 0.716 | 0.776 |
| scene | 0.2173 | 0.2922 | 0.2225 | 0.2974 |
| appearance_style | 0.1990 | 0.1982 | 0.2003 | 0.2010 |
| temporal_style | 0.2214 | 0.2305 | 0.2260 | 0.2283 |
| overall_consistency | 0.2240 | 0.2386 | 0.2298 | 0.2391 |

## 表注(写作红线,与验收整改一致)

- 第四行为**审计臂(GAN=0,relay 配方)**,由 Ch3 判别器审计产出、事后补入;**加冕叙事不变——按 G2 预注册结果仍为 E1a@1000**,E2a 行的作用是完整性与审计叙事支撑,非重新加冕。
- E2a 在 vb946 域同样质量维领先(aes 0.648/imaging 0.692 均为四行最高)且动态度不塌(0.80)——与 q150 域三臂结论同向,跨域一致性增强 Ch3 裁读 2。**禁写"超越 teacher"**;q150 域 teacher 对照的表述限定为"静态画质指标更高、多样性缺口仍在(E2a div 0.586-0.604 vs teacher 0.732)"。
- 12 维含官方 Quality Score 全部 7 维(可按官方权重合成);Semantic/Total 因 GRiT 4 维不可算——表脚注声明,非随意删维。
- E2a 的 vb946 仅 @2000 单档(审计用),不做全 sweep;dynamic_degree 跨域口径不混引(q150 0.567 / vb946 0.800 域不同)。

## Quality Score 官方权重合成(2026-07-26,冻结核查 B3)

**权重出处**:Vchitect/VBench 官方仓库 `scripts/cal_final_score.py` + `scripts/constant.py`(master 分支,https://github.com/Vchitect/VBench/blob/master/scripts/constant.py,2026-07-26 取证)。
**公式**:每维 `norm = (raw − Min)/(Max − Min)`,加权求和后除以质量维权重和 6.5。七个质量维参数:subject (Min .1462, Max 1, w1) / background (.2615, 1, w1) / flickering (.6293, 1, w1) / motion_smooth (.706, .9975, w1) / dynamic (0, 1, **w0.5**) / aesthetic (0, 1, w1) / imaging (0, 1, w1)。

各维归一化中间值(供 planner 独立重算):

| 维(norm) | E1a@1000 | W7@1000 | E1b@500 | E2a@2000 |
|---|---|---|---|---|
| subject | 0.96803 | 0.96404 | 0.96170 | 0.97107 |
| background | 0.94299 | 0.93338 | 0.92092 | 0.94326 |
| flickering | 0.97141 | 0.94497 | 0.94875 | 0.96709 |
| motion_smooth | 0.94408 | 0.91492 | 0.92178 | 0.93516 |
| dynamic(×0.5 计入) | 0.5806 | 0.9111 | 0.8806 | 0.8000 |
| aesthetic | 0.5967 | 0.6087 | 0.5802 | 0.6482 |
| imaging | 0.6687 | 0.6687 | 0.6614 | 0.6924 |

**Quality Score:E2a@2000 = 0.85495(85.50) / W7@1000 = 0.84466(84.47) / E1b@500 = 0.83616(83.62) / E1a@1000 = 0.82803(82.80)**。

- 解释义务:E1a(G2 加冕臂)合成分最低,主因 dynamic 维(0.581,权重 0.5);页面呈现须并列权重结构说明,加冕依据是 G2 预注册协议(q150 六维+多样性),不受该合成分影响。
- **强警示:W7 84.47 与 CoDMD 文献值 84.46 为数字巧合**——前者 7 维 Quality Score、后者 16 维 Total,量纲不同,禁止任何并列或比较表述。
- Semantic/Total 不可算(GRiT 4 维缺失),维持声明。
