# 2026-07-22 — full VBench 打分启动(e1a_flat + w7_flat,16 维,用户拍板)

用户 15:0x 拍板"2/4/5 空出,直接执行打分"。部署 15:11:30,烟测通过。

## 配置

- 输入:`reports/vb946/{e1a_flat,w7_flat}`(各 4720 条,{prompt}-{k}.mp4 标准命名;946 prompt 中 2 条重复文本已实证,与 skipped=10 吻合,VBench 官方同为同名处理,打分无缺失)。
- 队列:`exp/eval/g2q/vb_score_{e1a,w7}.jobs`,各 16 行(每行 = `score_vb946_one.sh <flat> <out> <dim>`,standard mode,done-marker 幂等)。**每模型一条串行队列**(避免"最新 eval_results.json 含维度键"成功检查在同 out 目录下的并行竞态);两模型并行。
- 分配:e1a→GPU2(队列 PID 4037121),w7→GPU4(PID 4037122);GPU5 留作缓冲。E2a(1/3/6/7)与 e1a@2000 补测门控(等 GPU6)均不受影响。
- 维序:已缓存权重的 6 维在前(subject/background/motion_smoothness/dynamic_degree/aesthetic/imaging),新权重维殿后(color/object_class/multiple_objects/human_action/scene/spatial_relationship/appearance_style/temporal_style/overall_consistency),temporal_flickering 居中。
- 日志:`reports/vb946/score_{e1a,w7}_queue.log`;结果 json + `done_<dim>`:`reports/vb946/scores/{e1a,w7}/`。

## 烟测(15:12:45)

- 双队列首维 subject_consistency 正常推进:DINO 权重命中本地缓存,4720 视频=360 批,~1.3 it/s → 首维 ~4.5 min/模型;无下载/环境障碍。
- 重维(aesthetic/imaging ~10×、ViCLIP 维需拉新权重)在后段,整体仍按"数小时至 1-2 天"预估;新权重维若 FAIL(mirror dance),run_queue 容错跳过,凭 done-marker 幂等补跑即可。

## 协议脚注(写作用,先记录)

- temporal_flickering:官方 25 样本/prompt 于专属 subset;我们 5 样本 → 报 Total 需脚注该偏差或弃该维。
- 946 prompt 含 2 条重复(《Hokusai 海滩》《外滩油画》各出现 2 次):同名文件顶替,合计 4720/4730 唯一样本,与官方处理一致。
- 对比数字(CoDMD 84.46 等)仅作文献坐标,协议差异必须脚注(T3 红线 8,禁 SOTA 对比表述)。

## 结果(待回填)

- 每维分数以 `scores/{e1a,w7}/` 最新 `*_eval_results.json` 为准;齐后跑聚合出主表。
