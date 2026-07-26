# 2026-07-26 — 冻结前核查与修订(A/B 四项核查 + 全部修订清单;供 planner 终审)

## A. 远端核查(只读)

**A1 W1 速度数字**:metrics.csv 定位于 `wan21_t2v_dmd2_OpenVid_global_8/reports/eval_10prompts/metrics.csv`(canonical 原引"reports/eval_10prompts/"缺 run 目录前缀,已随复核注补全)。实读:teacher avg 165.24s(10 条,min 164.23/max 166.77)——**与记录精确一致**;student 9 档 avg 6.591(@1000)至 6.656(@3000),speedup 24.83-25.07×;@500 行 missing(0 videos,记录未引用,无碍。**唯一不符项:记录范围"6.59-6.63"上限与实读 6.656 差 0.026s**(@2500/@3000/@3500 三档超出)——按纪律未改数字,是否改写"6.59-6.66"待 planner 裁;各处警示句已改"已复核+路径+勘误待裁"。

**A2 dm40 组成**:`exp/eval/make_motion_set.py` 头部注释与 Ch1 §1.7 描述**逐要点一致**(Source A = 官方 human_action.txt 99 条 uniform stride 取 20;Source B = all_dimension.txt 经 MOTION_CUE 正则 AND NOT STATIC_BLOCK 过滤后 uniform stride 取 20;确定性无随机)。md5 实测 `324d75a0fee9afd499d5bf5c8e2af450` 前缀与记录 324d75a0 一致。**核查通过,Ch1 §1.7 无需修改**。

## B. 官方取源

**B3 Quality Score**:出处 Vchitect/VBench master `scripts/cal_final_score.py`(公式)+ `scripts/constant.py`(QUALITY_LIST 7 维、DIM_WEIGHT 全 1 除 dynamic_degree=0.5、NORMALIZE_DIC min-max、分母 6.5、QUALITY_WEIGHT:SEMANTIC_WEIGHT=4:1)。四值:**E2a@2000 85.50 / W7 84.47 / E1b 83.62 / E1a 82.80**(公式、每维归一化中间值、解释义务与 CoDMD 84.46 巧合强警示,全在 `2026-07-26-e2a-vb946-fourth-row.md` 新小节;storyline P14 同步)。

**B4 temporal_flickering**:官方 `prompts/README.md`@master 原文坐实——"for the Temporal Flickering dimension, sample **25 videos** to ensure sufficient coverage **after applying the static filter**",专属 75-prompt 子集。比此前记录多一要素(static_filter 预筛)。Ch1 §1.7 与 storyline P14 脚注已按实修订并附出处。

## C/D. 修订清单(全部完成)

1. canonical:①E2a n=3 句改"逐 seed 配对 3/3 同向、带缘轻微接触";②多样性"0.60-0.65"→"0.59-0.64(含审计臂)";③CD-FVD 加"(计划项,实际未执行…)"注;④W1 节两处警示改"已复核+路径+勘误待裁"。
2. Ch2:发现 5 第 4 条按 07-23 §4 终裁改写(teacher 4-seed、W7>teacher 4/4 +64% 硬结论、E1a 3/4 "不高于 teacher"、删旧 +22%/−34%、seed2 三模型齐低句);效度威胁改"四方已 4-seed 配对;E2a/E2b sweep 仍 seed0(Ch3 纪律同)"。
3. Ch3 §3.4:补 E1a aesthetic @1000 0.5665 → @5000 0.5196 pattern 佐证句(带"非严格单调/仅 pattern 一致性/禁作归因"红线与出处)。
4. Ch1:§1.4/§1.8 速度警示改已复核;§1.7 flickering 脚注升级(25 样本+static_filter+出处)。storyline:P2 出处改已复核;P14 加 Quality Score 四值、权重出处、E1a 最低解释义务、84.47/84.46 巧合禁比。

## 冻结条件判定

- **达到冻结条件,除一项待裁**:A2/B3/B4 全通过零数字变更;A1 teacher 精确一致、唯 student 范围上限 0.026s 勘误(6.63→6.656)待 planner 裁定写法——裁定后即可宣布冻结。
- 未决项仅此一条;另注意 84.47/84.46 巧合已设禁比红线,建议 planner 终审时确认该警示在最终页面存活。
