// DMD2 progress report deck builder — 2026-07-28 (spec: research/report_storyline_0728.md v4)
// Numbers frozen at commit 0c28815; every figure below is transcribed from the cited source notes.
// Build: node build_deck.js  ->  DMD2_report_0728.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ASSETS = path.resolve(__dirname, "../slides_assets");
const COVERS = path.resolve(__dirname, "covers");

const FONT = "PingFang SC";
const NAVY = "1F3864";
const NAVY_SOFT = "8FAADC";
const ORANGE = "C55A11";
const INK = "333333";
const GRAY = "767676";
const LIGHT = "F2F3F5";
const PANEL = "FAFBFD";
const BORDER = "D9D9D9";
const QBLOCK = "DDE3EF";
const SBLOCK = "ECECEC";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

// --- QA instrumentation: record every element op (geometry + text) for the HTML mirror renderer ---
const QA_LOG = [];
const _addSlide = pres.addSlide.bind(pres);
pres.addSlide = function () {
  const s = _addSlide();
  const rec = { ops: [] };
  QA_LOG.push(rec);
  ["addText", "addShape", "addTable", "addImage", "addMedia"].forEach((name) => {
    const orig = s[name].bind(s);
    s[name] = (...args) => {
      const clean = args.map((a) => {
        if (a && typeof a === "object" && !Array.isArray(a)) {
          const c = Object.assign({}, a);
          delete c.cover;
          return c;
        }
        return a;
      });
      rec.ops.push({ op: name, args: clean });
      return orig(...args);
    };
  });
  return s;
};

const L = pres.ShapeType.line;
const ELL = pres.ShapeType.ellipse;
const RR = pres.ShapeType.roundRect;
const RECT = pres.ShapeType.rect;

let pageNo = 0;
const TOTAL = "14";

function cover(name) {
  return "image/png;base64," + fs.readFileSync(path.join(COVERS, name + ".png")).toString("base64");
}
const QA_IMAGES = !!process.env.QA_IMAGES; // QA build: identical geometry, poster image instead of embedded movie
function video(s, rel, x, y, w, h) {
  const base = path.basename(rel, ".mp4");
  if (QA_IMAGES) {
    s.addImage({ path: path.join(COVERS, base + ".png"), x, y, w, h });
  } else {
    s.addMedia({ type: "video", path: path.join(ASSETS, rel), x, y, w, h, cover: cover(base) });
  }
}
function title(s, text) {
  s.addText(text, { x: 0.5, y: 0.2, w: 12.33, h: 0.98, fontFace: FONT, fontSize: 26, bold: true, color: NAVY, valign: "top", margin: 0 });
}
function footer(s, label, src) {
  s.addText(label, { x: 0.4, y: 7.16, w: 2.6, h: 0.26, fontFace: FONT, fontSize: 9, color: GRAY, margin: 0 });
  s.addText(src, { x: 4.2, y: 7.16, w: 8.73, h: 0.26, fontFace: FONT, fontSize: 8, color: GRAY, align: "right", margin: 0 });
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ fontFace: FONT, color: INK, margin: 0, valign: "top" }, o));
}
function seg(s, x1, y1, x2, y2, o) {
  const opt = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: Object.assign({ color: NAVY, width: 2.5 }, o || {}) };
  if ((x2 - x1) * (y2 - y1) < 0) opt.flipV = true;
  s.addShape(L, opt);
}
function dot(s, cx, cy, r, color) {
  s.addShape(ELL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: color }, line: { color: "FFFFFF", width: 0.75 } });
}
function polyline(s, xs, ys, lineOpts, endDots, color) {
  for (let i = 0; i < xs.length - 1; i++) seg(s, xs[i], ys[i], xs[i + 1], ys[i + 1], lineOpts);
  if (endDots) [0, xs.length - 1].forEach((i) => dot(s, xs[i], ys[i], 0.05, color));
}
// table cell helpers
function C(text, o) { return { text: String(text), options: Object.assign({ fontFace: FONT, fontSize: 10, color: INK, valign: "middle" }, o || {}) }; }
function BU(text, o) { return C(text, Object.assign({ bold: true, underline: true }, o || {})); }
function HC(text, o) { return C(text, Object.assign({ bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 9.5, align: "center" }, o || {})); }

/* ============================================================ S1 | P1 封面 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Wan2.1-T2V 少步蒸馏:25× 加速下的质量退化诊断与归因");
  const vw = 5.39, vh = 3.11;
  video(s, "p1/p1_teacher_sportscar.mp4", 1.03, 1.12, vw, vh);
  video(s, "p1/p1_e1a_sportscar_s1.mp4", 6.91, 1.12, vw, vh);
  txt(s, "teacher:50 步 + CFG(≈100 NFE)\n单条 165.24s", { x: 1.03, y: 4.27, w: vw, h: 0.52, fontSize: 11, align: "center", color: GRAY });
  txt(s, "4-step 学生(E1a@1000,seed 1):无 CFG(4 NFE)\n单条 ≈6.6s", { x: 6.91, y: 4.27, w: vw, h: 0.52, fontSize: 11, align: "center", color: GRAY });
  s.addText([
    { text: "165.24s", options: { bold: true, color: NAVY } },
    { text: "  →  ", options: { color: GRAY } },
    { text: "6.59–6.66s", options: { bold: true, color: NAVY } },
    { text: "   ≈25×", options: { bold: true, color: ORANGE } },
  ], { x: 0.5, y: 4.92, w: 12.33, h: 0.68, fontFace: FONT, fontSize: 34, align: "center", margin: 0 });
  txt(s, "任务:将公开 Wan2.1-T2V-1.3B 的 50-step 教师模型以 DMD2 蒸馏为 4-step 学生;训练框架与全部单阶段超参承自 NVIDIA FastGen 公开配置,我们的工作在训练日程层(50→8→4 步数接力)与系统性的退化审计。", { x: 0.9, y: 5.72, w: 11.53, h: 0.62, fontSize: 13 });
  txt(s, "本报告回答的问题:加速 25 倍之后,生成质量损失了什么?损失来自哪个环节?", { x: 0.9, y: 6.4, w: 11.53, h: 0.4, fontSize: 14, bold: true, color: NAVY });
  footer(s, `1 / ${TOTAL}`, "出处:canonical 实验记录 W1 节(速度 2026-07-26 直读 metrics.csv 复核)");
  s.addNotes("讲稿(75s):先点击同时播放两条视频(同 prompt:红色跑车雨夜街景;约 5 秒即切出)。口头定义:扩散模型需多步去噪所以慢;蒸馏 = 用教师模型引导学生以极少步数生成。强调:画面乍看接近,本报告给出定量诊断。上游归属一句必须讲(配方承自 NVIDIA FastGen 公开配置,我们的工作在训练日程层与退化审计)。红线:方法名只说“步数接力”;速度数字已于 2026-07-26 复核(teacher 165.24 精确一致,student 全 sweep 6.591–6.656,≈25×)。");
}

/* ============================================================ S2 | P2 方法论 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "主观选点在两条训练线上均被量化推翻——全部结论建立在量化协议之上");
  // left: hand-drawn line chart (deterministic coordinates), W7 aes by iter
  txt(s, "接力臂 W7:aesthetic(q150,seed0)随训练迭代单调下降 0.5768 → 0.5379", { x: 0.55, y: 1.3, w: 7.1, h: 0.28, fontSize: 12, bold: true });
  // orange annotation: quantitative optimum bracket over first two points
  txt(s, "量化最优:第 500–1000 步", { x: 0.9, y: 1.6, w: 3.0, h: 0.26, fontSize: 10.5, bold: true, color: ORANGE });
  const px = [1.35, 2.75, 4.15, 5.55, 6.95];
  const vals = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379];
  const y0 = 4.75, yTop = 1.98, f = 2.8 / 0.055; // range 0.530-0.585
  const py = vals.map((v) => y0 - (v - 0.53) * f);
  seg(s, 1.35, y0, 6.95, y0, { color: "AAAAAA", width: 1 });
  seg(s, 1.35, yTop, 1.35, y0, { color: "AAAAAA", width: 1 });
  [0.54, 0.56, 0.58].forEach((t) => {
    const yy = y0 - (t - 0.53) * f;
    txt(s, t.toFixed(2), { x: 0.72, y: yy - 0.09, w: 0.55, h: 0.18, fontSize: 8.5, color: GRAY, align: "right" });
    seg(s, 1.29, yy, 1.35, yy, { color: "AAAAAA", width: 1 });
  });
  ["500", "1000", "1500", "2000", "2500"].forEach((t, k) => txt(s, t, { x: px[k] - 0.35, y: y0 + 0.05, w: 0.7, h: 0.18, fontSize: 9, color: GRAY, align: "center" }));
  txt(s, "训练迭代", { x: 1.35, y: 5.03, w: 5.6, h: 0.2, fontSize: 9.5, color: GRAY, align: "center" });
  txt(s, "纵轴范围 0.530–0.585(非零起点)", { x: 4.5, y: 1.62, w: 2.45, h: 0.2, fontSize: 8, color: GRAY, align: "right" });
  // orange bracket over points 0-1
  seg(s, 1.35, 1.9, 2.75, 1.9, { color: ORANGE, width: 1.5 });
  seg(s, 1.35, 1.9, 1.35, 1.98, { color: ORANGE, width: 1.5 });
  seg(s, 2.75, 1.9, 2.75, 1.98, { color: ORANGE, width: 1.5 });
  polyline(s, px, py, { color: NAVY, width: 2.75 }, false, NAVY);
  px.forEach((x, i) => dot(s, x, py[i], 0.05, NAVY));
  txt(s, "0.5768", { x: 1.48, y: py[0] - 0.27, w: 0.75, h: 0.2, fontSize: 9, bold: true, color: NAVY });
  txt(s, "0.5379", { x: 6.3, y: py[4] + 0.09, w: 0.75, h: 0.2, fontSize: 9, bold: true, color: NAVY });
  txt(s, "主观选择:第 2500 步(下降段末端)", { x: 4.35, y: 3.32, w: 2.5, h: 0.42, fontSize: 10.5, color: INK });
  seg(s, 6.45, 3.76, 6.9, py[4] - 0.08, { color: "AAAAAA", width: 1 });
  // right: W4 counter-example
  txt(s, "反例:均匀 t_list 消融(W4)——协议的阳性对照", { x: 7.7, y: 1.3, w: 5.13, h: 0.28, fontSize: 12, bold: true });
  s.addTable([
    [HC("指标"), HC("W4 值"), HC("全表位置")],
    [C("subject / background 一致性", { fontSize: 9.5 }), C("0.9745 / 0.9791", { align: "center" }), C("全表最高", { align: "center", bold: true })],
    [C("成像质量 imaging", { fontSize: 9.5 }), C("0.2555", { align: "center" }), C("全表最低", { align: "center", bold: true })],
    [C("跨 seed 多样性", { fontSize: 9.5 }), C("0.4617", { align: "center" }), C("全表最低", { align: "center", bold: true })],
  ], { x: 7.7, y: 1.68, w: 5.13, colW: [2.33, 1.4, 1.4], rowH: [0.34, 0.4, 0.4, 0.4], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  txt(s, "仅看一致性类指标,会将一个已坍缩为近静态输出的模型误判为最优。", { x: 7.7, y: 3.42, w: 5.13, h: 0.6, fontSize: 11.5, bold: true, color: NAVY });
  txt(s, "由此立联读规则:一致性类指标必须与动态度、多样性联读。", { x: 7.7, y: 4.1, w: 5.13, h: 0.4, fontSize: 9.5, color: GRAY });
  // bottom band
  s.addShape(RR, { x: 0.5, y: 5.62, w: 12.33, h: 1.08, rectRadius: 0.06, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "由此建立四组件量化协议:质量六维(q150)/ 清洁动态度(dm40)/ 跨 seed 多样性(d40×8)/ 连续光流(RAFT)——细节见备页 B1。\n选点规则:每 500 迭代存档、全量扫描取优(best-of-sweep)+ 冠军档换 seed 复检(n=3)。", { x: 0.78, y: 5.78, w: 11.8, h: 0.85, fontSize: 12, lineSpacingMultiple: 1.25 });
  footer(s, `2 / ${TOTAL}`, "出处:experiments/results/2026-07-14-e0-full-table-g1.md");
  s.addNotes("讲稿(60s):两条训练线的主观选点方向相反地出错——W7 主观选第 2500 步(实为下降段末端),W1 主观选第 1000 步(量化 @1500 在 5/6 质量维反超)→ 这是协议问题而非单次失误。W4(均匀 t_list)说明常用指标存在系统性盲区。伏笔(必讲):“质量在训练早期见顶后回落——这一现象的机制在第 5 页回答。”红线:不主张 t_list 消融首创(TMD 已有);一致性指标必须与动态/多样性联读。");
}

/* ============================================================ S3 | P3 核心发现 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "少步蒸馏的主要退化轴是跨 seed 多样性(约 −15~20%);动态度并未下降");
  txt(s, "同一 prompt(“An astronaut walking slowly across a dusty red desert…”,域内 eval 集):仅随机种子不同;缩略图为第 40 帧,可点击播放", { x: 0.5, y: 1.13, w: 12.33, h: 0.24, fontSize: 9.5, color: GRAY });
  const cw = 1.385, ch = 0.799, gap = 0.028, x0 = 1.52;
  const tSeeds = [0, 1, 2, 3, 4, 5, 6, 7];
  const eSeeds = [0, 2, 3, 4, 6, 8, 9, 11];
  txt(s, "teacher\n50 步", { x: 0.42, y: 1.62, w: 1.02, h: 0.5, fontSize: 10, bold: true, align: "center" });
  txt(s, "165.24s/条", { x: 0.42, y: 2.1, w: 1.02, h: 0.16, fontSize: 7.5, color: GRAY, align: "center" });
  txt(s, "学生 E1a\n4 步", { x: 0.42, y: 2.65, w: 1.02, h: 0.5, fontSize: 10, bold: true, align: "center" });
  txt(s, "≈6.6s/条", { x: 0.42, y: 3.13, w: 1.02, h: 0.16, fontSize: 7.5, color: GRAY, align: "center" });
  tSeeds.forEach((sd, k) => {
    const x = x0 + k * (cw + gap);
    video(s, `p3/p3v2_teacher_s${sd}.mp4`, x, 1.44, cw, ch);
    txt(s, `s${sd}`, { x, y: 2.25, w: cw, h: 0.15, fontSize: 8, color: GRAY, align: "center" });
  });
  eSeeds.forEach((sd, k) => {
    const x = x0 + k * (cw + gap);
    video(s, `p3/p3v2_e1a_s${sd}.mp4`, x, 2.47, cw, ch);
    txt(s, `s${sd}`, { x, y: 3.28, w: cw, h: 0.15, fontSize: 8, color: GRAY, align: "center" });
  });
  // bottom double panel (hand-drawn horizontal bars, shared 0-1 scale)
  const barX = 2.0, barW = 4.1, barX2 = 8.38;
  function hbar(x, y, len, color) { s.addShape(RECT, { x, y, w: Math.max(len, 0.02), h: 0.36, fill: { color }, line: { type: "none" } }); }
  // panel A
  txt(s, "动态度 DD_clean(dm40 清洁集):未下降", { x: 0.5, y: 3.62, w: 5.95, h: 0.26, fontSize: 12, bold: true });
  txt(s, "teacher", { x: 0.55, y: 4.02, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  txt(s, "学生范围", { x: 0.55, y: 4.62, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  seg(s, barX, 3.96, barX, 5.24, { color: "AAAAAA", width: 1 });
  hbar(barX, 4.0, 0.625 * barW, NAVY);
  txt(s, "0.625", { x: barX + 0.625 * barW + 0.06, y: 4.06, w: 0.7, h: 0.2, fontSize: 9.5, bold: true, color: NAVY });
  hbar(barX + 0.75 * barW, 4.6, 0.25 * barW, NAVY_SOFT);
  txt(s, "0.75–1.0(全体学生冠军档)", { x: barX + 0.06, y: 4.66, w: 2.9, h: 0.2, fontSize: 9.5, color: GRAY });
  [0, 0.5, 1.0].forEach((t) => txt(s, String(t), { x: barX + t * barW - 0.2, y: 5.26, w: 0.4, h: 0.16, fontSize: 8, color: GRAY, align: "center" }));
  txt(s, "运动平滑 0.97+:高动态为真实运动而非抖动;文献担忧的动态坍缩未复现(实验域不同)", { x: 0.5, y: 5.5, w: 5.95, h: 0.4, fontSize: 9.5, color: GRAY });
  // panel B (orange highlight)
  txt(s, "跨 seed 多样性(LPIPS,d40×8):一致下降——主退化轴", { x: 6.88, y: 3.62, w: 5.95, h: 0.26, fontSize: 12, bold: true, color: ORANGE });
  txt(s, "teacher", { x: 6.93, y: 4.02, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  txt(s, "学生范围", { x: 6.93, y: 4.62, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  seg(s, barX2, 3.96, barX2, 5.24, { color: "AAAAAA", width: 1 });
  hbar(barX2, 4.0, 0.732 * barW, NAVY);
  txt(s, "0.732", { x: barX2 + 0.732 * barW + 0.06, y: 4.06, w: 0.7, h: 0.2, fontSize: 9.5, bold: true, color: NAVY });
  hbar(barX2 + 0.59 * barW, 4.6, 0.05 * barW, ORANGE);
  txt(s, "0.59–0.64(全体学生冠军档)", { x: barX2 + 0.06, y: 4.66, w: 2.3, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE });
  [0, 0.5, 1.0].forEach((t) => txt(s, String(t), { x: barX2 + t * barW - 0.2, y: 5.26, w: 0.4, h: 0.16, fontSize: 8, color: GRAY, align: "center" }));
  txt(s, "相对 teacher 约 −15~20%;弱/强配方、直蒸/接力全部同向——本项目复现次数最多的发现", { x: 6.88, y: 5.5, w: 5.95, h: 0.4, fontSize: 9.5, color: GRAY });
  txt(s, "多样性定义:同一 prompt、8 个随机种子生成的视频之间的平均感知距离(LPIPS,越高越多样)。上排/下排即其直观对照:teacher 各 seed 构图彼此不同,学生趋同于同一模板。", { x: 0.5, y: 6.05, w: 12.33, h: 0.55, fontSize: 10.5, color: INK });
  footer(s, `3 / ${TOTAL}`, "出处:experiments/results/2026-07-14-e0-full-table-g1.md、2026-07-20-g2-relay-vs-direct-final.md");
  s.addNotes("讲稿(75s):文献普遍担忧少步蒸馏损失动态;我们的量化结果是“动态未降、运动平滑 0.97+ 排除抖动伪动态”(与文献实验域不同,表述为“未复现”而非反驳)。真正一致下降的是多样性——弱配方、强配方、直蒸、接力全部同向,是本项目复现次数最多的发现。多样性定义一句(见页面底部)。视频墙:上排 teacher 8 seed 内容彼此明显不同,下排学生 8 seed 构图明显趋同(学生排为 seed 0/2/3/4/6/8/9/11,如实标注)。过渡句(固定):“该退化在所有配置中出现——下面通过两组受控消融定位其来源。”");
}

/* ============================================================ S4 | P4 消融一 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "匹配预算的受控对照:步数接力无质量收益、多样性更低;其唯一稳定效应是更高的运动幅值");
  s.addTable([
    [HC("模型(best-of-sweep)", { align: "left" }), HC("美学 aes"), HC("成像 imaging"), HC("动态度 DD_clean"), HC("多样性 div")],
    [C("teacher 50-step CFG5", { fontSize: 10 }), BU("0.590", { align: "center" }), C("0.692", { align: "center" }), C("0.625", { align: "center" }), BU("0.732", { align: "center" })],
    [C("直蒸 E1a @1000(G2 加冕)", { fontSize: 10 }), C("0.567", { align: "center" }), BU("0.717", { align: "center" }), C("0.750", { align: "center" }), C("0.635", { align: "center" })],
    [C("接力 W7 @500", { fontSize: 10 }), C("0.577", { align: "center" }), C("0.694", { align: "center" }), C("0.825", { align: "center" }), C("0.598", { align: "center" })],
    [C("接力 W7 @1000", { fontSize: 10 }), C("0.559", { align: "center" }), C("0.697", { align: "center" }), BU("1.000", { align: "center" }), C("0.613", { align: "center" })],
    [C("直蒸 E1b @500", { fontSize: 10 }), C("0.532", { align: "center" }), C("0.695", { align: "center" }), C("0.975", { align: "center" }), C("0.628", { align: "center" })],
  ], { x: 0.5, y: 1.55, w: 7.35, colW: [2.5, 1.1, 1.15, 1.35, 1.25], rowH: [0.42, 0.4, 0.4, 0.4, 0.4, 0.4], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  txt(s, "协议:美学/成像 = q150 六维(seed0);动态度 = dm40 清洁集;多样性 = d40×8 LPIPS(三个域不混引);各列最优加粗下划线", { x: 0.5, y: 4.06, w: 7.35, h: 0.24, fontSize: 8.5, color: GRAY });
  txt(s, "注:E1a 成像 0.717 高于 teacher 0.692,按锐度/静态偏置口径解读,须与动态度、多样性联读;E1a 动态度 0.75 低于其余学生臂(0.825–1.0),如实标注。", { x: 0.5, y: 4.34, w: 7.35, h: 0.46, fontSize: 9.5, color: GRAY });
  // videos bottom-left
  txt(s, "同 prompt·同 seed 对照(cyclist):运动幅值差异直观呈现", { x: 0.6, y: 4.9, w: 6.1, h: 0.2, fontSize: 8.5, color: GRAY });
  video(s, "p4/p4_w7_cyclist.mp4", 0.6, 5.12, 2.78, 1.604);
  video(s, "p4/p4_e1a_cyclist.mp4", 3.72, 5.12, 2.78, 1.604);
  txt(s, "接力 W7@1000(4 步)", { x: 0.6, y: 6.74, w: 2.78, h: 0.2, fontSize: 9, align: "center", color: GRAY });
  txt(s, "直蒸 E1a@1000(4 步)", { x: 3.72, y: 6.74, w: 2.78, h: 0.2, fontSize: 9, align: "center", color: GRAY });
  // right-top design box
  s.addShape(RR, { x: 8.1, y: 1.55, w: 4.73, h: 2.62, rectRadius: 0.06, fill: { color: PANEL }, line: { color: NAVY, width: 1 } });
  txt(s, "受控设计(预注册)", { x: 8.3, y: 1.68, w: 4.35, h: 0.26, fontSize: 12, bold: true, color: NAVY });
  txt(s, "接力臂 50→8→4:W5(2500 迭代)+ W7(2500 迭代)\n直蒸臂 ×2 50→4:E1a(接力二段配方)/ E1b(接力一段配方,上游出厂 LR),各 5000 迭代\n不变量:总预算 · 数据 · 4-step t_list · 单阶段配方 · 评估协议;双臂 bracket 防“基线未调优”", { x: 8.3, y: 1.98, w: 4.35, h: 1.6, fontSize: 10, lineSpacingMultiple: 1.15 });
  txt(s, "预注册预期:两者相当 → 实测:直蒸略优", { x: 8.3, y: 3.72, w: 4.35, h: 0.3, fontSize: 10.5, bold: true, color: NAVY });
  // right-bottom flow bars
  txt(s, "运动幅值(RAFT 光流中位,dm40,4-seed 均值)", { x: 8.1, y: 4.42, w: 4.73, h: 0.24, fontSize: 11, bold: true });
  txt(s, "接力 W7", { x: 8.1, y: 4.78, w: 1.15, h: 0.3, fontSize: 9.5, align: "right" });
  txt(s, "直蒸 E1a", { x: 8.1, y: 5.28, w: 1.15, h: 0.3, fontSize: 9.5, align: "right" });
  seg(s, 9.35, 4.72, 9.35, 5.66, { color: "AAAAAA", width: 1 });
  s.addShape(RECT, { x: 9.35, y: 4.76, w: (3.36 / 4) * 3.0, h: 0.34, fill: { color: NAVY }, line: { type: "none" } });
  txt(s, "3.36", { x: 9.35 + (3.36 / 4) * 3.0 + 0.06, y: 4.82, w: 0.6, h: 0.2, fontSize: 9.5, bold: true, color: NAVY });
  s.addShape(RECT, { x: 9.35, y: 5.26, w: (1.81 / 4) * 3.0, h: 0.34, fill: { color: NAVY_SOFT }, line: { type: "none" } });
  txt(s, "1.81", { x: 9.35 + (1.81 / 4) * 3.0 + 0.06, y: 5.32, w: 0.6, h: 0.2, fontSize: 9.5, bold: true, color: INK });
  txt(s, "4/4 seed 方向一致,约 1.9×;评价取决于以贴近 teacher 分布还是以更大动态为准则——本文不选边(多 seed 全表见备页 B5)", { x: 8.1, y: 5.78, w: 4.73, h: 0.75, fontSize: 9, color: GRAY });
  footer(s, `4 / ${TOTAL}`, "出处:2026-07-20-g2-relay-vs-direct-final.md、thesis_ch2_draft.md 发现 1/5、2026-07-23-flow-multiseed-e1b946-e2a-eval.md");
  s.addNotes("讲稿(90s):接力是我们的原始设计,因此这项对照以预注册方式进行以避免偏向;结论如实:质量判平或直蒸略优(imaging E1a 0.717 为学生最高;aes n=3 带重叠判平)、多样性直蒸双臂一致更高(0.635/0.628 vs 0.598/0.613);运动幅值是接力唯一稳定的实测差异,其来源在下页消融中回答。红线:E1a 成像高于 teacher 只按锐度/静态偏置口径讲;E1a 动态度 0.75 偏低须如实提;单 seed 百分比禁引;不宣称直蒸路线首创。\n\n备问卡 #1「负结果为什么值得报告?」——两个负结果均为受控排除的进展:接力对照是该基座上首个受控的 relay-vs-direct 比较(GPD/CoDMD/FastWan 均未做),并给出接力唯一的真实效应(运动幅值,4/4 seed);配对消融首次检验了被广泛复制的默认设计。");
}

/* ============================================================ S5 | P5 消融二 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "单变量消融表明:GAN 分支驱动“质量随训练回落”与运动幅值;(t,ε) 配对约定未检测到影响");
  txt(s, "同一初始化(W5@2500)、同一配方,仅 GAN 设置不同:aesthetic(q150,seed0)随训练迭代", { x: 0.55, y: 1.5, w: 7.3, h: 0.27, fontSize: 11.5, bold: true });
  const px = [0, 1, 2, 3, 4].map((k) => 1.35 + k * 1.175);
  const yb = 5.35, f = 3.4 / 0.09; // range 0.53-0.62
  const Y = (v) => yb - (v - 0.53) * f;
  const e2a = [0.5908, 0.5921, 0.5984, 0.6109, 0.6074].map(Y);
  const e2b = [0.5774, 0.567, 0.5477, 0.5471, 0.5487].map(Y);
  const w7 = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379].map(Y);
  seg(s, 1.35, yb, 6.05, yb, { color: "AAAAAA", width: 1 });
  seg(s, 1.35, 1.95, 1.35, yb, { color: "AAAAAA", width: 1 });
  [0.54, 0.56, 0.58, 0.6, 0.62].forEach((t) => {
    txt(s, t.toFixed(2), { x: 0.72, y: Y(t) - 0.09, w: 0.55, h: 0.18, fontSize: 8.5, color: GRAY, align: "right" });
    seg(s, 1.29, Y(t), 1.35, Y(t), { color: "AAAAAA", width: 1 });
  });
  ["500", "1000", "1500", "2000", "2500"].forEach((t, k) => txt(s, t, { x: px[k] - 0.35, y: yb + 0.05, w: 0.7, h: 0.18, fontSize: 8.5, color: GRAY, align: "center" }));
  txt(s, "训练迭代", { x: 1.35, y: 5.6, w: 4.7, h: 0.2, fontSize: 9.5, color: GRAY, align: "center" });
  txt(s, "纵轴 0.53–0.62", { x: 1.42, y: 1.98, w: 1.4, h: 0.18, fontSize: 8, color: GRAY });
  polyline(s, px, w7, { color: NAVY, width: 2.25, dashType: "dash" }, true, NAVY);
  polyline(s, px, e2b, { color: NAVY, width: 2.25 }, true, NAVY);
  polyline(s, px, e2a, { color: ORANGE, width: 2.75 }, true, ORANGE);
  dot(s, px[3], e2a[3], 0.05, ORANGE);
  txt(s, "0.5908", { x: 1.44, y: e2a[0] - 0.30, w: 0.7, h: 0.18, fontSize: 8.5, bold: true, color: ORANGE });
  txt(s, "≈0.577(两臂同起点)", { x: 1.45, y: 3.72, w: 1.8, h: 0.18, fontSize: 8.5, color: GRAY });
  txt(s, "0.6109(峰)@2000", { x: 4.28, y: e2a[3] - 0.28, w: 1.5, h: 0.18, fontSize: 8.5, color: ORANGE });
  txt(s, "E2a:GAN 关闭\n终值 0.6074", { x: 6.15, y: e2a[4] - 0.2, w: 1.7, h: 0.42, fontSize: 10, bold: true, color: ORANGE });
  txt(s, "E2b:GAN 开启(独立 t,ε)终值 0.5487", { x: 6.15, y: e2b[4] - 0.34, w: 1.72, h: 0.55, fontSize: 9, color: NAVY });
  txt(s, "W7:GAN 开启(配对)终值 0.5379", { x: 6.15, y: w7[4] - 0.02, w: 1.72, h: 0.55, fontSize: 9, color: NAVY });
  // right column: three conclusion boxes
  const box = (y, head, body, headColor) => {
    s.addShape(RR, { x: 7.95, y, w: 4.9, h: 1.5, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
    txt(s, head, { x: 8.12, y: y + 0.08, w: 4.6, h: 0.26, fontSize: 11, bold: true, color: headColor || NAVY });
    txt(s, body, { x: 8.12, y: y + 0.38, w: 4.6, h: 1.05, fontSize: 9.5, lineSpacingMultiple: 1.1 });
  };
  box(1.5, "① 质量“早峰后滑”← GAN 分支(候选机制)", "关闭 GAN(E2a):质量随训练一路向好,五档同向;开启 GAN(E2b/W7):随训练回落——现象在单变量对照中消失并反向。冠军档 n=3 逐 seed 配对 3/3 同向。", ORANGE);
  box(3.12, "② 运动幅值主要由 GAN 分支维持", "关闭 GAN:回落至 teacher 水平(2.1–2.7);开启 GAN:自低点重建至 4.71。在匹配配方内与接力初始化存在交互(直蒸臂 E1a 开启 GAN 仍低)。");
  box(4.74, "③ (t,ε) 配对约定:未检测到效应", "real/fake 共享 (t,ε) 为上游默认设计,据我们所知此前无受控检验;五个对照档质量差均 ≤0.011,各维走势同构。");
  s.addShape(RR, { x: 0.5, y: 6.42, w: 12.33, h: 0.5, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "三条臂的跨 seed 多样性均落在 0.586–0.613——对 GAN 开关与配对方式均不敏感(下一页收束)。", { x: 0.78, y: 6.52, w: 11.8, h: 0.3, fontSize: 11.5, bold: true });
  footer(s, `5 / ${TOTAL}`, "出处:2026-07-24-e2a-fulltable-ch3.md、2026-07-25-e2b-fulltable-ch3-threearm.md");
  s.addNotes("讲稿(105s):先一句交代消融设计的干净性——在接力臂 W7 配方上每次只改一个字段(E2a 仅 gan_loss_weight_gen 0.03→0;E2b 仅 gan_use_same_t_noise True→False),配置逐值核对。中心图讲 20 秒(同一初始化、同一配方,仅 GAN 设置不同);三个结论各 15 秒;强调第③条的价值:被广泛复制的默认设计首次得到受控检验。红线:“候选机制”措辞(单权重点 0.03、单血统);禁“GAN 必然抬高运动”(直蒸臂 E1a 开启 GAN 仍低);E2a 美学 0.613 高于 teacher 0.590 不作对比表述;配对结论说“未检测到效应”而非“无差异”。\n\n备问卡 #3「关闭 GAN 后画质更高,为何不建议关闭?」——单权重点(0.03)、单血统(接力)结论;且关闭后运动幅值回落至 teacher 水平——是取舍不是免费改进;本工作给出的是归因而非配方建议。");
}

/* ============================================================ S6 | P6 归因收束 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "逐部件受控排除后,多样性坍缩仍然存在——在受测部件内,归因于分布匹配蒸馏本身");
  const comp = (y, name, ev) => {
    s.addShape(RR, { x: 0.6, y, w: 4.9, h: 1.08, rectRadius: 0.06, fill: { color: PANEL }, line: { color: NAVY, width: 1 } });
    txt(s, name, { x: 0.82, y: y + 0.12, w: 3.1, h: 0.28, fontSize: 12.5, bold: true, color: NAVY });
    s.addShape(RR, { x: 3.95, y: y + 0.12, w: 1.4, h: 0.3, rectRadius: 0.12, fill: { color: LIGHT }, line: { color: GRAY, width: 0.75 } });
    txt(s, "已受控排除", { x: 3.95, y: y + 0.16, w: 1.4, h: 0.22, fontSize: 8.5, color: INK, align: "center" });
    txt(s, ev, { x: 0.82, y: y + 0.48, w: 4.5, h: 0.55, fontSize: 9.5, color: GRAY });
    seg(s, 5.5, y + 0.54, 6.15, y + 0.54, { color: GRAY, width: 1.25, endArrowType: "triangle" });
  };
  comp(1.7, "步数接力(50→8→4)", "证据:P4——接力臂多样性反而更低(0.598/0.613 vs 直蒸 0.628/0.635)");
  comp(3.15, "GAN 判别器分支", "证据:P5——开关不改变坍缩量级(三臂 0.586–0.613)");
  comp(4.6, "(t,ε) 配对约定", "证据:P5——五个对照档均未检测到效应");
  txt(s, "受控排除后仍然存在:跨 seed 多样性坍缩", { x: 6.4, y: 1.75, w: 6.4, h: 0.3, fontSize: 12.5, bold: true });
  const bx = 7.75, bw = 4.2; // scale 0-0.8
  txt(s, "teacher", { x: 6.35, y: 2.56, w: 1.3, h: 0.3, fontSize: 10, align: "right" });
  txt(s, "全体学生", { x: 6.35, y: 3.36, w: 1.3, h: 0.3, fontSize: 10, align: "right" });
  seg(s, bx, 2.45, bx, 3.95, { color: "AAAAAA", width: 1 });
  s.addShape(RECT, { x: bx, y: 2.52, w: (0.732 / 0.8) * bw, h: 0.42, fill: { color: NAVY }, line: { type: "none" } });
  txt(s, "0.732", { x: bx + (0.732 / 0.8) * bw + 0.07, y: 2.62, w: 0.7, h: 0.2, fontSize: 10, bold: true, color: NAVY });
  s.addShape(RECT, { x: bx + (0.586 / 0.8) * bw, y: 3.32, w: (0.063 / 0.8) * bw, h: 0.42, fill: { color: ORANGE }, line: { type: "none" } });
  txt(s, "0.586–0.649", { x: bx + (0.586 / 0.8) * bw - 0.35, y: 3.02, w: 1.3, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE });
  [0, 0.4, 0.8].forEach((t) => txt(s, t.toFixed(1), { x: bx + (t / 0.8) * bw - 0.2, y: 3.98, w: 0.4, h: 0.16, fontSize: 8, color: GRAY, align: "center" }));
  txt(s, "跨 seed 多样性(LPIPS,d40×8;全体学生 = 各臂冠军档与弱配方参考档)", { x: 6.4, y: 4.22, w: 6.4, h: 0.22, fontSize: 8.5, color: GRAY });
  txt(s, "在受测部件内,坍缩归因于分布匹配蒸馏本身。未测部件(如实列):数据构成、teacher CFG 蒸馏方式、其它 GAN 权重档等。", { x: 6.4, y: 4.55, w: 6.4, h: 0.75, fontSize: 10.5 });
  s.addShape(RR, { x: 1.2, y: 5.85, w: 10.93, h: 0.9, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1.5 } });
  txt(s, "本工作不提供该问题的解法;提供的是一个经过测量、并对来源做过受控排除的精确问题定义(开放问题)。", { x: 1.5, y: 6.08, w: 10.3, h: 0.45, fontSize: 13.5, bold: true, color: NAVY, align: "center" });
  footer(s, `6 / ${TOTAL}`, "出处:2026-07-25-e2b-fulltable-ch3-threearm.md(裁读 4)");
  s.addNotes("讲稿(45s):30 秒讲排除逻辑——换训练路线(接力→直蒸)坍缩在且接力更深;关掉 GAN 分支坍缩在;换 (t,ε) 配对坍缩在。15 秒讲开放问题定位:这是本报告希望留给听众的核心信息。红线:“归因于蒸馏本身”必须限定“在受测部件内”(未测:数据构成、teacher CFG 蒸馏方式等);不写“不可解”。");
}

/* ============================================================ S7 | P7 VBench */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "标准基准上不存在支配模型:不同模型在不同维度领先;多样性上界仍属 teacher");
  const span = (t, fill) => [{ text: t, options: { colspan: 5, fill: { color: fill }, bold: true, fontSize: 8.5, color: INK, align: "left", fontFace: FONT, valign: "middle" } }];
  const R = (name, v, best) => [C(name, { fontSize: 9 })].concat(v.map((x, i) => (i === best ? BU(x, { align: "center", fontSize: 9.5 }) : C(x, { align: "center", fontSize: 9.5 }))));
  s.addTable([
    [HC("维度", { align: "left" }), HC("E1a@1000 G2 加冕·主对照"), HC("W7@1000 接力"), HC("E1b@500 直蒸B"), HC("E2a@2000 审计臂 GAN=0·仅单档")],
    span("质量维(7)——恰为官方 Quality Score 全部质量维,可按官方权重合成(备页 B4)", QBLOCK),
    R("subject_consistency(主体一致)", ["0.9727", "0.9693", "0.9673", "0.9753"], 3),
    R("background_consistency(背景一致)", ["0.9579", "0.9508", "0.9416", "0.9581"], 3),
    R("motion_smoothness(运动平滑)", ["0.9812", "0.9727", "0.9747", "0.9786"], 0),
    R("dynamic_degree(动态度)", ["0.5806", "0.9111", "0.8806", "0.8000"], 1),
    R("aesthetic_quality(美学)", ["0.5967", "0.6087", "0.5802", "0.6482"], 3),
    R("imaging_quality(成像)", ["0.6687", "0.6687", "0.6614", "0.6924"], 3),
    R("temporal_flickering(闪烁)", ["0.9894", "0.9796", "0.9810", "0.9878"], 0),
    span("语义类原始维(5)——非官方 Semantic(GRiT 4 维缺失,不可合成)", SBLOCK),
    R("human_action(动作)", ["0.690", "0.794", "0.716", "0.776"], 1),
    R("scene(场景)", ["0.2173", "0.2922", "0.2225", "0.2974"], 3),
    R("appearance_style(外观风格)", ["0.1990", "0.1982", "0.2003", "0.2010"], 3),
    R("temporal_style(时序风格)", ["0.2214", "0.2305", "0.2260", "0.2283"], 1),
    R("overall_consistency(总体一致)", ["0.2240", "0.2386", "0.2298", "0.2391"], 1),
  ], { x: 0.45, y: 1.5, w: 8.55, colW: [3.1, 1.4, 1.2, 1.2, 1.65], rowH: [0.5, 0.22].concat(Array(7).fill(0.31), [0.22], Array(5).fill(0.31)), border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  txt(s, "full VBench standard mode,946 prompts × 5 seeds,12/16 维;每行最优加粗下划线。缺失 4 维均为检测器依赖维(GRiT/detectron2),官方 Semantic 与 Total 不可合成,如实声明;temporal_flickering 官方协议为专属子集、25 样本/prompt、static_filter 预筛,本表 5 样本未筛,不与官方数字直比。", { x: 0.45, y: 6.28, w: 8.55, h: 0.62, fontSize: 8, color: GRAY });
  // right reading column
  txt(s, "读法(各有所长)", { x: 9.25, y: 1.5, w: 3.6, h: 0.28, fontSize: 12, bold: true, color: NAVY });
  const rb = (y, head, body) => {
    txt(s, head, { x: 9.25, y, w: 3.6, h: 0.24, fontSize: 10.5, bold: true });
    txt(s, body, { x: 9.25, y: y + 0.26, w: 3.6, h: 0.72, fontSize: 9.5, color: INK, lineSpacingMultiple: 1.08 });
  };
  rb(1.88, "E1a(主对照)", "一致性 / 平滑 / 闪烁类领先");
  rb(2.78, "W7(接力)", "动态度 0.911 与动作语义类(0.794 / 0.292)领先");
  rb(3.68, "E2a(审计臂)", "静态画质最高(0.6482 / 0.6924),动态 0.80 未塌——与消融结论跨域同向");
  rb(4.82, "teacher", "多样性 0.732(q150 域)仍为上界——该轴不在 VBench 维度内,正是自建协议的原因(备页 B4)");
  footer(s, `7 / ${TOTAL}`, "出处:experiments/results/2026-07-26-e2a-vb946-fourth-row.md");
  s.addNotes("讲稿(60s):四模型各有所长——E1a 赢一致性/平滑/闪烁类,W7 赢动态度与动作语义类,E1b 居中,E2a 静态画质最高且动态不塌(与 q150 域消融结论跨域同向)。多样性不在 VBench 维度内(自建协议的原因)。红线:最终模型选择遵循 G2 预注册结果(E1a@1000),非事后挑选;合成分与文献数字巧合仅在备页 B4/备问卡处理,不上正片;两域动态度数字(q150 0.567 / vb946 0.800)不混引。\n\n备问卡 #2「与 CoDMD 的 84.46 相比如何?」——协议不可比(我们 12/16 维、5 样本 flickering、无 GRiT 维),该数字仅作文献坐标,不做 SOTA 对比;我方 W7 的 Quality Score 84.47 与其为数字巧合(7 维质量合成 vs 16 维总分),不并列。\n备问卡 #6「主对照 E1a 的合成分为何最低(82.80)?」——合成分权重结构:动态度以 0.5 权重计入,E1a 动态偏低(0.5806)拖低总分,其领先的一致性/闪烁维归一化后区分度小;且合成分不含多样性项,无法反映主退化轴。模型选择依据是预注册协议,合成分仅作参考。\n备问卡 #7「E2a 合成分最高(85.50),为何不改选它?」——①预注册纪律:最终模型在 E2a 存在之前已按协议确定,事后更换即选择性报告;②E2a 多样性为全场最低带(0.586–0.604)、动态回落至 teacher 水平——合成分恰好测不到这两点;③单次训练、单权重点。");
}

/* ============================================================ S8 | P8 结论 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "结论:25× 加速可行;主退化被定位为多样性坍缩;给出三条可操作建议");
  const col = (x, head, items, fs) => {
    txt(s, head, { x, y: 1.45, w: 4.05, h: 0.32, fontSize: 14, bold: true, color: NAVY });
    s.addText(items.map((t, i) => ({ text: t, options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 11 } })), { x, y: 1.85, w: 4.05, h: 4.6, fontFace: FONT, fontSize: fs || 12, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.18 });
  };
  col(0.5, "主要结论", [
    "诊断:主退化轴 = 跨 seed 多样性坍缩(teacher 0.732 → 学生 0.59–0.64),而非动态度",
    "归因:质量回落与运动幅值主要与 GAN 分支相关;(t,ε) 配对约定未检测到影响;多样性坍缩源自蒸馏本身(开放问题)",
    "方法:轻量、seed 受控的退化审计协议(含主观选点不可靠的证据)",
    "工程:25× 加速;该基座上首个受控的接力-直蒸对照(据我们所知,含检索覆盖)",
  ]);
  col(4.78, "实践建议", [
    "每 500 迭代存档、全量扫描选点:质量峰值通常出现在训练早期,勿依赖主观观察或默认取末档",
    "监控跨 seed 多样性与连续光流——常规指标与合成分不会对主退化轴报警",
    "GAN 权重是“静态画质 ↔ 运动幅值”的取舍参数;(t,ε) 配对约定无需调整(附注:结论来自单权重点、单血统,外推需谨慎)",
  ]);
  col(9.06, "局限与后续", [
    "无人工评测;标准基准缺 4 维",
    "R1 正则臂受 32G 显存限制未测(校准值留档,80G 级设备可复现)",
    "各配置均为单次训练(同族臂方向互证缓解);接力源选点早于选点政策确立",
    "后续:人评 · 跨血统消融 · 80G 复现 · 多样性坍缩的干预研究",
  ], 11.5);
  txt(s, "谢谢!欢迎指正——论文将于 2026-07-31 提交。", { x: 0.5, y: 6.62, w: 12.33, h: 0.35, fontSize: 12.5, bold: true, color: NAVY, align: "center" });
  footer(s, `8 / ${TOTAL}`, "出处:T3_novelty_adjudication.md §4.1、2026-07-25-e2b-fulltable-ch3-threearm.md、acceptance-log.md #11–#13");
  s.addNotes("讲稿(60s):三栏各 20 秒。红线:“首个”带“据我们所知 + 检索覆盖”限定;建议③必须带 caveat(单权重点、单血统)。\n\n备问卡 #4「为什么没有人工评测?」——时间窗内优先完成受控消融;人评已列入局限与后续工作(计划 T2VHE 式对 teacher 评测)。\n备问卡 #5「R1 正则臂为什么未测?」——确定性 OOM(32G 单卡,崩溃点为 R1 的第二次判别器前向),非配方失败;校准值与配置已留档,80G 级设备可直接复现。");
}

/* ============================================================ S9 | B1 协议 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "备页 B1|评估协议四组件与主表协议");
  const blk = (x, y, head, body) => {
    s.addShape(RR, { x, y, w: 5.99, h: 1.78, rectRadius: 0.05, fill: { color: PANEL }, line: { color: BORDER, width: 0.75 } });
    txt(s, head, { x: x + 0.18, y: y + 0.1, w: 5.6, h: 0.26, fontSize: 12, bold: true, color: NAVY });
    txt(s, body, { x: x + 0.18, y: y + 0.42, w: 5.66, h: 1.3, fontSize: 10, lineSpacingMultiple: 1.12 });
  };
  blk(0.5, 1.35, "① q150 质量六维", "VBench 官方 all_dimension 确定性抽样 150 条(md5 690f2919),custom-input 模式;sweep 为 seed0,冠军档 n=3(seed 0/1/2)换 seed 复检。");
  blk(6.84, 1.35, "② dm40 清洁动态度(可引用 DD)", "40 条 motion 导向 prompt(20 条官方 human_action.txt uniform stride + 20 条 all_dimension.txt 经 MOTION_CUE 正则过滤并排除 STATIC_BLOCK;md5 324d75a0)。q150-DD 受静态指令 prompt 混淆仅作脚注(teacher 两域 DD:0.300 vs 0.625)。");
  blk(0.5, 3.33, "③ d40×8 跨 seed 多样性", "40 prompts × 8 seeds,平均成对 LPIPS-alex(8 帧 @256px;md5 b4c1f9e3;越高越多样)——本项目的主退化轴度量。");
  blk(6.84, 3.33, "④ RAFT 连续光流(运动幅值)", "dm40 域,像素/帧;中位数为主读、均值并报(teacher 重尾:中位 2.75 / 均值 5.16)。多 seed 纪律:臂间方向按逐 seed 配对报告,单 seed 绝对百分比不单独引用。动机:二值 DD 对好学生饱和(0.75–1.0),无分辨率。");
  s.addShape(RR, { x: 0.5, y: 5.35, w: 12.33, h: 1.4, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "主表协议:full VBench standard mode(946 prompts × 5 seeds),12/16 维(GRiT 4 维缺失,Semantic/Total 不可合成,已声明);temporal_flickering 与官方协议差异脚注。\n通用规则:q150 / dm40 / vb946 三域数字不跨表混引;训练健康指标(loss)不是质量证据;每实验只改一个变量;checkpoint 一律 best-of-sweep。", { x: 0.78, y: 5.5, w: 11.8, h: 1.15, fontSize: 10.5, lineSpacingMultiple: 1.2 });
  footer(s, `9 / ${TOTAL}(备页 B1)`, "出处:research/thesis_ch1_draft.md §1.7");
  s.addNotes("被问评估协议细节时展开。md5 与抽样准则均已入库(exp/eval/,make_motion_set.py 头部;2026-07-26 与远端逐要点核实)。");
}

/* ============================================================ S10 | B2 E5 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "备页 B2|E5 层×t 可分性探针(观察性证据,非主结论)");
  s.addTable([
    [HC("噪声水平 t"), HC("real vs 生成 线性探针 AUC(全 9 层)")],
    [C("t = 0.999(最高噪)", { align: "center" }), C("0.28–0.52(随机及以下)", { align: "center" })],
    [C("t ≤ 0.937(其余全部档)", { align: "center" }), C("1.0(全层饱和,n=64)", { align: "center", bold: true })],
  ], { x: 0.7, y: 1.5, w: 7.0, colW: [2.8, 4.2], rowH: [0.4, 0.42, 0.42], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  s.addText([
    { text: "悬崖式 t 依赖:", options: { bold: true } },
    { text: "判别器的可用监督信号集中于中低噪声端,最高噪声档几乎无区分度——为 (t,ε) 配对消融“未检测到效应”提供机制背景(配对影响的高噪端本就是信号最弱区间)。", options: {} },
  ], { x: 0.7, y: 3.0, w: 12.0, h: 0.55, fontFace: FONT, fontSize: 11, color: INK, margin: 0 });
  s.addText([
    { text: "层选择的公允结论:", options: { bold: true } },
    { text: "AUC 在 t ≤ 0.937 全层饱和、无层间分辨率(早期“均值 AUC 0.88–0.92 / L7 略优”读法系饱和平均假象,已弃用);连续的 Fréchet 距离随深度平缓上升、在 L27/29 陡增。对上游判别器选层 {15, 22, 29}:覆盖中深两段,无证据是坏选择,也无证据是唯一最优。", options: {} },
  ], { x: 0.7, y: 3.7, w: 12.0, h: 0.85, fontFace: FONT, fontSize: 11, color: INK, margin: 0 });
  s.addText([
    { text: "诚实性对照(teachergen):", options: { bold: true } },
    { text: "teacher 自身 50-step 生成物与真实数据的可分性同构(AUC 同为悬崖形,FD 反而更大)→ 特征可分性主要反映“生成域 vs 真实域”共性与 prompt 域差,不能作为蒸馏退化的直接度量。", options: {} },
  ], { x: 0.7, y: 4.7, w: 12.0, h: 0.75, fontFace: FONT, fontSize: 11, color: INK, margin: 0 });
  txt(s, "协议:64 clip/侧;特征路径逐字段对齐训练侧判别器;null-text 统一条件;5-fold 线性探针 AUC + Fréchet 距离。定位:to-our-knowledge 级观察性证据。", { x: 0.7, y: 5.7, w: 12.0, h: 0.5, fontSize: 9.5, color: GRAY });
  footer(s, `10 / ${TOTAL}(备页 B2)`, "出处:research/E5_probe_results.md(修正口径,权威)");
  s.addNotes("被问判别器机制/E5 时展开。效度框:n=64/侧、单批次、null-text、AUC 饱和处无分辨率、FD 对样本量敏感。");
}

/* ============================================================ S11 | B3 G2 设计细节 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "备页 B3|接力-直蒸受控对照:设计细节");
  const row = (y, head, body) => {
    txt(s, head, { x: 0.7, y, w: 2.6, h: 0.6, fontSize: 12, bold: true, color: NAVY });
    txt(s, body, { x: 3.5, y, w: 9.3, h: 0.85, fontSize: 11, lineSpacingMultiple: 1.15 });
  };
  row(1.5, "接力臂", "W5(8-step,LR 1e-5 / batch 12,2500 迭代,teacher 起训)→ W7(4-step,LR 5e-6 / batch 16,2500 迭代;仅继承 W5@2500 生成器权重,优化器 / fake score / 判别器全部重置)。总预算 5000 迭代。");
  row(2.55, "直蒸双臂", "E1a = 接力二段配方(LR 5e-6 / batch 16);E1b = 接力一段配方(LR 1e-5 / batch 12,恰为上游 FastGen 出厂默认 LR);各 5000 迭代,teacher 起训。双臂 bracket 使“直蒸基线未调优”的质疑失效。");
  row(3.6, "不变量", "数据(OpenVid-1M)· 4-step t_list · 判别器构型 · 全部单阶段超参(上游公开配置值)· 评估协议 · checkpoint 粒度(每 500 迭代)。");
  row(4.55, "选点", "全臂 best-of-sweep(E1a/E1b 各 10 档、W7 5 档;32 行全表零缺格);冠军档 n=3 换 seed 复检。");
  s.addShape(RR, { x: 0.7, y: 5.55, w: 12.0, h: 0.95, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "预注册纪律:对照启动前登记领先假设为“两者相当”(中性);实测为直蒸略优——按预注册结果报告,不做事后调整。", { x: 0.95, y: 5.75, w: 11.5, h: 0.55, fontSize: 11.5, bold: true, color: NAVY });
  footer(s, `11 / ${TOTAL}(备页 B3)`, "出处:research/thesis_ch2_draft.md §2.1");
  s.addNotes("被问 G2 设计细节(为什么可信、防了哪些质疑)时展开。");
}

/* ============================================================ S12 | B4 QS */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "备页 B4|Quality Score 合成与其盲区");
  txt(s, "官方权重(Vchitect/VBench scripts/constant.py + cal_final_score.py @master):每维 min-max 归一化;dynamic_degree 权重 0.5,其余 6 维权重 1;加权和 ÷ 6.5。", { x: 0.6, y: 1.42, w: 12.1, h: 0.5, fontSize: 10.5, color: INK });
  s.addTable([
    [HC("模型"), HC("Quality Score(7 维合成)"), HC("备注")],
    [C("E2a@2000(审计臂 GAN=0,仅单档)", { fontSize: 10 }), C("85.50", { align: "center", bold: true }), C("多样性为全场最低带 0.586–0.604;动态回落 teacher 水平", { fontSize: 9 })],
    [C("W7@1000(接力)", { fontSize: 10 }), C("84.47", { align: "center" }), C("与 CoDMD 84.46 为数字巧合,禁止并列(见下)", { fontSize: 9 })],
    [C("E1b@500(直蒸B)", { fontSize: 10 }), C("83.62", { align: "center" }), C("高动态低美学", { fontSize: 9 })],
    [C("E1a@1000(G2 加冕·主对照)", { fontSize: 10 }), C("82.80", { align: "center" }), C("动态度 0.5806 以 0.5 权重拖低;优势维归一化后区分度小", { fontSize: 9 })],
  ], { x: 0.6, y: 2.0, w: 8.6, colW: [3.3, 2.1, 3.2], rowH: [0.38, 0.44, 0.44, 0.44, 0.44], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  s.addShape(RR, { x: 0.6, y: 4.45, w: 8.6, h: 1.15, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "盲区:合成分不含多样性项——无法反映本项目确认的主退化轴(低多样性的 E2a 排第一、主对照排末位,与 W4 教训同构,正是自建协议存在的理由)。最终模型选择依据为 G2 预注册协议,不受合成分影响。", { x: 0.85, y: 4.6, w: 8.1, h: 0.9, fontSize: 10.5, lineSpacingMultiple: 1.15 });
  s.addShape(RR, { x: 0.6, y: 5.85, w: 8.6, h: 0.85, rectRadius: 0.05, fill: { color: "FFFFFF" }, line: { color: ORANGE, width: 1.5 } });
  txt(s, "警示:W7 的 84.47 与 CoDMD 文献值 84.46 为数字巧合——前者为 7 维 Quality Score、后者为 16 维 Total,量纲不同,禁止任何并列或比较表述。", { x: 0.85, y: 6.0, w: 8.1, h: 0.6, fontSize: 10.5, bold: true, color: ORANGE });
  // demo videos for cards #3/#7
  txt(s, "演示素材(同 prompt·同 seed,cyclist)", { x: 9.5, y: 2.0, w: 3.3, h: 0.22, fontSize: 9.5, bold: true, color: NAVY });
  video(s, "backup/backup_e2a_cyclist.mp4", 9.5, 2.28, 3.15, 1.817);
  txt(s, "审计臂 E2a@2000(GAN=0):画质高、动态回落", { x: 9.5, y: 4.11, w: 3.3, h: 0.34, fontSize: 8.5, color: GRAY });
  video(s, "backup/backup_w7_cyclist.mp4", 9.5, 4.55, 3.15, 1.817);
  txt(s, "接力 W7@1000(GAN 配对):动态大、质量随迭代回落", { x: 9.5, y: 6.38, w: 3.3, h: 0.34, fontSize: 8.5, color: GRAY });
  footer(s, `12 / ${TOTAL}(备页 B4)`, "出处:experiments/results/2026-07-26-e2a-vb946-fourth-row.md(QS 节)");
  s.addNotes("配合备问卡 #6/#7 使用:E1a 最低源于动态度权重与归一化结构;E2a 最高但多样性最低带且动态回落,合成分恰好测不到这两点;预注册纪律不改选。");
}

/* ============================================================ S13 | B5 flow 多 seed */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "备页 B5|运动幅值多 seed 全表(RAFT 中位,dm40)");
  s.addTable([
    [HC("seed"), HC("接力 W7@1000"), HC("直蒸 E1a@1000"), HC("teacher 50-step")],
    [C("s0", { align: "center" }), C("4.44", { align: "center" }), C("2.15", { align: "center" }), C("2.75", { align: "center" })],
    [C("s1", { align: "center" }), C("3.28", { align: "center" }), C("1.83", { align: "center" }), C("2.17", { align: "center" })],
    [C("s2", { align: "center" }), C("1.27", { align: "center" }), C("0.46", { align: "center" }), C("0.86", { align: "center" })],
    [C("s3", { align: "center" }), C("4.44", { align: "center" }), C("2.80", { align: "center" }), C("2.41", { align: "center" })],
    [C("4-seed 均值", { align: "center", bold: true }), C("3.36", { align: "center", bold: true }), C("1.81", { align: "center", bold: true }), C("2.05", { align: "center", bold: true })],
  ], { x: 0.9, y: 1.5, w: 7.6, colW: [1.6, 2.0, 2.0, 2.0], rowH: [0.4, 0.38, 0.38, 0.38, 0.38, 0.4], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  s.addText([
    { text: "逐 seed 配对结论:", options: { bold: true, breakLine: true } },
    { text: "W7 > E1a:4/4 全同向(约 1.9×)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "W7 > teacher:4/4 全同向(按均值 +64%)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "E1a 不高于 teacher(3/4 同向,按均值 −12%)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "seed2 三模型同低:初始噪声主导部分动态水平——逐 seed 配对设计必要性的实证", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "纪律:单 seed 绝对百分比不单独引用(seed 间中位可差 6 倍)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true } },
  ], { x: 0.9, y: 4.35, w: 11.5, h: 2.2, fontFace: FONT, fontSize: 11, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.15 });
  footer(s, `13 / ${TOTAL}(备页 B5)`, "出处:experiments/results/2026-07-23-flow-multiseed-e1b946-e2a-eval.md §1/§4");
  s.addNotes("被问运动幅值证据强度时展开;正片 P4 只报 4-seed 均值与方向。");
}

/* ============================================================ S14 | B6 上游关系 */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "备页 B6|与上游 NVIDIA FastGen 的关系(声明原文)");
  s.addShape(RR, { x: 0.7, y: 1.5, w: 12.0, h: 3.9, rectRadius: 0.05, fill: { color: PANEL }, line: { color: BORDER, width: 0.75 } });
  txt(s, "我们的全部训练基于 NVIDIA FastGen(NVlabs/FastGen,Apache-2.0),复用其原生 DMD2 实现与官方 Wan2.1-T2V-1.3B 配置——包括 teacher CFG=5、生成端 GAN 权重 0.03、real/fake 共享 timestep 与噪声(gan_use_same_t_noise=True 为官方 Wan 配置出厂值)、teacher 第 15/22/29 层特征上的 multiscale MLP 判别器、student_update_freq=5 的 two-time-scale 更新,以及 4-step t_list=[0.999, 0.937, 0.833, 0.624, 0.0]。\n\n在此之上,我们的配方贡献限于训练日程层:官方仓库仅提供从 50-step teacher 一次蒸到 4-step/2-step 的单阶段配置,我们改为 50→8→4 的 step-count relay(步数接力),新增 8-step 中间 student 阶段,并规定 4-step 阶段仅继承 8-step 最优 checkpoint 的生成器权重、优化器/fake score/判别器全部重新初始化;数据侧选用 OpenVid-1M(上游不绑定数据集)。", { x: 1.0, y: 1.75, w: 11.4, h: 3.5, fontSize: 11, lineSpacingMultiple: 1.25 });
  txt(s, "此外的贡献:本报告的整套受控审计(接力对照、三臂判别器消融)与可移植的退化评估协议。判别器表述统一为:冻结 teacher backbone 第 15/22/29 层特征 + 可训练 multiscale MLP 头(2026-07-06 代码核实)。", { x: 0.7, y: 5.65, w: 12.0, h: 0.7, fontSize: 10.5, color: INK });
  footer(s, `14 / ${TOTAL}(备页 B6)`, "出处:research/T3_novelty_adjudication.md §6.2");
  s.addNotes("被问“你们与 FastGen 什么关系 / 哪些是你们做的”时直接出示本页并照读。不称“改进了 FastGen”;不把单阶段超参说成我方设计。");
}

const OUT = QA_IMAGES ? "DMD2_report_0728_qa.pptx" : "DMD2_report_0728.pptx";
fs.writeFileSync(path.join(__dirname, "qa", "ops.json"), JSON.stringify(QA_LOG));
pres.writeFile({ fileName: path.join(__dirname, OUT) }).then(() => {
  console.log("deck written: " + OUT + " | slides: " + QA_LOG.length);
});
