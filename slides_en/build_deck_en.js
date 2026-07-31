// English grayscale deck — 2026-07-28 progress report (spec: research/report_storyline_0728.md v4)
// Page order: 1 = blank cover placeholder, 2..8 = P2..P8, 9 = "Backup" divider, 10..15 = B1..B6.
// Style: academic black & white. Numbers frozen at commit 0c28815 — transcribed verbatim.
// Build: node build_deck_en.js -> DMD2_report_0728_en.pptx  (QA_IMAGES=1 -> poster-image variant)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ASSETS = path.resolve(__dirname, "../slides_assets");
const COVERS = path.resolve(__dirname, "../slides/covers");

const FONT = "PingFang SC";
const BLACK = "000000";
const INK = "1A1A1A";
const GRAY = "595959";
const LTGRAY = "8C8C8C";
const FILL5 = "F2F2F2"; // 5% gray
const RULE = "000000";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

const L = pres.ShapeType.line;
const ELL = pres.ShapeType.ellipse;
const RR = pres.ShapeType.roundRect;
const RECT = pres.ShapeType.rect;

const TOTAL = "15";

// --- QA instrumentation (mirror renderer reads qa/ops.json) ---
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
        if (a && typeof a === "object" && !Array.isArray(a)) { const c = Object.assign({}, a); delete c.cover; return c; }
        return a;
      });
      rec.ops.push({ op: name, args: clean });
      return orig(...args);
    };
  });
  return s;
};

const QA_IMAGES = !!process.env.QA_IMAGES;
function cover(name) { return "image/png;base64," + fs.readFileSync(path.join(COVERS, name + ".png")).toString("base64"); }
function video(s, rel, x, y, w, h) {
  const base = path.basename(rel, ".mp4");
  if (QA_IMAGES) s.addImage({ path: path.join(COVERS, base + ".png"), x, y, w, h });
  else s.addMedia({ type: "video", path: path.join(ASSETS, rel), x, y, w, h, cover: cover(base) });
}
function title(s, text) {
  s.addText(text, { x: 0.5, y: 0.18, w: 12.33, h: 1.02, fontFace: FONT, fontSize: 28, bold: true, color: BLACK, valign: "top", margin: 0 });
}
function footer(s, label, src) {
  s.addText(label, { x: 0.4, y: 7.1, w: 2.6, h: 0.32, fontFace: FONT, fontSize: 12, color: GRAY, margin: 0 });
  s.addText(src, { x: 3.2, y: 7.1, w: 9.73, h: 0.32, fontFace: FONT, fontSize: 12, color: GRAY, align: "right", margin: 0 });
}
function txt(s, text, o) { s.addText(text, Object.assign({ fontFace: FONT, color: INK, margin: 0, valign: "top" }, o)); }
function seg(s, x1, y1, x2, y2, o) {
  const opt = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: Object.assign({ color: BLACK, width: 2 }, o || {}) };
  if ((x2 - x1) * (y2 - y1) < 0) opt.flipV = true;
  s.addShape(L, opt);
}
function dotF(s, cx, cy, r, color) { s.addShape(ELL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: color || BLACK }, line: { color: "FFFFFF", width: 0.5 } }); }
function dotH(s, cx, cy, r, border) { s.addShape(ELL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: "FFFFFF" }, line: { color: border || GRAY, width: 1.2 } }); }
function sqH(s, cx, cy, r, border) { s.addShape(RECT, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: "FFFFFF" }, line: { color: border || GRAY, width: 1.2 } }); }
function polyline(s, xs, ys, lineOpts) { for (let i = 0; i < xs.length - 1; i++) seg(s, xs[i], ys[i], xs[i + 1], ys[i + 1], lineOpts); }
// three-line (booktabs) table: no cell borders; rules drawn as shapes
function C(text, o) { return { text: String(text), options: Object.assign({ fontFace: FONT, fontSize: 12, color: INK, valign: "middle", border: { type: "none" } }, o || {}) }; }
function BU(text, o) { return C(text, Object.assign({ bold: true, underline: true }, o || {})); }
function H(text, o) { return C(text, Object.assign({ bold: true, color: BLACK, align: "center" }, o || {})); }
function threeLineRules(s, x, w, yTop, rowHs, midAfter) {
  seg(s, x, yTop, x + w, yTop, { color: RULE, width: 1.6 });
  let y = yTop;
  rowHs.forEach((h, i) => { y += h; if (midAfter.includes(i)) seg(s, x, y, x + w, y, { color: RULE, width: 0.9 }); });
  seg(s, x, y, x + w, y, { color: RULE, width: 1.6 });
}
function noteBox(s, x, y, w, h) { s.addShape(RECT, { x, y, w, h, fill: { color: FILL5 }, line: { color: BLACK, width: 0.9 } }); }

/* ============ Slide 1 | cover placeholder ============ */
{
  const s = pres.addSlide();
  s.addShape(RECT, { x: 3.17, y: 2.7, w: 7, h: 1.7, fill: { color: "FFFFFF" }, line: { color: LTGRAY, width: 1, dashType: "dash" } });
  txt(s, "Cover — to be added(封面待补)", { x: 3.17, y: 3.05, w: 7, h: 0.55, fontSize: 28, bold: true, color: LTGRAY, align: "center" });
  txt(s, "Placeholder so that page numbers match the design spec (P2 = slide 2, …).", { x: 3.17, y: 3.7, w: 7, h: 0.35, fontSize: 12, color: LTGRAY, align: "center" });
  footer(s, `1 / ${TOTAL}`, "");
  s.addNotes("Blank placeholder. Final cover to be added; keeps page numbering aligned with the design spec.");
}

/* ============ Slide 2 | P2 methodology ============ */
{
  const s = pres.addSlide();
  title(s, "Subjective checkpoint picking was overturned on both training lines — every conclusion rests on the quantitative protocol");
  txt(s, "Relay arm W7: aesthetic (q150, seed 0) declines monotonically with training, 0.5768 → 0.5379", { x: 0.55, y: 1.34, w: 7.15, h: 0.5, fontSize: 14, bold: true });
  txt(s, "Quantitative optimum: iters 500–1000", { x: 0.9, y: 1.9, w: 3.6, h: 0.28, fontSize: 13, bold: true, underline: true, color: BLACK });
  const px = [1.35, 2.75, 4.15, 5.55, 6.95];
  const vals = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379];
  const y0 = 4.9, f = 2.65 / 0.055; // y range 0.530-0.585 mapped over 2.65in
  const py = vals.map((v) => y0 - (v - 0.53) * f);
  seg(s, 1.35, y0, 6.95, y0, { color: LTGRAY, width: 1 });
  seg(s, 1.35, 2.2, 1.35, y0, { color: LTGRAY, width: 1 });
  [0.54, 0.56, 0.58].forEach((t) => {
    const yy = y0 - (t - 0.53) * f;
    txt(s, t.toFixed(2), { x: 0.62, y: yy - 0.1, w: 0.65, h: 0.2, fontSize: 11, color: GRAY, align: "right" });
    seg(s, 1.29, yy, 1.35, yy, { color: LTGRAY, width: 1 });
  });
  ["500", "1000", "1500", "2000", "2500"].forEach((t, k) => txt(s, t, { x: px[k] - 0.4, y: y0 + 0.06, w: 0.8, h: 0.2, fontSize: 11, color: GRAY, align: "center" }));
  txt(s, "training iteration", { x: 1.35, y: y0 + 0.3, w: 5.6, h: 0.22, fontSize: 11, color: GRAY, align: "center" });
  txt(s, "y-axis 0.530–0.585 (not zero-based)", { x: 4.3, y: 2.22, w: 2.65, h: 0.2, fontSize: 11, color: GRAY, align: "right" });
  seg(s, 1.35, 2.14, 2.75, 2.14, { color: BLACK, width: 1.4 });
  seg(s, 1.35, 2.14, 1.35, 2.22, { color: BLACK, width: 1.4 });
  seg(s, 2.75, 2.14, 2.75, 2.22, { color: BLACK, width: 1.4 });
  polyline(s, px, py, { color: BLACK, width: 2.5 });
  px.forEach((x, i) => dotF(s, x, py[i], 0.055, BLACK));
  txt(s, "0.5768", { x: 1.5, y: py[0] - 0.28, w: 0.85, h: 0.2, fontSize: 12, bold: true, color: BLACK });
  txt(s, "0.5379", { x: 6.28, y: py[4] + 0.1, w: 0.85, h: 0.2, fontSize: 12, bold: true, color: BLACK });
  txt(s, "Subjective pick: iter 2500\n(end of the decline)", { x: 4.62, y: 3.32, w: 2.35, h: 0.55, fontSize: 12, color: INK });
  seg(s, 6.6, 3.9, 6.9, py[4] - 0.08, { color: LTGRAY, width: 1 });
  // right: W4 counter-example (three-line table)
  txt(s, "Counter-example: uniform t_list ablation (W4) — the protocol's positive control", { x: 7.7, y: 1.34, w: 5.13, h: 0.55, fontSize: 14, bold: true });
  const w4Rows = [
    [H("Metric", { align: "left", fontSize: 12 }), H("W4", { fontSize: 12 }), H("Rank in full table", { fontSize: 12 })],
    [C("subject / background consistency"), C("0.9745 / 0.9791", { align: "center" }), C("highest", { align: "center", bold: true })],
    [C("imaging quality"), C("0.2555", { align: "center" }), C("lowest", { align: "center", bold: true })],
    [C("cross-seed diversity"), C("0.4617", { align: "center" }), C("lowest", { align: "center", bold: true })],
  ];
  const w4RH = [0.38, 0.42, 0.42, 0.42];
  s.addTable(w4Rows, { x: 7.7, y: 1.98, w: 5.13, colW: [2.53, 1.4, 1.2], rowH: w4RH, border: { type: "none" }, fontFace: FONT, valign: "middle" });
  threeLineRules(s, 7.7, 5.13, 1.98, w4RH, [0]);
  txt(s, "Consistency-type metrics alone would rank an almost-static, collapsed model as the best.", { x: 7.7, y: 3.8, w: 5.13, h: 0.85, fontSize: 15, bold: true, color: BLACK });
  txt(s, "Joint-reading rule: consistency metrics are always read together with dynamic degree and diversity.", { x: 7.7, y: 4.7, w: 5.13, h: 0.6, fontSize: 12, color: GRAY });
  // bottom band
  noteBox(s, 0.5, 5.55, 12.33, 1.42);
  txt(s, "This established the four-component protocol: six quality dims (q150) / clean dynamic degree (dm40) / cross-seed diversity (d40×8) / continuous optical flow (RAFT) — details in Backup B1.\nCheckpoint policy: save every 500 iters, score the full sweep (best-of-sweep), re-check champions across seeds (n=3).", { x: 0.78, y: 5.7, w: 11.8, h: 1.15, fontSize: 15, lineSpacingMultiple: 1.12 });
  footer(s, `2 / ${TOTAL}`, "Source: experiments/results/2026-07-14-e0-full-table-g1.md");
  s.addNotes("60s. The two lines err in opposite directions (W7 picked too late, W1 too early) — a protocol failure, not a one-off. W4 shows the systematic blind spot of consistency-type metrics. Plant the hook: quality peaks early then declines — mechanism answered on slide 5. Red lines: do not claim novelty of t_list ablation (TMD); consistency always read jointly with dynamics/diversity.");
}

/* ============ Slide 3 | P3 core finding ============ */
{
  const s = pres.addSlide();
  title(s, "The main degradation axis is cross-seed diversity (≈ −15–20%); dynamic degree does not drop");
  txt(s, "Same prompt (“An astronaut walking slowly across a dusty red desert…”, in-domain eval set); only the random seed differs. Thumbnails = frame 40; click to play.", { x: 0.5, y: 1.24, w: 12.33, h: 0.26, fontSize: 12, color: GRAY });
  const cw = 1.385, ch = 0.799, gap = 0.028, x0 = 1.52;
  const tSeeds = [0, 1, 2, 3, 4, 5, 6, 7];
  const eSeeds = [0, 2, 3, 4, 6, 8, 9, 11];
  txt(s, "teacher\n50 steps", { x: 0.34, y: 1.66, w: 1.1, h: 0.5, fontSize: 12, bold: true, align: "center" });
  txt(s, "165.24 s/clip", { x: 0.34, y: 2.16, w: 1.1, h: 0.18, fontSize: 9.5, color: GRAY, align: "center" });
  txt(s, "student E1a\n4 steps", { x: 0.34, y: 2.72, w: 1.1, h: 0.5, fontSize: 12, bold: true, align: "center" });
  txt(s, "≈6.6 s/clip", { x: 0.34, y: 3.22, w: 1.1, h: 0.18, fontSize: 9.5, color: GRAY, align: "center" });
  tSeeds.forEach((sd, k) => { const x = x0 + k * (cw + gap); video(s, `p3/p3v2_teacher_s${sd}.mp4`, x, 1.54, cw, ch); txt(s, `s${sd}`, { x, y: 2.35, w: cw, h: 0.16, fontSize: 10, color: GRAY, align: "center" }); });
  eSeeds.forEach((sd, k) => { const x = x0 + k * (cw + gap); video(s, `p3/p3v2_e1a_s${sd}.mp4`, x, 2.57, cw, ch); txt(s, `s${sd}`, { x, y: 3.38, w: cw, h: 0.16, fontSize: 10, color: GRAY, align: "center" }); });
  // two panels, horizontal bars, shared 0-1 scale
  const barW = 3.9, barX = 2.15, barX2 = 8.55;
  function tbar(x, y, len) { s.addShape(RECT, { x, y, w: Math.max(len, 0.02), h: 0.34, fill: { color: BLACK }, line: { type: "none" } }); }
  function sband(x, y, len) { s.addShape(RECT, { x, y, w: Math.max(len, 0.03), h: 0.34, fill: { color: "FFFFFF" }, line: { color: BLACK, width: 1.2 } }); }
  txt(s, "Dynamic degree DD_clean (dm40 clean motion set, scale 0–1): no drop", { x: 0.5, y: 3.68, w: 6.1, h: 0.45, fontSize: 13.5, bold: true });
  txt(s, "teacher", { x: 0.6, y: 4.24, w: 1.45, h: 0.3, fontSize: 12, align: "right" });
  txt(s, "students", { x: 0.6, y: 4.8, w: 1.45, h: 0.3, fontSize: 12, align: "right" });
  seg(s, barX, 4.16, barX, 5.28, { color: LTGRAY, width: 1 });
  tbar(barX, 4.2, 0.625 * barW);
  txt(s, "0.625", { x: barX + 0.625 * barW + 0.07, y: 4.26, w: 0.75, h: 0.22, fontSize: 12, bold: true, color: BLACK });
  sband(barX + 0.75 * barW, 4.76, 0.25 * barW);
  txt(s, "0.75–1.0 (all student champions)", { x: barX + 0.07, y: 4.82, w: 2.8, h: 0.22, fontSize: 11, color: INK });
  [0, 0.5, 1].forEach((t) => txt(s, String(t), { x: barX + t * barW - 0.2, y: 5.32, w: 0.4, h: 0.2, fontSize: 10.5, color: GRAY, align: "center" }));
  txt(s, "Motion smoothness 0.97+: real motion, not jitter. The dynamics collapse feared in the literature is not reproduced here (different experimental domain).", { x: 0.5, y: 5.58, w: 6.1, h: 0.58, fontSize: 12, color: GRAY });
  txt(s, "Cross-seed diversity (pairwise LPIPS, d40×8, scale 0–1): consistent drop — the main degradation axis", { x: 6.9, y: 3.68, w: 6.0, h: 0.45, fontSize: 13.5, bold: true, underline: true });
  txt(s, "teacher", { x: 7.0, y: 4.24, w: 1.45, h: 0.3, fontSize: 12, align: "right" });
  txt(s, "students", { x: 7.0, y: 4.8, w: 1.45, h: 0.3, fontSize: 12, align: "right" });
  seg(s, barX2, 4.16, barX2, 5.28, { color: LTGRAY, width: 1 });
  tbar(barX2, 4.2, 0.732 * barW);
  txt(s, "0.732", { x: barX2 + 0.732 * barW + 0.07, y: 4.26, w: 0.75, h: 0.22, fontSize: 12, bold: true, color: BLACK });
  sband(barX2 + 0.59 * barW, 4.76, 0.05 * barW);
  txt(s, "0.59–0.64 (all student champions)", { x: barX2 + 0.07, y: 4.82, w: 2.2, h: 0.22, fontSize: 11, bold: true, color: BLACK });
  [0, 0.5, 1].forEach((t) => txt(s, String(t), { x: barX2 + t * barW - 0.2, y: 5.32, w: 0.4, h: 0.2, fontSize: 10.5, color: GRAY, align: "center" }));
  txt(s, "≈ −15–20% vs the teacher; weak/strong recipes, direct and relay — all in the same direction. The most-replicated finding of this project.", { x: 6.9, y: 5.58, w: 6.0, h: 0.58, fontSize: 12, color: GRAY });
  txt(s, "Diversity = mean pairwise perceptual distance (LPIPS) among 8 videos from the same prompt with different seeds (higher = more diverse). The two rows above visualize it: the teacher varies across seeds; the student collapses to one template.", { x: 0.5, y: 6.2, w: 12.33, h: 0.72, fontSize: 14.5 });
  footer(s, `3 / ${TOTAL}`, "Source: experiments/results/2026-07-14-e0-full-table-g1.md · 2026-07-20-g2-relay-vs-direct-final.md");
  s.addNotes("75s. Literature worries about losing dynamics; our measurement says dynamics do not drop (smoothness 0.97+ rules out jitter; phrase as 'not reproduced', domains differ). What drops consistently is cross-seed diversity — every recipe and route, same direction. Wall: student seeds are 0/2/3/4/6/8/9/11, labeled as-is. Fixed transition: 'This degradation appears in every configuration — two controlled ablations now locate its source.'");
}

/* ============ Slide 4 | P4 ablation I ============ */
{
  const s = pres.addSlide();
  title(s, "Budget-matched controlled comparison: the step-count relay adds no quality gain and lowers diversity; its only stable effect is higher motion magnitude");
  const rows = [
    [H("Model (best-of-sweep)", { align: "left", fontSize: 12 }), H("aesthetic", { fontSize: 12 }), H("imaging", { fontSize: 12 }), H("DD_clean", { fontSize: 12 }), H("diversity", { fontSize: 12 })],
    [C("teacher 50-step CFG5"), BU("0.590", { align: "center" }), C("0.692", { align: "center" }), C("0.625", { align: "center" }), BU("0.732", { align: "center" })],
    [C("Direct E1a @1000 (final model, G2)"), C("0.567", { align: "center" }), BU("0.717", { align: "center" }), C("0.750", { align: "center" }), C("0.635", { align: "center" })],
    [C("Relay W7 @500"), C("0.577", { align: "center" }), C("0.694", { align: "center" }), C("0.825", { align: "center" }), C("0.598", { align: "center" })],
    [C("Relay W7 @1000"), C("0.559", { align: "center" }), C("0.697", { align: "center" }), BU("1.000", { align: "center" }), C("0.613", { align: "center" })],
    [C("Direct E1b @500"), C("0.532", { align: "center" }), C("0.695", { align: "center" }), C("0.975", { align: "center" }), C("0.628", { align: "center" })],
  ];
  const RH = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4];
  s.addTable(rows, { x: 0.5, y: 1.5, w: 7.35, colW: [3.15, 1.0, 1.0, 1.1, 1.1], rowH: RH, border: { type: "none" }, fontFace: FONT, valign: "middle" });
  threeLineRules(s, 0.5, 7.35, 1.5, RH, [0]);
  txt(s, "Protocol: aesthetic/imaging = q150 six-dim (seed 0); DD_clean = dm40; diversity = d40×8 LPIPS — the three domains are never mixed. Column best = bold + underline.", { x: 0.5, y: 3.98, w: 7.35, h: 0.42, fontSize: 12, color: GRAY });
  txt(s, "Note: E1a imaging 0.717 is higher than the teacher's 0.692 — read as sharpness/static bias, jointly with dynamics and diversity. E1a's DD_clean 0.75 is lower than the other student arms (0.825–1.0); stated as-is.", { x: 0.5, y: 4.42, w: 7.35, h: 0.55, fontSize: 12, color: GRAY });
  txt(s, "Same prompt & seed (cyclist): the motion-magnitude contrast, directly visible", { x: 0.6, y: 5.02, w: 6.5, h: 0.24, fontSize: 12, color: GRAY });
  video(s, "p4/p4_w7_cyclist.mp4", 0.6, 5.3, 2.72, 1.57);
  video(s, "p4/p4_e1a_cyclist.mp4", 3.66, 5.3, 2.72, 1.57);
  txt(s, "Relay W7@1000 (4 steps)", { x: 0.6, y: 6.88, w: 2.72, h: 0.22, fontSize: 12, align: "center", color: GRAY });
  txt(s, "Direct E1a@1000 (4 steps)", { x: 3.66, y: 6.88, w: 2.72, h: 0.22, fontSize: 12, align: "center", color: GRAY });
  // right-top design box
  noteBox(s, 8.1, 1.5, 4.73, 2.85);
  txt(s, "Controlled design (pre-registered)", { x: 8.3, y: 1.62, w: 4.35, h: 0.3, fontSize: 14, bold: true, color: BLACK });
  txt(s, "Relay arm 50→8→4: W5 (2500 it) + W7 (2500 it)\nDirect arms ×2, 50→4: E1a (relay stage-2 recipe) / E1b (stage-1 recipe = upstream default LR), 5000 it each\nInvariants: total budget · data · 4-step t_list · single-stage recipe · eval protocol; the two-arm bracket forestalls “untuned baseline”", { x: 8.3, y: 1.96, w: 4.35, h: 1.9, fontSize: 11.5, lineSpacingMultiple: 1.12 });
  txt(s, "Pre-registered expectation: parity → observed: direct slightly ahead", { x: 8.3, y: 3.9, w: 4.35, h: 0.4, fontSize: 12, bold: true, underline: true, color: BLACK });
  // right-bottom flow bars
  txt(s, "Motion magnitude (RAFT flow median, dm40, 4-seed mean)", { x: 8.1, y: 4.58, w: 4.73, h: 0.28, fontSize: 13, bold: true });
  txt(s, "Relay W7", { x: 8.1, y: 4.98, w: 1.25, h: 0.3, fontSize: 12, align: "right" });
  txt(s, "Direct E1a", { x: 8.1, y: 5.48, w: 1.25, h: 0.3, fontSize: 12, align: "right" });
  seg(s, 9.45, 4.92, 9.45, 5.88, { color: LTGRAY, width: 1 });
  s.addShape(RECT, { x: 9.45, y: 4.96, w: (3.36 / 4) * 2.9, h: 0.34, fill: { color: BLACK }, line: { type: "none" } });
  txt(s, "3.36", { x: 9.45 + (3.36 / 4) * 2.9 + 0.07, y: 5.02, w: 0.6, h: 0.22, fontSize: 12, bold: true, color: BLACK });
  s.addShape(RECT, { x: 9.45, y: 5.46, w: (1.81 / 4) * 2.9, h: 0.34, fill: { color: "FFFFFF" }, line: { color: BLACK, width: 1.2 } });
  txt(s, "1.81", { x: 9.45 + (1.81 / 4) * 2.9 + 0.07, y: 5.52, w: 0.6, h: 0.22, fontSize: 12, bold: true, color: BLACK });
  txt(s, "Same direction in 4/4 seeds, ≈1.9×. Whether this is good depends on prizing closeness to the teacher vs larger dynamics — we take no side (full table: Backup B5).", { x: 8.1, y: 5.98, w: 4.73, h: 0.8, fontSize: 12, color: GRAY });
  footer(s, `4 / ${TOTAL}`, "Source: 2026-07-20-g2-relay-vs-direct-final.md · thesis_ch2_draft.md (F1/F5) · 2026-07-23-flow-multiseed-….md");
  s.addNotes("90s. The relay was our own original design, hence the pre-registration. Honest outcome: quality parity or direct slightly ahead (imaging 0.717 is the student max; aesthetic tie under n=3 bands); diversity higher on both direct arms; motion magnitude is the relay's only stable measured difference — mechanism next slide. Red lines: sharpness/static-bias reading for 0.717; state E1a's low DD 0.75; never quote single-seed percentages; no claim that direct distillation is novel.\n\nQ&A card #1 (why negative results matter): both are progress by controlled exclusion — first controlled relay-vs-direct comparison on this base (GPD/CoDMD/FastWan did not run one), plus the relay's one real effect (motion, 4/4 seeds); the pairing ablation is the first controlled test of a widely copied default.");
}

/* ============ Slide 5 | P5 ablation II ============ */
{
  const s = pres.addSlide();
  title(s, "Single-variable ablations: the GAN branch drives the quality decline over training and the motion magnitude; the (t,ε) pairing convention shows no detected effect");
  txt(s, "Same initialization (W5@2500), same recipe, only the GAN setting differs — aesthetic (q150, seed 0) vs training iteration", { x: 0.55, y: 1.42, w: 7.3, h: 0.5, fontSize: 13.5, bold: true });
  const px = [0, 1, 2, 3, 4].map((k) => 1.35 + k * 1.155);
  const yb = 5.62, f2 = 3.3 / 0.09; // 0.53-0.62 over 3.3in
  const Y = (v) => yb - (v - 0.53) * f2;
  const e2a = [0.5908, 0.5921, 0.5984, 0.6109, 0.6074].map(Y);
  const e2b = [0.5774, 0.567, 0.5477, 0.5471, 0.5487].map(Y);
  const w7 = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379].map(Y);
  seg(s, 1.35, yb, 5.97, yb, { color: LTGRAY, width: 1 });
  seg(s, 1.35, 2.1, 1.35, yb, { color: LTGRAY, width: 1 });
  [0.54, 0.56, 0.58, 0.6, 0.62].forEach((t) => {
    txt(s, t.toFixed(2), { x: 0.62, y: Y(t) - 0.1, w: 0.65, h: 0.2, fontSize: 11, color: GRAY, align: "right" });
    seg(s, 1.29, Y(t), 1.35, Y(t), { color: LTGRAY, width: 1 });
  });
  ["500", "1000", "1500", "2000", "2500"].forEach((t, k) => txt(s, t, { x: px[k] - 0.4, y: yb + 0.06, w: 0.8, h: 0.2, fontSize: 11, color: GRAY, align: "center" }));
  txt(s, "training iteration", { x: 1.35, y: yb + 0.3, w: 4.62, h: 0.22, fontSize: 11, color: GRAY, align: "center" });
  txt(s, "y-axis 0.53–0.62", { x: 1.44, y: 2.12, w: 1.6, h: 0.2, fontSize: 11, color: GRAY });
  polyline(s, px, w7, { color: GRAY, width: 2, dashType: "dashDot" });
  polyline(s, px, e2b, { color: GRAY, width: 2, dashType: "dash" });
  polyline(s, px, e2a, { color: BLACK, width: 2.6 });
  px.forEach((x, i) => { sqH(s, x, w7[i], 0.05); dotH(s, x, e2b[i], 0.05); dotF(s, x, e2a[i], 0.055, BLACK); });
  txt(s, "0.5908", { x: 1.46, y: e2a[0] - 0.3, w: 0.8, h: 0.2, fontSize: 11, bold: true, color: BLACK });
  txt(s, "≈0.577 (both GAN-on arms start together)", { x: 1.46, y: 3.86, w: 3.3, h: 0.2, fontSize: 10.5, color: GRAY });
  txt(s, "0.6109 (peak) @2000", { x: 4.15, y: e2a[3] - 0.3, w: 1.8, h: 0.2, fontSize: 11, color: BLACK });
  txt(s, "E2a: GAN off — final 0.6074", { x: 6.05, y: e2a[4] - 0.22, w: 1.85, h: 0.6, fontSize: 12, bold: true, color: BLACK });
  txt(s, "E2b: GAN on (indep. t,ε) — final 0.5487", { x: 6.05, y: e2b[4] - 0.5, w: 1.85, h: 0.75, fontSize: 11, color: GRAY });
  txt(s, "W7: GAN on (paired) — final 0.5379", { x: 6.05, y: w7[4] + 0.02, w: 1.85, h: 0.75, fontSize: 11, color: GRAY });
  // right column: three conclusion boxes
  const box = (y, h, head, body) => {
    noteBox(s, 8.0, y, 4.85, h);
    txt(s, head, { x: 8.18, y: y + 0.09, w: 4.5, h: 0.52, fontSize: 13, bold: true, color: BLACK });
    txt(s, body, { x: 8.18, y: y + 0.62, w: 4.5, h: h - 0.72, fontSize: 11.5, lineSpacingMultiple: 1.08 });
  };
  box(1.42, 1.72, "① Quality “early peak, later decline” ← GAN branch (candidate mechanism)", "GAN off (E2a): quality improves through training, all five checkpoints one way. GAN on (E2b/W7): it declines — the phenomenon disappears and reverses under a single-variable control; champions n=3, 3/3 seeds paired one way.");
  box(3.26, 1.6, "② Motion magnitude is mainly sustained by the GAN branch", "GAN off: falls back to the teacher's level (2.1–2.7). GAN on: rebuilds from a low point up to 4.71. Interacts with the relay initialization within the matched recipe (direct arm E1a has GAN on yet stays low).");
  box(4.98, 1.52, "③ (t,ε) pairing convention: no detected effect", "Sharing (t,ε) between real/fake is the upstream default; to our knowledge it had never been tested under control. Across all five checkpoint pairs the quality gap is ≤0.011; every metric follows the same trajectory.");
  noteBox(s, 0.5, 6.6, 12.33, 0.42);
  txt(s, "Cross-seed diversity of all three arms stays within 0.586–0.613 — insensitive to the GAN switch and the pairing convention (wrapped up next).", { x: 0.78, y: 6.68, w: 11.8, h: 0.3, fontSize: 13.5, bold: true });
  footer(s, `5 / ${TOTAL}`, "Source: 2026-07-24-e2a-fulltable-ch3.md · 2026-07-25-e2b-fulltable-ch3-threearm.md");
  s.addNotes("105s. One sentence on cleanliness: each arm changes exactly one field of the W7 recipe (E2a: gan_loss_weight_gen 0.03→0; E2b: gan_use_same_t_noise True→False), configs verified value-by-value. 20s on the center chart; 15s per conclusion; stress ③: a widely copied default tested under control for the first time. Red lines: 'candidate mechanism' (single weight 0.03, single lineage); never 'GAN always raises motion' (E1a is the counter-example); E2a aesthetic 0.613 vs teacher 0.590 — no comparison wording; say 'no detected effect', not 'no difference'.\n\nQ&A card #3 (why not just turn GAN off): single weight point, single lineage; and motion falls back to teacher level — a trade-off, not a free win; we deliver attribution, not a recipe recommendation.");
}

/* ============ Slide 6 | P6 attribution wrap-up ============ */
{
  const s = pres.addSlide();
  title(s, "After component-wise controlled exclusion the diversity collapse persists — within the tested components it is attributed to distribution-matching distillation itself");
  const comp = (y, name, ev) => {
    s.addShape(RECT, { x: 0.6, y, w: 5.0, h: 1.14, fill: { color: "FFFFFF" }, line: { color: BLACK, width: 1 } });
    txt(s, name, { x: 0.82, y: y + 0.1, w: 3.1, h: 0.3, fontSize: 14, bold: true, color: BLACK });
    s.addShape(RR, { x: 4.05, y: y + 0.12, w: 1.42, h: 0.32, rectRadius: 0.12, fill: { color: FILL5 }, line: { color: GRAY, width: 0.75 } });
    txt(s, "excluded (ctrl.)", { x: 4.05, y: y + 0.17, w: 1.42, h: 0.24, fontSize: 10.5, color: INK, align: "center" });
    txt(s, ev, { x: 0.82, y: y + 0.46, w: 4.6, h: 0.62, fontSize: 11.5, color: GRAY });
    seg(s, 5.6, y + 0.57, 6.25, y + 0.57, { color: GRAY, width: 1.25, endArrowType: "triangle" });
  };
  comp(1.62, "Step-count relay (50→8→4)", "Evidence P4 — the relay arms have LOWER diversity (0.598/0.613 vs direct 0.628/0.635)");
  comp(3.1, "GAN discriminator branch", "Evidence P5 — switching it does not move the collapse (three arms 0.586–0.613)");
  comp(4.58, "(t,ε) pairing convention", "Evidence P5 — no detected effect at any of the five checkpoints");
  txt(s, "Still present after exclusion: cross-seed diversity collapse", { x: 6.5, y: 1.66, w: 6.4, h: 0.32, fontSize: 14, bold: true });
  const bx = 7.85, bw = 4.2; // scale 0-0.8
  txt(s, "teacher", { x: 6.45, y: 2.62, w: 1.3, h: 0.3, fontSize: 12, align: "right" });
  txt(s, "students", { x: 6.45, y: 3.42, w: 1.3, h: 0.3, fontSize: 12, align: "right" });
  seg(s, bx, 2.5, bx, 4.02, { color: LTGRAY, width: 1 });
  s.addShape(RECT, { x: bx, y: 2.58, w: (0.732 / 0.8) * bw, h: 0.4, fill: { color: BLACK }, line: { type: "none" } });
  txt(s, "0.732", { x: bx + (0.732 / 0.8) * bw + 0.07, y: 2.66, w: 0.7, h: 0.22, fontSize: 12, bold: true, color: BLACK });
  s.addShape(RECT, { x: bx + (0.586 / 0.8) * bw, y: 3.38, w: (0.063 / 0.8) * bw, h: 0.4, fill: { color: "FFFFFF" }, line: { color: BLACK, width: 1.2 } });
  txt(s, "0.586–0.649", { x: bx + (0.586 / 0.8) * bw - 0.45, y: 3.06, w: 1.4, h: 0.22, fontSize: 11.5, bold: true, color: BLACK });
  [0, 0.4, 0.8].forEach((t) => txt(s, t.toFixed(1), { x: bx + (t / 0.8) * bw - 0.2, y: 4.06, w: 0.4, h: 0.2, fontSize: 10.5, color: GRAY, align: "center" }));
  txt(s, "Cross-seed diversity (LPIPS, d40×8; students = per-arm champions plus the weak-recipe reference)", { x: 6.5, y: 4.32, w: 6.4, h: 0.26, fontSize: 11, color: GRAY });
  txt(s, "Within the tested components, the collapse is attributed to distribution-matching distillation itself. Untested components, listed as-is: data composition, how the teacher's CFG is distilled, other GAN weights.", { x: 6.5, y: 4.66, w: 6.4, h: 0.85, fontSize: 13 });
  noteBox(s, 1.2, 5.9, 10.93, 0.95);
  txt(s, "This work does not offer a fix. It delivers a precisely measured problem statement whose candidate sources have been excluded under control — an open problem.", { x: 1.5, y: 6.1, w: 10.3, h: 0.6, fontSize: 15, bold: true, color: BLACK, align: "center" });
  footer(s, `6 / ${TOTAL}`, "Source: 2026-07-25-e2b-fulltable-ch3-threearm.md (verdict 4)");
  s.addNotes("45s. 30s on the exclusion logic: switch the route — still there (and deeper on relay); turn the GAN off — still there; change the pairing — still there. 15s on the open-problem framing: the core takeaway. Red lines: scope 'within the tested components'; never write 'unsolvable'.");
}

/* ============ Slide 7 | P7 VBench ============ */
{
  const s = pres.addSlide();
  title(s, "No dominant model on the standard benchmark — different models lead on different dimensions; the diversity upper bound still belongs to the teacher");
  const span = (t) => [{ text: t, options: { colspan: 5, fill: { color: FILL5 }, bold: true, fontSize: 10.5, color: INK, align: "left", fontFace: FONT, valign: "middle", border: { type: "none" } } }];
  const R = (name, v, best) => [C(name, { fontSize: 11 })].concat(v.map((x, i) => (i === best ? BU(x, { align: "center", fontSize: 11 }) : C(x, { align: "center", fontSize: 11 }))));
  const rows = [
    [H("Dimension", { align: "left", fontSize: 10.5 }), H("E1a@1000 final (G2)", { fontSize: 10 }), H("W7@1000 relay", { fontSize: 10 }), H("E1b@500 direct-B", { fontSize: 10 }), H("E2a@2000 audit GAN=0 · single ckpt", { fontSize: 10 })],
    span("Quality dimensions (7) — exactly the official Quality Score set; composable with official weights (Backup B4)"),
    R("subject consistency", ["0.9727", "0.9693", "0.9673", "0.9753"], 3),
    R("background consistency", ["0.9579", "0.9508", "0.9416", "0.9581"], 3),
    R("motion smoothness", ["0.9812", "0.9727", "0.9747", "0.9786"], 0),
    R("dynamic degree", ["0.5806", "0.9111", "0.8806", "0.8000"], 1),
    R("aesthetic quality", ["0.5967", "0.6087", "0.5802", "0.6482"], 3),
    R("imaging quality", ["0.6687", "0.6687", "0.6614", "0.6924"], 3),
    R("temporal flickering", ["0.9894", "0.9796", "0.9810", "0.9878"], 0),
    span("Semantic-type raw dimensions (5) — NOT the official Semantic (4 GRiT dimensions missing; not composable)"),
    R("human action", ["0.690", "0.794", "0.716", "0.776"], 1),
    R("scene", ["0.2173", "0.2922", "0.2225", "0.2974"], 3),
    R("appearance style", ["0.1990", "0.1982", "0.2003", "0.2010"], 3),
    R("temporal style", ["0.2214", "0.2305", "0.2260", "0.2283"], 1),
    R("overall consistency", ["0.2240", "0.2386", "0.2298", "0.2391"], 1),
  ];
  const RH = [0.52, 0.26].concat(Array(7).fill(0.3), [0.26], Array(5).fill(0.3));
  s.addTable(rows, { x: 0.45, y: 1.42, w: 8.6, colW: [2.5, 1.5, 1.45, 1.5, 1.65], rowH: RH, border: { type: "none" }, fontFace: FONT, valign: "middle" });
  threeLineRules(s, 0.45, 8.6, 1.42, RH, [0, 1, 8, 9]);
  txt(s, "full VBench standard mode, 946 prompts × 5 seeds, 12/16 dimensions; row best = bold + underline. The 4 missing dimensions are detector-dependent (GRiT/detectron2), so the official Semantic and Total cannot be composed — stated as-is. temporal_flickering: the official protocol uses a dedicated subset, 25 samples/prompt after a static filter; ours is 5 samples, unfiltered — not directly comparable.", { x: 0.45, y: 6.28, w: 8.6, h: 0.78, fontSize: 12, color: GRAY });
  txt(s, "How to read", { x: 9.3, y: 1.42, w: 3.55, h: 0.3, fontSize: 14, bold: true, color: BLACK });
  const rb = (y, head, body, hh) => {
    txt(s, head, { x: 9.3, y, w: 3.55, h: 0.26, fontSize: 12.5, bold: true });
    txt(s, body, { x: 9.3, y: y + 0.28, w: 3.55, h: hh || 0.7, fontSize: 11.5, color: INK, lineSpacingMultiple: 1.08 });
  };
  rb(1.82, "E1a (final model)", "leads on consistency / smoothness / flicker");
  rb(2.78, "W7 (relay)", "leads on dynamic degree 0.911 and action semantics (0.794 / 0.292)");
  rb(3.78, "E2a (audit arm)", "best static quality (0.6482 / 0.6924); dynamics hold at 0.80 — agrees with the ablation across domains", 0.95);
  rb(5.0, "teacher", "diversity 0.732 (q150 domain) remains the upper bound; the axis is absent from VBench — exactly why we built our own protocol (Backup B4)", 1.2);
  footer(s, `7 / ${TOTAL}`, "Source: experiments/results/2026-07-26-e2a-vb946-fourth-row.md");
  s.addNotes("60s. Each model leads somewhere; E2a's cross-domain agreement supports the ablation; diversity is absent from VBench. Red lines: the final model follows the pre-registered G2 outcome (E1a@1000), no after-the-fact switching; composite scores and the literature-number coincidence live only in Backup B4 / oral answers, not on the main slides; never mix dynamic-degree numbers across domains (q150 0.567 vs vb946 0.800).\n\nQ&A #2 (CoDMD 84.46): protocols not comparable (12/16 dims, 5-sample flickering, no GRiT dims) — literature coordinate only; our W7 Quality Score 84.47 is a numeric coincidence (7-dim quality composite vs 16-dim total), never side by side.\nQ&A #6 (why is the final model's composite lowest, 82.80): weight structure — dynamic degree enters at 0.5 weight and E1a's 0.5806 drags it; its winning dims lose resolution after normalization; and the composite has no diversity term. Selection follows the pre-registered protocol.\nQ&A #7 (E2a highest, 85.50 — why not switch): pre-registration discipline; E2a has the lowest diversity band (0.586–0.604) and teacher-level dynamics — exactly what the composite cannot see; single run, single weight point.");
}

/* ============ Slide 8 | P8 conclusions ============ */
{
  const s = pres.addSlide();
  title(s, "Conclusions: the 25× speed-up works; the main degradation is localized to diversity collapse; three actionable recommendations");
  const col = (x, head, items, fs) => {
    txt(s, head, { x, y: 1.42, w: 4.05, h: 0.34, fontSize: 16, bold: true, color: BLACK });
    s.addText(items.map((t) => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 10 } })), { x, y: 1.84, w: 4.05, h: 4.75, fontFace: FONT, fontSize: fs || 13.5, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.12 });
  };
  col(0.5, "Main conclusions", [
    "Diagnosis: the main axis is cross-seed diversity collapse (teacher 0.732 → students 0.59–0.64), not dynamic degree",
    "Attribution: the quality decline and the motion magnitude tie mainly to the GAN branch; the (t,ε) pairing shows no detected effect; the diversity collapse comes from distillation itself (open problem)",
    "Method: a lightweight, seed-controlled degradation-audit protocol (incl. evidence that subjective checkpoint picking is unreliable)",
    "Engineering: 25× speed-up; the first controlled relay-vs-direct comparison on this base model (to our knowledge, with search coverage)",
  ], 12.5);
  col(4.78, "Recommendations", [
    "Save every 500 iterations and score the full sweep — quality usually peaks early; never trust subjective picks or the last checkpoint",
    "Monitor cross-seed diversity and continuous optical flow — standard metrics and composite scores do not flag the main degradation axis",
    "Treat the GAN weight as a static-quality ↔ motion-magnitude trade-off; the (t,ε) pairing needs no tuning (caveat: single weight point, single lineage — extrapolate with care)",
  ], 12.5);
  col(9.06, "Limitations & next", [
    "No human evaluation; 4 benchmark dimensions missing",
    "R1 arm untested under the 32 GB VRAM limit (calibration archived; reproducible on 80 GB-class devices)",
    "One training run per configuration (mitigated by same-family agreement); the relay source was picked before the selection policy existed",
    "Next: human eval · cross-lineage ablation · 80 GB rerun · interventions on diversity collapse",
  ], 12.5);
  txt(s, "Thank you — comments welcome. Thesis due 2026-07-31.", { x: 0.5, y: 6.72, w: 12.33, h: 0.34, fontSize: 14, bold: true, color: BLACK, align: "center" });
  footer(s, `8 / ${TOTAL}`, "Source: T3_novelty_adjudication.md §4.1 · 2026-07-25-e2b-fulltable-ch3-threearm.md · acceptance-log.md #11–#13");
  s.addNotes("60s, ~20s per column. Red lines: every 'first' carries 'to our knowledge + search coverage'; recommendation ③ must keep its caveat.\n\nQ&A #4 (why no human eval): the time window went to controlled ablations; human eval (T2VHE-style vs teacher) is listed under limitations and next steps.\nQ&A #5 (why is the R1 arm untested): deterministic OOM on the 32 GB card (crash at R1's second discriminator forward), not a recipe failure; calibration values and config are archived, directly reproducible on 80 GB.");
}

/* ============ Slide 9 | Backup divider ============ */
{
  const s = pres.addSlide();
  txt(s, "Backup", { x: 0.5, y: 3.0, w: 12.33, h: 0.9, fontSize: 44, bold: true, color: BLACK, align: "center" });
  txt(s, "B1–B6 follow; shown on demand.", { x: 0.5, y: 4.0, w: 12.33, h: 0.35, fontSize: 14, color: GRAY, align: "center" });
  footer(s, `9 / ${TOTAL}`, "");
  s.addNotes("Divider. Backup pages B1–B6 follow.");
}

/* ============ Slide 10 | B1 protocol ============ */
{
  const s = pres.addSlide();
  title(s, "B1 | The four-component evaluation protocol and the main-table protocol");
  const blk = (x, y, head, body) => {
    noteBox(s, x, y, 5.99, 1.92);
    txt(s, head, { x: x + 0.18, y: y + 0.1, w: 5.6, h: 0.28, fontSize: 14, bold: true, color: BLACK });
    txt(s, body, { x: x + 0.18, y: y + 0.44, w: 5.66, h: 1.4, fontSize: 11.5, lineSpacingMultiple: 1.1 });
  };
  blk(0.5, 1.4, "① q150 quality six-dim", "Official VBench all_dimension suite, deterministic sample of 150 prompts (md5 690f2919), custom-input mode. Sweeps at seed 0; champions re-checked at n=3 (seeds 0/1/2).");
  blk(6.84, 1.4, "② dm40 clean dynamic degree (the citable DD)", "40 motion-oriented prompts (20 from official human_action.txt, uniform stride + 20 from all_dimension.txt via a MOTION_CUE regex excluding STATIC_BLOCK; md5 324d75a0). q150-DD is confounded by still-style prompts — footnote-level only (teacher DD across the two domains: 0.300 vs 0.625).");
  blk(0.5, 3.5, "③ d40×8 cross-seed diversity", "40 prompts × 8 seeds; mean pairwise LPIPS-alex (8 frames @256 px; md5 b4c1f9e3; higher = more diverse) — the measurement of this project's main degradation axis.");
  blk(6.84, 3.5, "④ RAFT continuous optical flow (motion magnitude)", "dm40 domain, px/frame; median primary, mean also reported (teacher is heavy-tailed: median 2.75 / mean 5.16). Multi-seed discipline: directions are reported seed-paired; single-seed percentages are never cited alone. Motivation: binary DD saturates for good students (0.75–1.0).");
  noteBox(s, 0.5, 5.6, 12.33, 1.28);
  txt(s, "Main-table protocol: full VBench standard mode (946 prompts × 5 seeds), 12/16 dimensions (4 GRiT dims missing; Semantic/Total not composable — declared). temporal_flickering protocol difference footnoted.\nGeneral rules: numbers from q150 / dm40 / vb946 are never mixed across tables; training-health metrics (loss) are not quality evidence; one variable per experiment; checkpoints always best-of-sweep.", { x: 0.78, y: 5.74, w: 11.8, h: 1.05, fontSize: 12, lineSpacingMultiple: 1.12 });
  footer(s, `10 / ${TOTAL} · B1`, "Source: research/thesis_ch1_draft.md §1.7");
  s.addNotes("Open when asked for protocol details. md5s and sampling criteria are in the repo (exp/eval/, make_motion_set.py header; re-verified against the remote on 2026-07-26).");
}

/* ============ Slide 11 | B2 E5 probe ============ */
{
  const s = pres.addSlide();
  title(s, "B2 | E5 layer×t separability probe (observational; not a headline claim)");
  const rows = [
    [H("noise level t", { fontSize: 12 }), H("real-vs-generated linear-probe AUC (all 9 layers)", { fontSize: 12 })],
    [C("t = 0.999 (highest noise)", { align: "center" }), C("0.28–0.52 (random or below)", { align: "center" })],
    [C("t ≤ 0.937 (all remaining levels)", { align: "center" }), C("1.0 (all layers saturated, n=64)", { align: "center", bold: true })],
  ];
  const RH = [0.42, 0.44, 0.44];
  s.addTable(rows, { x: 0.7, y: 1.45, w: 7.6, colW: [3.0, 4.6], rowH: RH, border: { type: "none" }, fontFace: FONT, valign: "middle" });
  threeLineRules(s, 0.7, 7.6, 1.45, RH, [0]);
  s.addText([
    { text: "Cliff-shaped t dependence: ", options: { bold: true } },
    { text: "the discriminator's usable supervision concentrates at low-to-mid noise; the highest-noise level carries almost no signal — mechanistic background for the (t,ε) result on P5 (the pairing acts exactly where the signal is weakest).", options: {} },
  ], { x: 0.7, y: 3.1, w: 12.0, h: 0.8, fontFace: FONT, fontSize: 13.5, color: INK, margin: 0, lineSpacingMultiple: 1.12 });
  s.addText([
    { text: "Fair verdict on layer choice: ", options: { bold: true } },
    { text: "AUC saturates for t ≤ 0.937 (no between-layer resolution; the early “mean AUC 0.88–0.92 / L7 slightly better” reading was a saturation-averaging artifact, discarded). The continuous Fréchet distance rises gently with depth and jumps at L27/29. Upstream's {15, 22, 29}: no evidence it is a bad choice, and no evidence it is uniquely optimal.", options: {} },
  ], { x: 0.7, y: 4.0, w: 12.0, h: 1.05, fontFace: FONT, fontSize: 13.5, color: INK, margin: 0, lineSpacingMultiple: 1.12 });
  s.addText([
    { text: "Honesty control (teachergen): ", options: { bold: true } },
    { text: "the teacher's own 50-step outputs separate from real data in the same cliff shape (with larger FD) → feature separability mainly reflects generated-vs-real domain gaps plus prompt-domain differences; it is not a direct measure of distillation degradation.", options: {} },
  ], { x: 0.7, y: 5.15, w: 12.0, h: 0.95, fontFace: FONT, fontSize: 13.5, color: INK, margin: 0, lineSpacingMultiple: 1.12 });
  txt(s, "Protocol: 64 clips/side; feature path field-matched to the training-side discriminator; null-text conditioning; 5-fold linear probe AUC + Fréchet distance. Status: to-our-knowledge observational evidence.", { x: 0.7, y: 6.2, w: 12.0, h: 0.6, fontSize: 12, color: GRAY });
  footer(s, `11 / ${TOTAL} · B2`, "Source: research/E5_probe_results.md (corrected reading — authoritative)");
  s.addNotes("Open when asked about discriminator mechanism / E5. Validity frame: n=64/side, single batch, null-text conditioning, no resolution where AUC saturates, FD sample-size sensitive.");
}

/* ============ Slide 12 | B3 design details ============ */
{
  const s = pres.addSlide();
  title(s, "B3 | Relay-vs-direct controlled comparison: design details");
  const row = (y, head, body) => {
    txt(s, head, { x: 0.7, y, w: 2.3, h: 0.6, fontSize: 14, bold: true, color: BLACK });
    txt(s, body, { x: 3.2, y, w: 9.6, h: 0.95, fontSize: 13, lineSpacingMultiple: 1.12 });
  };
  row(1.5, "Relay arm", "W5 (8-step, LR 1e-5 / batch 12, 2500 it, from the teacher) → W7 (4-step, LR 5e-6 / batch 16, 2500 it; inherits only W5@2500 generator weights — optimizer / fake score / discriminator all re-initialized). Total budget 5000 it.");
  row(2.62, "Direct arms", "E1a = relay stage-2 recipe (LR 5e-6 / batch 16); E1b = stage-1 recipe (LR 1e-5 / batch 12 — exactly the upstream FastGen default LR); 5000 it each, from the teacher. The two-arm bracket forestalls the “untuned direct baseline” objection.");
  row(3.78, "Invariants", "Data (OpenVid-1M) · 4-step t_list · discriminator architecture · all single-stage hyper-parameters (upstream published values) · evaluation protocol · checkpoint granularity (every 500 it).");
  row(4.8, "Selection", "All arms best-of-sweep (E1a/E1b: 10 checkpoints each; W7: 5; the 32-row table has no missing cells); champions re-checked at n=3 seeds.");
  noteBox(s, 0.7, 5.85, 12.0, 0.95);
  txt(s, "Pre-registration discipline: before launch, the leading hypothesis was logged as “parity”. The observed result — direct slightly ahead — is reported exactly as pre-registered, with no after-the-fact adjustment.", { x: 0.95, y: 6.02, w: 11.5, h: 0.65, fontSize: 13, bold: true, color: BLACK });
  footer(s, `12 / ${TOTAL} · B3`, "Source: research/thesis_ch2_draft.md §2.1");
  s.addNotes("Open when asked why the comparison is trustworthy / which objections were pre-empted.");
}

/* ============ Slide 13 | B4 Quality Score ============ */
{
  const s = pres.addSlide();
  title(s, "B4 | Quality Score synthesis and its blind spot");
  txt(s, "Official weights (Vchitect/VBench scripts/constant.py + cal_final_score.py @master): per-dimension min-max normalization; dynamic_degree weight 0.5, the other six weight 1; weighted sum ÷ 6.5.", { x: 0.6, y: 1.4, w: 12.1, h: 0.55, fontSize: 12.5, color: INK });
  const rows = [
    [H("Model", { align: "left", fontSize: 12 }), H("Quality Score (7-dim)", { fontSize: 12 }), H("Remark", { align: "left", fontSize: 12 })],
    [C("E2a@2000 (audit arm GAN=0, single ckpt)"), C("85.50", { align: "center", bold: true }), C("lowest diversity band of all arms (0.586–0.604); dynamics back at teacher level", { fontSize: 11 })],
    [C("W7@1000 (relay)"), C("84.47", { align: "center" }), C("numeric coincidence with CoDMD's 84.46 — never side by side (below)", { fontSize: 11 })],
    [C("E1b@500 (direct-B)"), C("83.62", { align: "center" }), C("high dynamics, low aesthetics", { fontSize: 11 })],
    [C("E1a@1000 (final model, G2)"), C("82.80", { align: "center" }), C("dynamic degree 0.5806 enters at 0.5 weight; its winning dims lose resolution after normalization", { fontSize: 11 })],
  ];
  const RH = [0.4, 0.5, 0.5, 0.4, 0.56];
  s.addTable(rows, { x: 0.6, y: 2.05, w: 8.5, colW: [3.2, 1.9, 3.4], rowH: RH, border: { type: "none" }, fontFace: FONT, valign: "middle" });
  threeLineRules(s, 0.6, 8.5, 2.05, RH, [0]);
  noteBox(s, 0.6, 4.6, 8.5, 1.15);
  txt(s, "Blind spot: the composite has no diversity term — it cannot see this project's main degradation axis (the lowest-diversity model ranks first; the final model ranks last — the W4 lesson at benchmark scale, and exactly why our own protocol exists). Model selection follows the pre-registered G2 protocol, not this composite.", { x: 0.82, y: 4.72, w: 8.05, h: 0.95, fontSize: 12, lineSpacingMultiple: 1.1 });
  s.addShape(RECT, { x: 0.6, y: 5.95, w: 8.5, h: 0.9, fill: { color: "FFFFFF" }, line: { color: BLACK, width: 1.5 } });
  txt(s, "Warning: W7's 84.47 vs the published CoDMD 84.46 is a numeric coincidence — a 7-dimension quality composite vs a 16-dimension total. Never place the two side by side.", { x: 0.82, y: 6.08, w: 8.05, h: 0.65, fontSize: 12.5, bold: true, color: BLACK });
  txt(s, "Demo clips (same prompt & seed, cyclist)", { x: 9.4, y: 2.05, w: 3.4, h: 0.26, fontSize: 12, bold: true, color: BLACK });
  video(s, "backup/backup_e2a_cyclist.mp4", 9.4, 2.36, 3.15, 1.817);
  txt(s, "Audit arm E2a@2000 (GAN=0): high static quality, dynamics fall back", { x: 9.4, y: 4.2, w: 3.4, h: 0.42, fontSize: 10.5, color: GRAY });
  video(s, "backup/backup_w7_cyclist.mp4", 9.4, 4.72, 3.15, 1.817);
  txt(s, "Relay W7@1000 (paired GAN): large dynamics, quality declines over iterations", { x: 9.4, y: 6.56, w: 3.4, h: 0.42, fontSize: 10.5, color: GRAY });
  footer(s, `13 / ${TOTAL} · B4`, "Source: experiments/results/2026-07-26-e2a-vb946-fourth-row.md (QS section)");
  s.addNotes("Use with oral answers #6/#7: E1a lowest via weight structure; E2a highest yet lowest-diversity band and teacher-level dynamics — the two things the composite cannot see; pre-registration forbids switching.");
}

/* ============ Slide 14 | B5 flow table ============ */
{
  const s = pres.addSlide();
  title(s, "B5 | Motion magnitude, multi-seed full table (RAFT median, dm40)");
  const rows = [
    [H("seed", { fontSize: 12 }), H("Relay W7@1000", { fontSize: 12 }), H("Direct E1a@1000", { fontSize: 12 }), H("teacher 50-step", { fontSize: 12 })],
    [C("s0", { align: "center" }), C("4.44", { align: "center" }), C("2.15", { align: "center" }), C("2.75", { align: "center" })],
    [C("s1", { align: "center" }), C("3.28", { align: "center" }), C("1.83", { align: "center" }), C("2.17", { align: "center" })],
    [C("s2", { align: "center" }), C("1.27", { align: "center" }), C("0.46", { align: "center" }), C("0.86", { align: "center" })],
    [C("s3", { align: "center" }), C("4.44", { align: "center" }), C("2.80", { align: "center" }), C("2.41", { align: "center" })],
    [C("4-seed mean", { align: "center", bold: true }), C("3.36", { align: "center", bold: true }), C("1.81", { align: "center", bold: true }), C("2.05", { align: "center", bold: true })],
  ];
  const RH = [0.42, 0.4, 0.4, 0.4, 0.4, 0.42];
  s.addTable(rows, { x: 0.9, y: 1.5, w: 8.0, colW: [1.7, 2.1, 2.1, 2.1], rowH: RH, border: { type: "none" }, fontFace: FONT, valign: "middle" });
  threeLineRules(s, 0.9, 8.0, 1.5, RH, [0, 4]);
  s.addText([
    { text: "Seed-paired conclusions:", options: { bold: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "W7 > E1a: 4/4 seeds, same direction (≈1.9×)", options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "W7 > teacher: 4/4 seeds (by means, +64%)", options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "E1a not above the teacher (3/4 seeds; by means, −12%)", options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "seed 2 is low for all three models: the initial noise dominates part of the dynamics level — empirical case for seed-paired design", options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 7 } },
    { text: "Discipline: single-seed absolute percentages are never cited alone (per-seed medians can differ by 6×)", options: { bullet: { code: "2022", indent: 12 }, breakLine: true } },
  ], { x: 0.9, y: 4.5, w: 11.6, h: 2.3, fontFace: FONT, fontSize: 13.5, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.15 });
  footer(s, `14 / ${TOTAL} · B5`, "Source: experiments/results/2026-07-23-flow-multiseed-e1b946-e2a-eval.md §1/§4");
  s.addNotes("Open when asked how strong the motion-magnitude evidence is; the main slide P4 reports only 4-seed means and directions.");
}

/* ============ Slide 15 | B6 upstream ============ */
{
  const s = pres.addSlide();
  title(s, "B6 | Relationship to upstream NVIDIA FastGen (statement)");
  noteBox(s, 0.7, 1.45, 12.0, 4.15);
  txt(s, "All training builds on NVIDIA FastGen (NVlabs/FastGen, Apache-2.0), reusing its native DMD2 implementation and the official Wan2.1-T2V-1.3B configuration — teacher CFG=5; generator-side GAN weight 0.03; shared real/fake timestep and noise (gan_use_same_t_noise=True is the factory default); the multiscale MLP discriminator on frozen-teacher features from layers 15/22/29; two-time-scale updates with student_update_freq=5; and the 4-step t_list=[0.999, 0.937, 0.833, 0.624, 0.0].\n\nOn top of this, our recipe contribution is confined to the training-schedule layer: the official repository ships only single-stage 50→4/2-step configurations; we run a 50→8→4 step-count relay with an added 8-step intermediate student, where the 4-step stage inherits only the generator weights of the best 8-step checkpoint (optimizer / fake score / discriminator all re-initialized). On the data side we use OpenVid-1M (upstream is dataset-agnostic).", { x: 1.0, y: 1.7, w: 11.4, h: 3.7, fontSize: 13.5, lineSpacingMultiple: 1.2 });
  txt(s, "Further contributions of this report: the controlled audits (relay comparison, three-arm discriminator ablation) and the portable degradation-evaluation protocol. Discriminator wording, everywhere: trainable multiscale MLP heads on frozen-teacher features, layers 15/22/29 (code-verified 2026-07-06).", { x: 0.7, y: 5.8, w: 12.0, h: 0.85, fontSize: 12.5, color: INK });
  footer(s, `15 / ${TOTAL} · B6`, "Source: research/T3_novelty_adjudication.md §6.2");
  s.addNotes("Show verbatim when asked what is ours vs upstream. Never say we 'improved' FastGen; never present single-stage hyper-parameters as our design.");
}

const OUT = QA_IMAGES ? "DMD2_report_0728_en_qa.pptx" : "DMD2_report_0728_en.pptx";
fs.writeFileSync(path.join(__dirname, "qa", "ops.json"), JSON.stringify(QA_LOG));
pres.writeFile({ fileName: path.join(__dirname, OUT) }).then(() => console.log("deck written: " + OUT + " | slides: " + QA_LOG.length));
