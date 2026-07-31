// DMD2 progress report deck — ENGLISH V2.3 port (2026-07-27) of slides/build_deck_v2.js (Chinese final).
// Same design system & geometry as the Chinese deck; all text in English; main slides carry no internal
// codenames (research/presentation_wording_guide.md). Numbers frozen at commit 0c28815 — transcribed verbatim.
// Build: node build_deck_en_v2.js -> DMD2_report_0728_en_V2.pptx  (QA_IMAGES=1 -> poster-image variant)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ASSETS = path.resolve(__dirname, "../slides_assets");
const COVERS = path.resolve(__dirname, "../slides/covers");

const FONT = "Helvetica Neue"; // Windows fallback: change to "Arial" and rebuild
const NAVY = "1F3864";
const NAVY_SOFT = "8FAADC";
const ORANGE = "C55A11";
const ORANGE_TINT = "FBE5D6"; // P7 row-best hot zone fill
const ORANGE_BAND = "FDF1E5"; // P2 quantitative-optimum interval band
const TEAL = "00796B"; // GAN-off ablation line — orange stays reserved for degradation/warnings
const DARKRED = "8A3324"; // hard-warning boxes (B4)
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

// --- QA instrumentation: record every element op for the HTML mirror renderer ---
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
const TOTAL = "15";

function cover(name) {
  return "image/png;base64," + fs.readFileSync(path.join(COVERS, name + ".png")).toString("base64");
}
const QA_IMAGES = !!process.env.QA_IMAGES;
function video(s, rel, x, y, w, h) {
  const base = path.basename(rel, ".mp4");
  if (QA_IMAGES) {
    s.addImage({ path: path.join(COVERS, base + ".png"), x, y, w, h });
  } else {
    s.addMedia({ type: "video", path: path.join(ASSETS, rel), x, y, w, h, cover: cover(base) });
  }
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ fontFace: FONT, color: INK, margin: 0, valign: "top" }, o));
}
function footer(s, suffix, src) {
  s.addText(`${pageNo} / ${TOTAL}${suffix}`, { x: 0.4, y: 7.16, w: 2.6, h: 0.26, fontFace: FONT, fontSize: 9, color: GRAY, margin: 0 });
  s.addText(src, { x: 4.2, y: 7.16, w: 8.73, h: 0.26, fontFace: FONT, fontSize: 8, color: GRAY, align: "right", margin: 0 });
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
function anchorBar(s, y) {
  s.addShape(RECT, { x: 0.5, y, w: 0.6, h: 0.055, fill: { color: NAVY }, line: { type: "none" } });
}
function progressDots(s, idx) {
  for (let i = 0; i < 8; i++) dot(s, 11.77 + i * 0.155, 0.23, 0.038, i === idx ? NAVY : BORDER);
}
function header(s, o) {
  txt(s, o.kicker, { x: 0.5, y: 0.13, w: 8, h: 0.2, fontSize: 9.5, bold: true, color: GRAY, charSpacing: 1.5 });
  progressDots(s, o.idx);
  txt(s, o.main, { x: 0.5, y: 0.36, w: 12.33, h: 0.5, fontSize: 26, bold: true, color: NAVY });
  if (o.sub) txt(s, o.sub, { x: 0.5, y: 0.9, w: 12.33, h: 0.3, fontSize: 14, color: GRAY });
  anchorBar(s, 1.26);
}
function title(s, text) {
  s.addText(text, { x: 0.5, y: 0.2, w: 12.33, h: 0.98, fontFace: FONT, fontSize: 26, bold: true, color: NAVY, valign: "top", margin: 0 });
  anchorBar(s, 0.82);
}
function chip(s, x, y, w, h, text, o) {
  s.addShape(RR, { x, y, w, h, rectRadius: 0.05, fill: { color: (o && o.fill) || "FFFFFF" }, line: { color: (o && o.lineColor) || NAVY, width: 1 } });
  txt(s, text, { x, y, w, h, fontSize: (o && o.fontSize) || 8, bold: !!(o && o.bold), color: (o && o.color) || INK, align: "center", valign: "middle" });
}
function C(text, o) { return { text: String(text), options: Object.assign({ fontFace: FONT, fontSize: 10, color: INK, valign: "middle" }, o || {}) }; }
function BU(text, o) { return C(text, Object.assign({ bold: true, underline: true }, o || {})); }
function HC(text, o) { return C(text, Object.assign({ bold: true, color: "FFFFFF", fill: { color: NAVY }, fontSize: 9.5, align: "center" }, o || {})); }
function MARK(text, o) {
  const runs = [{ text: String(text), options: Object.assign({ fontFace: FONT }, o || {}) }, { text: " ▲", options: { fontFace: FONT, color: ORANGE, bold: true } }];
  return { text: runs, options: { fontFace: FONT, fontSize: 10.5, color: INK, valign: "middle", align: "right" } };
}

/* ============================================================ S1 | P0 formal cover */
{
  const s = pres.addSlide(); pageNo++;
  txt(s, "Final Presentation · 2026/07/28", { x: 0.9, y: 1.7, w: 8, h: 0.28, fontSize: 13, bold: true, color: GRAY, charSpacing: 2.5 });
  s.addShape(RECT, { x: 0.9, y: 2.16, w: 1.0, h: 0.07, fill: { color: NAVY }, line: { type: "none" } });
  txt(s, "Wan2.1-T2V Few-Step Distillation", { x: 0.9, y: 2.44, w: 11.5, h: 0.78, fontSize: 40, bold: true, color: NAVY });
  s.addText([
    { text: "25×", options: { color: ORANGE } },
    { text: " Speed-up: Quality-Degradation Diagnosis & Attribution", options: { color: NAVY } },
  ], { x: 0.9, y: 3.34, w: 11.5, h: 0.52, fontFace: FONT, fontSize: 24, bold: true, margin: 0 });
  txt(s, "With DMD2 we distill the 50-step Wan2.1-T2V-1.3B teacher into a 4-step student (165.24 s → ≈6.6 s per clip), and deliver a systematic, controlled diagnosis and attribution of the quality degradation.", { x: 0.9, y: 4.42, w: 11.2, h: 0.66, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2 });
  txt(s, "Presenter", { x: 0.9, y: 5.5, w: 4, h: 0.2, fontSize: 10.5, color: GRAY });
  txt(s, "Chen Hing Chin (21205180)", { x: 0.9, y: 5.72, w: 8, h: 0.34, fontSize: 16, bold: true, color: INK });
  txt(s, "PVTT Task 3 · DMD Distillation & Acceleration", { x: 0.9, y: 6.12, w: 8, h: 0.22, fontSize: 10.5, color: GRAY });
  footer(s, "", "Source: speed figures, canonical experiment record §W1 (re-read from metrics.csv, 2026-07-26)");
  s.addNotes("Opening (10 s): title, name and student ID; click to the next slide for the same-prompt comparison. Date note: the project milestone is the 2026-07-28 presentation; the earlier '2026/6/28' input was ruled a typo — edit the first line of this slide if it ever needs changing.");
}

/* ============================================================ S2 | P1 opening */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 0, kicker: "OPENING | TASK & QUESTION",
    main: "At 25× speed-up, the outputs look close at first glance",
    sub: "Same prompt: teacher, 50 steps + CFG vs the 4-step student (no CFG); this report delivers a quantitative diagnosis and attribution",
  });
  const vw = 5.11, vh = 2.95;
  video(s, "p1/p1_teacher_sportscar.mp4", 1.31, 1.42, vw, vh);
  video(s, "p1/p1_e1a_sportscar_s1.mp4", 6.92, 1.42, vw, vh);
  txt(s, "teacher: 50 steps + CFG (≈100 NFE)\n165.24 s per clip", { x: 1.31, y: 4.41, w: vw, h: 0.52, fontSize: 11, align: "center", color: GRAY });
  txt(s, "4-step student (random seed 1): no CFG (4 NFE)\n≈6.6 s per clip", { x: 6.92, y: 4.41, w: vw, h: 0.52, fontSize: 11, align: "center", color: GRAY });
  s.addText([
    { text: "165.24 s", options: { bold: true, color: NAVY } },
    { text: "  →  ", options: { color: GRAY } },
    { text: "6.59–6.66 s", options: { bold: true, color: NAVY } },
    { text: "   ≈25×", options: { bold: true, color: ORANGE } },
  ], { x: 0.5, y: 5.05, w: 12.33, h: 0.68, fontFace: FONT, fontSize: 34, align: "center", margin: 0 });
  txt(s, "Task: distill the public Wan2.1-T2V-1.3B 50-step teacher into a 4-step student with DMD2; the framework and all hyper-parameters come from NVIDIA FastGen's public configuration. Our work: the 50→8→4 step-count-relay schedule, plus a systematic diagnosis and attribution of the degradation.", { x: 0.9, y: 5.85, w: 11.53, h: 0.62, fontSize: 13 });
  txt(s, "The question this report answers: after a 25× speed-up, what is lost — and where does the loss come from?", { x: 0.9, y: 6.5, w: 11.53, h: 0.4, fontSize: 14, bold: true, color: NAVY });
  footer(s, "", "Source: canonical experiment record §W1 (speed re-read from metrics.csv, 2026-07-26)");
  s.addNotes("75 s. Click to play both clips at once (same prompt: red sports car, rainy night street; cut away after ~5 s). Oral definitions: diffusion models are slow because they denoise over many steps; distillation = the teacher guides a student to generate in very few steps. Stress: the frames look close at first glance — this report quantifies what is actually lost. The upstream-attribution sentence is mandatory (recipe from NVIDIA FastGen's public config; our work is the schedule layer and the degradation audit). Red lines: method name is 'step-count relay' only; speed figures re-verified 2026-07-26 (teacher 165.24 exact; student full sweep 6.591–6.656, ≈25×).");
}

/* ============================================================ S3 | P2 protocol */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 1, kicker: "① QUANTITATIVE PROTOCOL",
    main: "Subjective checkpoint picking was overturned on both training lines",
    sub: "Every conclusion rests on the four-component quantitative protocol and best-of-sweep selection",
  });
  txt(s, "Relay student: aesthetic (150 standard prompts) declines monotonically, 0.5768 → 0.5379", { x: 0.55, y: 1.42, w: 7.1, h: 0.28, fontSize: 12, bold: true });
  txt(s, "Quantitative optimum: iters 500–1000", { x: 1.42, y: 1.74, w: 2.9, h: 0.24, fontSize: 10.5, bold: true, color: ORANGE });
  txt(s, "y-axis 0.535–0.582 (not zero-based)", { x: 4.55, y: 1.78, w: 2.4, h: 0.18, fontSize: 8, color: GRAY, align: "right" });
  const px = [1.35, 2.75, 4.15, 5.55, 6.95];
  const vals = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379];
  const y0 = 4.78, yTop = 2.04, f = 2.74 / 0.047;
  const py = vals.map((v) => y0 - (v - 0.535) * f);
  s.addShape(RECT, { x: 1.35, y: yTop, w: 1.4, h: y0 - yTop, fill: { color: ORANGE_BAND }, line: { type: "none" } });
  seg(s, 1.35, y0, 6.95, y0, { color: "AAAAAA", width: 1 });
  seg(s, 1.35, yTop, 1.35, y0, { color: "AAAAAA", width: 1 });
  [0.54, 0.55, 0.56, 0.57, 0.58].forEach((t) => {
    const yy = y0 - (t - 0.535) * f;
    txt(s, t.toFixed(2), { x: 0.72, y: yy - 0.09, w: 0.55, h: 0.18, fontSize: 8.5, color: GRAY, align: "right" });
    seg(s, 1.29, yy, 1.35, yy, { color: "AAAAAA", width: 1 });
  });
  ["500", "1000", "1500", "2000", "2500"].forEach((t, k) => txt(s, t, { x: px[k] - 0.35, y: y0 + 0.05, w: 0.7, h: 0.18, fontSize: 9, color: GRAY, align: "center" }));
  txt(s, "training iteration", { x: 1.35, y: 5.08, w: 5.6, h: 0.2, fontSize: 9.5, color: GRAY, align: "center" });
  polyline(s, px, py, { color: NAVY, width: 2.75 }, false, NAVY);
  px.forEach((x, i) => dot(s, x, py[i], 0.05, NAVY));
  txt(s, "0.5768", { x: 1.48, y: py[0] - 0.27, w: 0.75, h: 0.2, fontSize: 9, bold: true, color: NAVY });
  txt(s, "0.5379", { x: 6.24, y: py[4] + 0.11, w: 0.75, h: 0.2, fontSize: 9, bold: true, color: NAVY });
  s.addShape(ELL, { x: 6.95 - 0.085, y: py[4] - 0.085, w: 0.17, h: 0.17, fill: { type: "none" }, line: { color: ORANGE, width: 1.75 } });
  txt(s, "Subjective pick: iter 2500 (end of the decline)", { x: 4.15, y: 3.5, w: 2.5, h: 0.42, fontSize: 10.5, color: INK });
  seg(s, 5.95, 3.94, 6.84, py[4] - 0.12, { color: "AAAAAA", width: 1 });
  // right: W4 counter-example
  txt(s, "Counter-example: uniform-timestep ablation — positive control", { x: 7.7, y: 1.42, w: 5.13, h: 0.28, fontSize: 12, bold: true });
  s.addTable([
    [HC("Metric"), HC("W4"), HC("Rank in full table")],
    [C("subject / background consistency", { fontSize: 9.5 }), C("0.9745 / 0.9791", { align: "center" }), C("highest", { align: "center", bold: true })],
    [C("imaging quality", { fontSize: 9.5 }), C("0.2555", { align: "center" }), C("lowest", { align: "center", bold: true })],
    [C("cross-seed diversity", { fontSize: 9.5 }), C("0.4617", { align: "center" }), C("lowest", { align: "center", bold: true })],
  ], { x: 7.7, y: 1.8, w: 5.13, colW: [2.33, 1.4, 1.4], rowH: [0.34, 0.4, 0.4, 0.4], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  txt(s, "Consistency-type metrics alone would rank an almost-static, collapsed model as the best.", { x: 7.7, y: 3.54, w: 5.13, h: 0.6, fontSize: 11.5, bold: true, color: NAVY });
  txt(s, "Hence the joint-reading rule: consistency metrics are always read together with dynamic degree and diversity.", { x: 7.7, y: 4.22, w: 5.13, h: 0.4, fontSize: 9.5, color: GRAY });
  s.addShape(RR, { x: 0.5, y: 5.62, w: 12.33, h: 1.08, rectRadius: 0.06, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "This established the four-component protocol: six quality dimensions (150 standard prompts) / dynamic degree (40 motion prompts) / cross-seed diversity (40 prompts × 8 seeds) / continuous optical flow (RAFT) — details in Backup B1.\nCheckpoint policy: save every 500 iterations, score the full sweep, re-check the best checkpoint on 3 random seeds.", { x: 0.78, y: 5.74, w: 11.8, h: 0.9, fontSize: 12, lineSpacingMultiple: 1.18 });
  footer(s, "", "Source: experiments/results/2026-07-14-e0-full-table-g1.md");
  s.addNotes("60 s. The two training lines err in opposite directions (the relay student was picked at 2500 — actually the end of the decline; the weak direct run was picked at 1000 while the quantitative optimum @1500 wins on 5/6 quality dims) — a protocol failure, not a one-off. W4 (uniform timesteps) shows the systematic blind spot of common metrics. Plant the hook (mandatory): 'quality peaks early in training then declines — the mechanism is answered on the discriminator-audit slide (④).' Red lines: do not claim novelty of the timestep ablation (TMD exists); consistency metrics are always read jointly with dynamics/diversity.");
}

/* ============================================================ S4 | P3 core finding */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 2, kicker: "② MAIN DEGRADATION AXIS",
    main: "The main degradation axis is cross-seed diversity",
    sub: "teacher 0.732 → students 0.59–0.64; dynamic degree does not drop",
  });
  txt(s, "Same prompt (“An astronaut walking slowly across a dusty red desert…”, in-domain prompt): only the random seed differs; thumbnails = frame 40, click to play", { x: 0.5, y: 1.32, w: 12.33, h: 0.2, fontSize: 9.5, color: GRAY });
  const cw = 1.314, ch = 0.758, gap = 0.028, x0 = 2.22;
  const tSeeds = [0, 1, 2, 3, 4, 5, 6, 7];
  const eSeeds = [0, 2, 3, 4, 6, 8, 9, 11];
  txt(s, "teacher · 50 steps", { x: 0.4, y: 1.58, w: 1.7, h: 0.22, fontSize: 10, bold: true });
  txt(s, "165.24 s/clip", { x: 0.4, y: 1.8, w: 1.7, h: 0.16, fontSize: 7.5, color: GRAY });
  txt(s, "8 seeds, 8 compositions", { x: 0.4, y: 2.0, w: 1.7, h: 0.36, fontSize: 9, bold: true, color: NAVY });
  txt(s, "student (4-step direct)", { x: 0.4, y: 2.59, w: 1.7, h: 0.22, fontSize: 10, bold: true });
  txt(s, "≈6.6 s/clip", { x: 0.4, y: 2.81, w: 1.7, h: 0.16, fontSize: 7.5, color: GRAY });
  txt(s, "8 seeds, one shared template", { x: 0.4, y: 3.01, w: 1.7, h: 0.36, fontSize: 9, bold: true, color: ORANGE });
  tSeeds.forEach((sd, k) => {
    const x = x0 + k * (cw + gap);
    video(s, `p3/p3v2_teacher_s${sd}.mp4`, x, 1.54, cw, ch);
    txt(s, `s${sd}`, { x, y: 2.31, w: cw, h: 0.14, fontSize: 8, color: GRAY, align: "center" });
  });
  eSeeds.forEach((sd, k) => {
    const x = x0 + k * (cw + gap);
    video(s, `p3/p3v2_e1a_s${sd}.mp4`, x, 2.55, cw, ch);
    txt(s, `s${sd}`, { x, y: 3.32, w: cw, h: 0.14, fontSize: 8, color: GRAY, align: "center" });
  });
  const barX = 2.0, barW = 4.1, barX2 = 8.38;
  function hbar(x, y, len, color) { s.addShape(RECT, { x, y, w: Math.max(len, 0.02), h: 0.36, fill: { color }, line: { type: "none" } }); }
  // panel A — dynamic degree, full 0-1 scale
  txt(s, "Dynamic degree (40 motion prompts): no drop", { x: 0.5, y: 3.62, w: 5.95, h: 0.26, fontSize: 12, bold: true });
  txt(s, "teacher", { x: 0.55, y: 4.02, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  txt(s, "students", { x: 0.55, y: 4.62, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  seg(s, barX, 3.96, barX, 5.24, { color: "AAAAAA", width: 1 });
  hbar(barX, 4.0, 0.625 * barW, NAVY);
  txt(s, "0.625", { x: barX + 0.625 * barW + 0.06, y: 4.06, w: 0.7, h: 0.2, fontSize: 9.5, bold: true, color: NAVY });
  hbar(barX + 0.75 * barW, 4.6, 0.25 * barW, NAVY_SOFT);
  txt(s, "0.75–1.0 (all students)", { x: 4.84, y: 4.37, w: 1.5, h: 0.2, fontSize: 9.5, bold: true, align: "center" });
  txt(s, "up, not down ↑", { x: 4.99, y: 5.0, w: 1.2, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE, align: "center" });
  [0, 0.5, 1.0].forEach((t) => txt(s, String(t), { x: barX + t * barW - 0.2, y: 5.26, w: 0.4, h: 0.16, fontSize: 8, color: GRAY, align: "center" }));
  txt(s, "Motion smoothness 0.97+: real motion, not jitter; the dynamics collapse feared in the literature is not reproduced here (different domain)", { x: 0.5, y: 5.5, w: 5.95, h: 0.4, fontSize: 9.5, color: GRAY });
  // panel B — diversity, zoomed 0.4-0.8 scale (declared)
  const posB = (v) => barX2 + ((v - 0.4) / 0.4) * barW;
  txt(s, "Cross-seed diversity (LPIPS): consistent drop — the main axis", { x: 6.88, y: 3.62, w: 5.95, h: 0.26, fontSize: 12, bold: true, color: ORANGE });
  txt(s, "teacher", { x: 6.93, y: 4.02, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  txt(s, "students", { x: 6.93, y: 4.62, w: 1.35, h: 0.3, fontSize: 10, align: "right" });
  seg(s, barX2, 3.96, barX2, 5.24, { color: "AAAAAA", width: 1 });
  hbar(barX2, 4.0, posB(0.732) - barX2, NAVY);
  txt(s, "▼", { x: posB(0.732) - 0.125, y: 3.86, w: 0.25, h: 0.18, fontSize: 8, color: NAVY, align: "center" });
  txt(s, "0.732", { x: posB(0.732) + 0.07, y: 4.06, w: 0.7, h: 0.2, fontSize: 9.5, bold: true, color: NAVY });
  hbar(posB(0.59), 4.6, posB(0.64) - posB(0.59), ORANGE);
  txt(s, "all students (best ckpts)", { x: posB(0.59) - 0.65, y: 4.37, w: 1.8, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE, align: "center" });
  txt(s, "0.59", { x: posB(0.59) - 0.62, y: 4.68, w: 0.56, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE, align: "right" });
  txt(s, "0.64", { x: posB(0.64) + 0.06, y: 4.68, w: 0.6, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE });
  seg(s, posB(0.59), 4.96, posB(0.59), 5.08, { color: ORANGE, width: 1 });
  seg(s, posB(0.64), 4.96, posB(0.64), 5.08, { color: ORANGE, width: 1 });
  [0.4, 0.6, 0.8].forEach((t) => txt(s, t.toFixed(1), { x: posB(t) - 0.2, y: 5.26, w: 0.4, h: 0.16, fontSize: 8, color: GRAY, align: "center" }));
  txt(s, "x-axis 0.4–0.8 (not zero-based). ≈ −13% to −19% vs the teacher (from this panel); weak/strong recipes, direct and relay — all one direction. The most-replicated finding of this project", { x: 6.88, y: 5.5, w: 5.95, h: 0.4, fontSize: 9.5, color: GRAY });
  txt(s, "Diversity = mean pairwise perceptual distance (LPIPS) among 8 videos from the same prompt with different random seeds (higher = more diverse). The two rows above visualize it: the teacher varies across seeds; the student converges to one template.", { x: 0.5, y: 6.05, w: 12.33, h: 0.55, fontSize: 10.5, color: INK });
  footer(s, "", "Source: E0 full table (07-14) · G2 final table (07-20)");
  s.addNotes("75 s. The literature worries that few-step distillation loses dynamics; our measurement: dynamics do not drop, and motion smoothness 0.97+ rules out jitter (different experimental domain — phrase as 'not reproduced', never as a refutation). What drops consistently is cross-seed diversity — weak and strong recipes, direct and relay, all one direction; the most-replicated finding of this project. One-sentence diversity definition at the bottom of the slide. Wall: teacher row seeds 0–7; student row seeds 0/2/3/4/6/8/9/11, labeled as-is. Fixed transition: 'This degradation appears in every configuration — two controlled ablations now locate its source.' Percentage ruling (2026-07-27): no percentage in the headline; body text uses the exact panel-basis '≈ −13% to −19%'.\n\nSeed fallback answer (ruling 2026-07-27, no on-slide disclosure): a few seeds produced corrupted clips and were replaced by the next available seeds; the quantitative diversity conclusion rests on the full 8-seed protocol and is unaffected by the demo material (MANIFEST on file).\n\nFull sources: experiments/results/2026-07-14-e0-full-table-g1.md, 2026-07-20-g2-relay-vs-direct-final.md.");
}

/* ============================================================ S5 | P4 ablation I */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 3, kicker: "③ CONTROLLED COMPARISON | RELAY VS DIRECT",
    main: "The relay adds no quality gain and lowers diversity",
    sub: "Budget-matched controlled comparison (G2); its only stable effect is higher motion magnitude",
  });
  const RN = (v, o) => C(v, Object.assign({ align: "right", fontSize: 10.5 }, o || {}));
  s.addTable([
    [HC("Model (best-of-sweep)", { align: "left" }), HC("aesthetic"), HC("imaging"), HC("DD_clean"), HC("diversity")],
    [C("teacher 50-step CFG5", { fontSize: 10 }), BU("0.590", { align: "right", fontSize: 10.5 }), RN("0.692"), RN("0.625"), BU("0.732", { align: "right", fontSize: 10.5 })],
    [C("Direct student @1000 (final model)", { fontSize: 10 }), RN("0.567"), MARK("0.717", { bold: true, underline: true }), MARK("0.750"), RN("0.635")],
    [C("Relay student @500", { fontSize: 10 }), RN("0.577"), RN("0.694"), RN("0.825"), RN("0.598")],
    [C("Relay student @1000", { fontSize: 10 }), RN("0.559"), RN("0.697"), BU("1.000", { align: "right", fontSize: 10.5 }), RN("0.613")],
    [C("Direct student B @500 (high LR)", { fontSize: 10 }), RN("0.532"), RN("0.695"), RN("0.975"), RN("0.628")],
  ], { x: 0.5, y: 1.42, w: 7.35, colW: [2.5, 1.1, 1.15, 1.35, 1.25], rowH: [0.42, 0.4, 0.4, 0.4, 0.4, 0.4], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  txt(s, "Protocol: aesthetic/imaging = 150 prompts; DD = 40 motion prompts; diversity = 40 prompts × 8 seeds (LPIPS); best = bold underline", { x: 0.5, y: 3.92, w: 7.35, h: 0.2, fontSize: 8.5, color: GRAY });
  s.addText([
    { text: "▲ ", options: { color: ORANGE, bold: true } },
    { text: "scope: 0.717 above teacher = sharpness bias, not “surpassing”; 0.750 below other students (0.825–1.0), as-is", options: { color: INK } },
  ], { x: 0.5, y: 4.15, w: 7.35, h: 0.26, fontFace: FONT, fontSize: 10.5, margin: 0 });
  txt(s, "Same prompt & seed (cyclist): the motion-magnitude gap, directly visible", { x: 0.6, y: 4.52, w: 6.1, h: 0.2, fontSize: 8.5, color: GRAY });
  video(s, "p4/p4_w7_cyclist.mp4", 0.6, 4.74, 2.78, 1.604);
  video(s, "p4/p4_e1a_cyclist.mp4", 3.72, 4.74, 2.78, 1.604);
  txt(s, "Relay student (4 steps)", { x: 0.6, y: 6.4, w: 2.78, h: 0.2, fontSize: 9, align: "center", color: GRAY });
  txt(s, "Direct student (4 steps)", { x: 3.72, y: 6.4, w: 2.78, h: 0.2, fontSize: 9, align: "center", color: GRAY });
  // right-top design box: fork schematic + invariants + badges
  s.addShape(RR, { x: 8.1, y: 1.42, w: 4.73, h: 2.28, rectRadius: 0.06, fill: { color: PANEL }, line: { color: NAVY, width: 1 } });
  txt(s, "Controlled design (pre-registered)", { x: 8.3, y: 1.54, w: 4.35, h: 0.26, fontSize: 12, bold: true, color: NAVY });
  chip(s, 8.3, 2.06, 1.02, 0.32, "teacher 50-step");
  seg(s, 9.36, 2.14, 9.66, 2.01, { color: GRAY, width: 1.25, endArrowType: "triangle" });
  chip(s, 9.68, 1.86, 0.94, 0.3, "8-step interm.");
  seg(s, 10.64, 2.01, 10.98, 2.01, { color: GRAY, width: 1.25, endArrowType: "triangle" });
  txt(s, "full reset ↺", { x: 10.44, y: 2.19, w: 0.75, h: 0.15, fontSize: 7.5, bold: true, color: ORANGE, align: "center" });
  chip(s, 11.0, 1.86, 0.86, 0.3, "4-step (relay)");
  txt(s, "relay path\n2×2500 it", { x: 11.92, y: 1.83, w: 0.79, h: 0.36, fontSize: 7.5, color: GRAY });
  seg(s, 9.36, 2.3, 10.58, 2.55, { color: GRAY, width: 1.25, endArrowType: "triangle" });
  chip(s, 10.6, 2.42, 1.14, 0.3, "4-step (direct ×2)");
  txt(s, "direct path\n5000 it each", { x: 11.8, y: 2.42, w: 0.9, h: 0.36, fontSize: 7.5, color: GRAY });
  txt(s, "Invariants: total budget · data · timestep schedule · training recipe · eval protocol; two direct arms (high/low LR) forestall an “untuned baseline”", { x: 8.3, y: 2.88, w: 4.35, h: 0.34, fontSize: 9, lineSpacingMultiple: 1.1 });
  s.addShape(RR, { x: 8.3, y: 3.26, w: 2.08, h: 0.32, rectRadius: 0.05, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 } });
  txt(s, "Pre-registered: parity", { x: 8.3, y: 3.26, w: 2.08, h: 0.32, fontSize: 9.5, color: NAVY, align: "center", valign: "middle" });
  s.addShape(RR, { x: 10.53, y: 3.26, w: 2.08, h: 0.32, rectRadius: 0.05, fill: { color: NAVY }, line: { color: NAVY, width: 1 } });
  txt(s, "Observed: direct slightly ahead", { x: 10.53, y: 3.26, w: 2.08, h: 0.32, fontSize: 9, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  // right-bottom flow bars with teacher reference line
  txt(s, "Motion magnitude (RAFT flow median, dm40, 4-seed mean)", { x: 8.1, y: 3.92, w: 4.73, h: 0.24, fontSize: 11, bold: true });
  txt(s, "Relay student", { x: 8.0, y: 4.3, w: 1.25, h: 0.3, fontSize: 9.5, align: "right" });
  txt(s, "Direct student", { x: 8.0, y: 4.8, w: 1.25, h: 0.3, fontSize: 9.5, align: "right" });
  seg(s, 9.35, 4.24, 9.35, 5.18, { color: "AAAAAA", width: 1 });
  s.addShape(RECT, { x: 9.35, y: 4.28, w: (3.36 / 4) * 3.0, h: 0.34, fill: { color: NAVY }, line: { type: "none" } });
  txt(s, "3.36", { x: 9.35 + (3.36 / 4) * 3.0 + 0.06, y: 4.34, w: 0.6, h: 0.2, fontSize: 9.5, bold: true, color: NAVY });
  s.addShape(RECT, { x: 9.35, y: 4.78, w: (1.81 / 4) * 3.0, h: 0.34, fill: { color: NAVY_SOFT }, line: { type: "none" } });
  txt(s, "1.81", { x: 9.35 + (1.81 / 4) * 3.0 + 0.06, y: 4.84, w: 0.6, h: 0.2, fontSize: 9.5, bold: true, color: INK });
  const tRef = 9.35 + (2.05 / 4) * 3.0;
  seg(s, tRef, 4.2, tRef, 5.22, { color: GRAY, width: 1.25, dashType: "dash" });
  txt(s, "teacher 2.05", { x: tRef - 0.575, y: 5.24, w: 1.15, h: 0.16, fontSize: 8.5, color: GRAY, align: "center" });
  txt(s, "Same direction in 4/4 seeds, ≈1.9×; dashed line = teacher reference (4-seed mean 2.05): relay above (4/4), direct not above (3/4). Whether this is good depends on prizing closeness to the teacher vs larger dynamics — we take no side (full table: Backup B5)", { x: 8.1, y: 5.48, w: 4.73, h: 0.85, fontSize: 9.5, color: GRAY });
  footer(s, "", "Source: G2 final table (07-20) · ch2 F1/F5 · flow multi-seed (07-23)");
  s.addNotes("90 s. The relay was our own original design, hence the pre-registration to avoid bias. Honest outcome: quality parity or direct slightly ahead (imaging 0.717 is the student max; aesthetic tie under n=3 bands); diversity higher on both direct arms (0.635/0.628 vs 0.598/0.613); motion magnitude is the relay's only stable measured difference — mechanism on the next slide. Red lines: sharpness/static-bias reading for 0.717 only; state the low DD 0.750 as-is; never quote single-seed percentages; no claim that direct distillation is novel.\n\nQ&A card #1 (why negative results matter): both are progress by controlled exclusion — the first controlled relay-vs-direct comparison on this base model (GPD/CoDMD/FastWan did not run one), plus the relay's one real effect (motion, 4/4 seeds).\n\nFull sources: 2026-07-20-g2-relay-vs-direct-final.md, research/thesis_ch2_draft.md F1/F5, 2026-07-23-flow-multiseed-e1b946-e2a-eval.md. Teacher reference 2.05 = the frozen 4-seed mean from Backup B5 (design item B2, adopted).");
}

/* ============================================================ S6 | P5 ablation II */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 4, kicker: "④ DISCRIMINATOR AUDIT | SINGLE-VARIABLE ABLATION",
    main: "The GAN branch drives the quality decline and the motion magnitude",
    sub: "Same initialization (one 8-step intermediate model), same recipe — only the GAN setting differs; the (t,ε) pairing shows no detected effect",
  });
  txt(s, "Aesthetic quality over training iterations: the three groups", { x: 0.55, y: 1.45, w: 7.3, h: 0.27, fontSize: 11.5, bold: true });
  const px = [0, 1, 2, 3, 4].map((k) => 1.35 + k * 1.175);
  const yb = 5.35, f = 3.4 / 0.09;
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
  txt(s, "training iteration", { x: 1.35, y: 5.6, w: 4.7, h: 0.2, fontSize: 9.5, color: GRAY, align: "center" });
  txt(s, "y-axis 0.53–0.62 (not zero-based)", { x: 1.42, y: 1.98, w: 2.3, h: 0.18, fontSize: 8, color: GRAY });
  polyline(s, px, w7, { color: NAVY_SOFT, width: 2.5, dashType: "dash" }, true, NAVY_SOFT);
  polyline(s, px, e2b, { color: NAVY, width: 2.25 }, true, NAVY);
  polyline(s, px, e2a, { color: TEAL, width: 2.75 }, true, TEAL);
  dot(s, px[3], e2a[3], 0.05, TEAL);
  txt(s, "0.5908", { x: 1.44, y: e2a[0] - 0.3, w: 0.7, h: 0.18, fontSize: 8.5, bold: true, color: TEAL });
  txt(s, "≈0.577 (both GAN-on groups start together)", { x: 1.45, y: 3.72, w: 3.1, h: 0.18, fontSize: 8.5, color: GRAY });
  txt(s, "0.6109 (peak) @2000", { x: 4.28, y: e2a[3] - 0.28, w: 1.6, h: 0.18, fontSize: 8.5, color: TEAL });
  txt(s, "GAN off\nfinal 0.6074", { x: 6.15, y: e2a[4] - 0.2, w: 1.7, h: 0.42, fontSize: 10, bold: true, color: TEAL });
  txt(s, "GAN on · indep. (t,ε)\nfinal 0.5487", { x: 6.15, y: e2b[4] - 0.34, w: 1.72, h: 0.55, fontSize: 9, color: NAVY });
  txt(s, "GAN on · shared (t,ε)\nfinal 0.5379", { x: 6.15, y: w7[4] - 0.02, w: 1.72, h: 0.55, fontSize: 9, color: INK });
  const box = (y, h, head, body, headColor) => {
    s.addShape(RR, { x: 7.95, y, w: 4.9, h, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
    txt(s, head, { x: 8.12, y: y + 0.1, w: 4.6, h: 0.26, fontSize: 11, bold: true, color: headColor || NAVY });
    txt(s, body, { x: 8.12, y: y + 0.4, w: 4.6, h: h - 0.52, fontSize: 9.5, lineSpacingMultiple: 1.1 });
  };
  box(1.5, 1.2, "① “Early peak, later decline” ← GAN branch (candidate)", "GAN off (E2a): quality improves through training, all five checkpoints one way; GAN on (E2b/W7): it declines — the phenomenon disappears and reverses under the single-variable control. Best checkpoint n=3: 3/3 seeds, same direction.", ORANGE);
  box(2.95, 1.2, "② Motion magnitude is mainly sustained by the GAN branch", "GAN off: falls back to the teacher's level (2.1–2.7); GAN on: rebuilds from a low point up to 4.71. Interacts with the relay initialization within the matched recipe (the direct student has GAN on yet stays low).");
  box(4.4, 1.2, "③ (t,ε) pairing convention: no detected effect", "Sharing (t,ε) between real/fake is the upstream default; to our knowledge never tested under control before. Across the five checkpoint pairs the quality gap is ≤0.011; every metric follows the same trajectory.");
  s.addShape(RR, { x: 0.5, y: 6.42, w: 12.33, h: 0.5, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "Cross-seed diversity of all three groups stays within 0.586–0.613 — insensitive to the GAN switch and the pairing convention (wrapped up next page).", { x: 0.78, y: 6.52, w: 11.8, h: 0.3, fontSize: 11.5, bold: true });
  footer(s, "", "Source: GAN-off full table (07-24) · three-group full table (07-25)");
  s.addNotes("105 s. One sentence on cleanliness first: each ablation changes exactly one field of the relay-student recipe (GAN-off: gan_loss_weight_gen 0.03→0; independent pairing: gan_use_same_t_noise True→False), configs verified value-by-value. 20 s on the center chart (same initialization, same recipe, only the GAN setting differs); 15 s per conclusion; stress ③: a widely copied default tested under control for the first time. Line colors: GAN-off teal solid, independent-pairing navy solid, shared-pairing light-blue dashed (orange stays reserved for the degradation axis and warnings). Red lines: 'candidate mechanism' wording (single weight point 0.03, single lineage); never 'the GAN always raises motion' (the direct student has GAN on yet stays low); no comparison wording for 0.613 vs teacher 0.590; say 'no detected effect', never 'no difference'.\n\nQ&A card #3 (quality is higher with the GAN off — why not just turn it off?): single weight point, single lineage; and motion falls back to teacher level — a trade-off, not a free win; we deliver attribution, not a recipe recommendation.\n\nFull sources: 2026-07-24-e2a-fulltable-ch3.md, 2026-07-25-e2b-fulltable-ch3-threearm.md.");
}

/* ============================================================ S7 | P6 attribution wrap-up */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 5, kicker: "⑤ ATTRIBUTION WRAP-UP",
    main: "After component-wise exclusion, the collapse persists",
    sub: "Within the tested components it is attributed to distribution-matching distillation itself; untested components listed as-is",
  });
  const comp = (y, name, ev) => {
    s.addShape(RR, { x: 0.6, y, w: 4.9, h: 0.92, rectRadius: 0.06, fill: { color: PANEL }, line: { color: NAVY, width: 1 } });
    txt(s, name, { x: 0.82, y: y + 0.1, w: 3.1, h: 0.28, fontSize: 12.5, bold: true, color: NAVY });
    s.addShape(RR, { x: 3.95, y: y + 0.1, w: 1.4, h: 0.3, rectRadius: 0.12, fill: { color: LIGHT }, line: { color: GRAY, width: 0.75 } });
    txt(s, "excluded (ctrl.)", { x: 3.95, y: y + 0.14, w: 1.4, h: 0.22, fontSize: 8.5, color: INK, align: "center" });
    txt(s, ev, { x: 0.82, y: y + 0.44, w: 4.5, h: 0.42, fontSize: 9.5, color: GRAY });
    seg(s, 5.5, y + 0.46, 6.15, y + 0.46, { color: GRAY, width: 1.25, endArrowType: "triangle" });
  };
  comp(1.7, "Step-count relay (50→8→4)", "Evidence: ③ controlled comparison — relay diversity is lower (0.598/0.613 vs direct 0.628/0.635)");
  comp(3.0, "GAN discriminator branch", "Evidence: ④ discriminator audit — the switch does not move the collapse (three groups 0.586–0.613)");
  comp(4.3, "(t,ε) pairing convention", "Evidence: ④ discriminator audit — no detected effect at any of the five checkpoints");
  txt(s, "Still present after exclusion: cross-seed diversity collapse", { x: 6.4, y: 1.7, w: 6.4, h: 0.3, fontSize: 12.5, bold: true });
  const bx = 7.75, bw = 4.2;
  const posC = (v) => bx + ((v - 0.4) / 0.4) * bw;
  txt(s, "teacher", { x: 6.35, y: 2.56, w: 1.3, h: 0.3, fontSize: 10, align: "right" });
  txt(s, "students", { x: 6.35, y: 3.36, w: 1.3, h: 0.3, fontSize: 10, align: "right" });
  seg(s, bx, 2.45, bx, 3.95, { color: "AAAAAA", width: 1 });
  s.addShape(RECT, { x: bx, y: 2.52, w: posC(0.732) - bx, h: 0.42, fill: { color: NAVY }, line: { type: "none" } });
  txt(s, "▼", { x: posC(0.732) - 0.125, y: 2.36, w: 0.25, h: 0.18, fontSize: 8, color: NAVY, align: "center" });
  txt(s, "0.732", { x: posC(0.732) + 0.07, y: 2.62, w: 0.7, h: 0.2, fontSize: 10, bold: true, color: NAVY });
  s.addShape(RECT, { x: posC(0.586), y: 3.32, w: posC(0.649) - posC(0.586), h: 0.42, fill: { color: ORANGE }, line: { type: "none" } });
  txt(s, "0.586", { x: posC(0.586) - 0.76, y: 3.44, w: 0.7, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE, align: "right" });
  txt(s, "0.649", { x: posC(0.649) + 0.07, y: 3.44, w: 0.7, h: 0.2, fontSize: 9.5, bold: true, color: ORANGE });
  seg(s, posC(0.586), 3.74, posC(0.586), 3.88, { color: ORANGE, width: 1 });
  seg(s, posC(0.649), 3.74, posC(0.649), 3.88, { color: ORANGE, width: 1 });
  [0.4, 0.6, 0.8].forEach((t) => txt(s, t.toFixed(1), { x: posC(t) - 0.2, y: 3.98, w: 0.4, h: 0.16, fontSize: 8, color: GRAY, align: "center" }));
  txt(s, "Cross-seed diversity (LPIPS, 40 prompts × 8 seeds; students = best ckpts + weak-recipe reference; x-axis 0.4–0.8, not zero-based)", { x: 6.4, y: 4.22, w: 6.4, h: 0.22, fontSize: 8.5, color: GRAY });
  txt(s, "Within the tested components, the collapse is attributed to distribution-matching distillation itself. Untested (listed as-is): data composition, how the teacher's CFG is distilled, other GAN weights.", { x: 6.4, y: 4.55, w: 6.4, h: 0.75, fontSize: 10.5 });
  s.addShape(RR, { x: 1.2, y: 5.85, w: 10.93, h: 0.9, rectRadius: 0.06, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1.5 } });
  txt(s, "This work does not offer a fix; it delivers a precisely measured problem statement whose candidate sources have been excluded under control (an open problem).", { x: 1.5, y: 6.02, w: 10.3, h: 0.62, fontSize: 13.5, bold: true, color: NAVY, align: "center" });
  footer(s, "", "Source: 2026-07-25-e2b-fulltable-ch3-threearm.md (verdict 4)");
  s.addNotes("45 s. 30 s on the exclusion logic — switch the training route (relay→direct): the collapse stays, and is deeper on relay; turn the GAN branch off: stays; change the (t,ε) pairing: stays. 15 s on the open-problem framing: this is the core takeaway of the talk. Red lines: 'attributed to distillation itself' must be scoped 'within the tested components' (untested: data composition, how the teacher's CFG is distilled, etc.); never write 'unsolvable'.");
}

/* ============================================================ S8 | P7 VBench */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 6, kicker: "⑥ EXTERNAL BENCHMARK",
    main: "No dominant model on the standard benchmark",
    sub: "full VBench, 12/16 dims (946 prompts × 5 seeds): each model leads somewhere; the diversity upper bound stays with the teacher",
  });
  const span = (t, fill) => [{ text: t, options: { colspan: 5, fill: { color: fill }, bold: true, fontSize: 8.5, color: INK, align: "left", fontFace: FONT, valign: "middle" } }];
  const R = (name, v, best) => [C(name, { fontSize: 9 })].concat(v.map((x, i) => (i === best
    ? BU(x, { align: "right", fontSize: 10, fill: { color: ORANGE_TINT } })
    : C(x, { align: "right", fontSize: 10 }))));
  s.addTable([
    [HC("Dimension", { align: "left" }), HC("Direct student @1000 · final model"), HC("Relay student @1000"), HC("Direct student B @500"), HC("GAN-off @2000 · ablation · single ckpt")],
    span("Quality dimensions (7) — exactly the official Quality Score set; composable with official weights (Backup B4)", QBLOCK),
    R("subject consistency", ["0.9727", "0.9693", "0.9673", "0.9753"], 3),
    R("background consistency", ["0.9579", "0.9508", "0.9416", "0.9581"], 3),
    R("motion smoothness", ["0.9812", "0.9727", "0.9747", "0.9786"], 0),
    R("dynamic degree", ["0.5806", "0.9111", "0.8806", "0.8000"], 1),
    R("aesthetic quality", ["0.5967", "0.6087", "0.5802", "0.6482"], 3),
    R("imaging quality", ["0.6687", "0.6687", "0.6614", "0.6924"], 3),
    R("temporal flickering", ["0.9894", "0.9796", "0.9810", "0.9878"], 0),
    span("Semantic-type raw dimensions (5) — NOT the official Semantic (4 GRiT dimensions missing; not composable)", SBLOCK),
    R("human action", ["0.690", "0.794", "0.716", "0.776"], 1),
    R("scene", ["0.2173", "0.2922", "0.2225", "0.2974"], 3),
    R("appearance style", ["0.1990", "0.1982", "0.2003", "0.2010"], 3),
    R("temporal style", ["0.2214", "0.2305", "0.2260", "0.2283"], 1),
    R("overall consistency", ["0.2240", "0.2386", "0.2298", "0.2391"], 1),
  ], { x: 0.45, y: 1.42, w: 8.55, colW: [3.1, 1.4, 1.2, 1.2, 1.65], rowH: [0.5, 0.22].concat(Array(7).fill(0.31), [0.22], Array(5).fill(0.31)), border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  txt(s, "Row best = bold underline + light-orange fill. The 4 missing dimensions are detector-dependent (GRiT/detectron2), so the official Semantic and Total cannot be composed — declared as-is; temporal_flickering: the official protocol uses a dedicated subset (25 samples/prompt, static-filtered); ours is 5 samples, unfiltered — not directly comparable.", { x: 0.45, y: 6.14, w: 8.55, h: 0.44, fontSize: 9.5, color: GRAY });
  txt(s, "How to read (each leads somewhere)", { x: 9.25, y: 1.42, w: 3.6, h: 0.28, fontSize: 12, bold: true, color: NAVY });
  const rb = (y, head, body) => {
    txt(s, head, { x: 9.25, y, w: 3.6, h: 0.24, fontSize: 10.5, bold: true });
    txt(s, body, { x: 9.25, y: y + 0.26, w: 3.6, h: 0.72, fontSize: 9.5, color: INK, lineSpacingMultiple: 1.08 });
  };
  rb(1.8, "Direct (final model)", "leads on consistency / smoothness / flicker");
  rb(2.7, "Relay", "leads on dynamic degree 0.911 and action semantics (0.794 / 0.292)");
  rb(3.6, "GAN-off (ablation)", "best static quality (0.6482 / 0.6924); dynamics hold at 0.80 — agrees with the ablation across domains");
  rb(4.74, "teacher", "diversity 0.732 (our protocol's metric) remains the upper bound — the axis is absent from VBench, exactly why we built our own protocol (Backup B4)");
  footer(s, "", "Source: experiments/results/2026-07-26-e2a-vb946-fourth-row.md");
  s.addNotes("60 s. Each model leads somewhere — direct wins consistency/smoothness/flicker, relay wins dynamics and action semantics, direct-B sits in between, GAN-off has the best static quality with dynamics holding (cross-domain agreement with the ablation). The light-orange hot zones make the four leading regions directly visible. Diversity is absent from VBench (why our own protocol exists). Red lines: the final model follows the pre-registered outcome (direct student @1000), no after-the-fact switching; composite scores and the literature-number coincidence live only in Backup B4 / oral answers, never on main slides; never mix dynamic-degree numbers across domains (150-prompt 0.567 vs VBench 0.800).\n\nQ&A #2 (CoDMD 84.46): protocols not comparable (12/16 dims, 5-sample flickering, no GRiT dims) — literature coordinate only; our relay student's Quality Score 84.47 is a numeric coincidence (7-dim quality composite vs 16-dim total), never side by side.\nQ&A #6 (why is the final model's composite lowest, 82.80): weight structure — dynamic degree enters at 0.5 weight and its 0.5806 drags the total; its winning dims lose resolution after normalization; and the composite has no diversity term, so it cannot reflect the main degradation axis. Selection follows the pre-registered protocol.\nQ&A #7 (GAN-off scores highest, 85.50 — why not switch): pre-registration discipline (the final model was fixed before this run existed); its diversity is the lowest band of all (0.586–0.604) and its dynamics fall back to teacher level — exactly what the composite cannot see; single run, single weight point.");
}

/* ============================================================ S9 | P8 conclusions */
{
  const s = pres.addSlide(); pageNo++;
  header(s, {
    idx: 7, kicker: "⑦ CONCLUSIONS",
    main: "The 25× speed-up works; the main degradation is diversity collapse",
    sub: "Conclusions and limitations, stated as-is",
  });
  // two wide columns — advice column removed (ruling 2026-07-27); the three tips live in speaker notes
  const col = (x, sym, head, items) => {
    s.addShape(RECT, { x, y: 1.45, w: 5.9, h: 0.38, fill: { color: NAVY }, line: { type: "none" } });
    txt(s, `${sym} ${head}`, { x: x + 0.14, y: 1.45, w: 5.6, h: 0.38, fontSize: 13, bold: true, color: "FFFFFF", valign: "middle" });
    s.addText(items.map((t) => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, breakLine: true, paraSpaceAfter: 12 } })), { x, y: 1.98, w: 5.9, h: 3.95, fontFace: FONT, fontSize: 13, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.22 });
  };
  col(0.5, "✓", "Main conclusions", [
    "Diagnosis: the main axis is cross-seed diversity collapse (teacher 0.732 → students 0.59–0.64), not dynamic degree",
    "Attribution: the quality decline and the motion magnitude tie mainly to the GAN branch; the (t,ε) pairing shows no detected effect; the diversity collapse comes from distillation itself (open problem)",
    "Method: a lightweight, seed-controlled degradation-audit protocol (incl. evidence that subjective checkpoint picking is unreliable)",
    "Engineering: 25× speed-up; the first controlled relay-vs-direct comparison on this base model (to our knowledge)",
  ]);
  col(6.93, "⚠", "Limitations & next steps", [
    "No human evaluation; 4 benchmark dimensions missing",
    "R1 regularization arm untested under the 32 GB VRAM limit (calibration archived; reproducible on 80 GB-class devices)",
    "One training run per configuration (mitigated by same-family agreement); the relay's intermediate model was picked before the selection rule existed",
    "Next: human eval · cross-lineage ablation · 80 GB rerun · interventions on diversity collapse",
  ]);
  s.addShape(RR, { x: 0.5, y: 6.08, w: 12.33, h: 0.58, rectRadius: 0.06, fill: { color: NAVY }, line: { type: "none" } });
  s.addText([
    { text: "Speed ", options: { fontSize: 11, color: "FFFFFF" } },
    { text: "165.24 s → ≈6.6 s (≈25×)", options: { fontSize: 15, bold: true, color: "FFFFFF" } },
    { text: "    |    ", options: { fontSize: 12, color: NAVY_SOFT } },
    { text: "Diversity ", options: { fontSize: 11, color: "FFFFFF" } },
    { text: "0.732 → 0.59–0.64", options: { fontSize: 15, bold: true, color: "FFFFFF" } },
    { text: "    |    ", options: { fontSize: 12, color: NAVY_SOFT } },
    { text: "Relay · GAN · (t,ε) ", options: { fontSize: 11, color: "FFFFFF" } },
    { text: "all three excluded under control", options: { fontSize: 15, bold: true, color: "FFFFFF" } },
  ], { x: 0.7, y: 6.08, w: 11.93, h: 0.58, fontFace: FONT, align: "center", valign: "middle", margin: 0 });
  txt(s, "Thank you — comments welcome. Thesis due 2026-07-31.", { x: 0.5, y: 6.78, w: 12.33, h: 0.3, fontSize: 12, bold: true, color: NAVY, align: "center" });
  footer(s, "", "Source: T3 adjudication §4.1 · three-group full table (07-25) · acceptance log #11–#13");
  s.addNotes("60 s. ~22 s per column; close by pointing at the number band (25× speed-up, diversity 0.732→0.59–0.64, three components excluded under control) — a calmer pace. Red lines: every 'first' carries 'to our knowledge'; the practical recommendations are oral-only, and recommendation ③ must keep its caveat (single weight point, single training path).\n\nPractical recommendations, oral-only (answer if asked 'what do you recommend to others?'; the full version stays in the thesis conclusions): ① save every 500 iterations and score the full sweep (quality usually peaks early); ② monitor cross-seed diversity and continuous optical flow (standard metrics and composite scores do not raise alarms); ③ treat the GAN weight as a static-quality ↔ motion-magnitude trade-off, and the (t,ε) pairing needs no tuning — always with the caveat: single weight point, single training path, extrapolate with care.\n\nQ&A #4 (why no human eval): the time window went to the controlled ablations; human eval (T2VHE-style vs the teacher) is listed under limitations and next steps.\nQ&A #5 (why is the R1 arm untested): deterministic OOM on the 32 GB card (crash at R1's second discriminator forward), not a recipe failure; calibration values and config are archived, directly reproducible on 80 GB-class devices.\n\nFull sources: research/T3_novelty_adjudication.md §4.1, 2026-07-25-e2b-fulltable-ch3-threearm.md, experiments/acceptance-log.md #11–#13.");
}

/* ============================================================ S10 | B1 protocol */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Backup B1 | Evaluation protocol: four components & the main table");
  const blk = (x, y, head, body) => {
    s.addShape(RR, { x, y, w: 5.99, h: 1.78, rectRadius: 0.05, fill: { color: PANEL }, line: { color: BORDER, width: 0.75 } });
    txt(s, head, { x: x + 0.18, y: y + 0.1, w: 5.6, h: 0.26, fontSize: 12, bold: true, color: NAVY });
    txt(s, body, { x: x + 0.18, y: y + 0.42, w: 5.66, h: 1.3, fontSize: 10, lineSpacingMultiple: 1.12 });
  };
  blk(0.5, 1.35, "① q150: six quality dimensions", "Official VBench all_dimension suite, deterministic sample of 150 prompts (md5 690f2919), custom-input mode; sweeps at seed 0, champions re-checked at n=3 (seeds 0/1/2).");
  blk(6.84, 1.35, "② dm40: clean dynamic degree (the citable DD)", "40 motion-oriented prompts (20 from official human_action.txt, uniform stride + 20 from all_dimension.txt via a MOTION_CUE regex excluding STATIC_BLOCK; md5 324d75a0). q150-DD is confounded by still-style prompts — footnote-level only (teacher DD across domains: 0.300 vs 0.625).");
  blk(0.5, 3.33, "③ d40×8: cross-seed diversity", "40 prompts × 8 seeds; mean pairwise LPIPS-alex (8 frames @256 px; md5 b4c1f9e3; higher = more diverse) — the measurement of this project's main degradation axis.");
  blk(6.84, 3.33, "④ RAFT continuous optical flow (motion magnitude)", "dm40 domain, px/frame; median primary, mean also reported (teacher heavy-tailed: median 2.75 / mean 5.16). Multi-seed discipline: directions reported seed-paired; single-seed percentages never cited alone. Motivation: binary DD saturates for good students (0.75–1.0) — no resolution.");
  s.addShape(RR, { x: 0.5, y: 5.35, w: 12.33, h: 1.4, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "Main-table protocol: full VBench standard mode (946 prompts × 5 seeds), 12/16 dimensions (4 GRiT dims missing; Semantic/Total not composable — declared); temporal_flickering protocol difference footnoted.\nGeneral rules: numbers from q150 / dm40 / vb946 are never mixed across tables; training-health metrics (loss) are not quality evidence; one variable per experiment; checkpoints always best-of-sweep.", { x: 0.78, y: 5.5, w: 11.8, h: 1.15, fontSize: 10.5, lineSpacingMultiple: 1.2 });
  footer(s, " · Backup B1", "Source: research/thesis_ch1_draft.md §1.7");
  s.addNotes("Open when asked for protocol details. md5s and sampling criteria are in the repo (exp/eval/, make_motion_set.py header; re-verified against the remote on 2026-07-26).");
}

/* ============================================================ S11 | B2 E5 probe */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Backup B2 | E5 layer×t separability probe (observational)");
  s.addTable([
    [HC("noise level t"), HC("real-vs-generated linear-probe AUC (all 9 layers)")],
    [C("t = 0.999 (highest noise)", { align: "center" }), C("0.28–0.52 (random or below)", { align: "center" })],
    [C("t ≤ 0.937 (all remaining levels)", { align: "center" }), C("1.0 (all layers saturated, n=64)", { align: "center", bold: true })],
  ], { x: 0.7, y: 1.5, w: 7.0, colW: [2.8, 4.2], rowH: [0.4, 0.42, 0.42], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  s.addText([
    { text: "Cliff-shaped t dependence: ", options: { bold: true } },
    { text: "the discriminator's usable supervision concentrates at low-to-mid noise; the highest-noise level carries almost no signal — mechanistic background for the pairing ablation's 'no detected effect' (the pairing acts exactly where the signal is weakest).", options: {} },
  ], { x: 0.7, y: 3.0, w: 12.0, h: 0.55, fontFace: FONT, fontSize: 11, color: INK, margin: 0 });
  s.addText([
    { text: "Fair verdict on layer choice: ", options: { bold: true } },
    { text: "AUC saturates at every layer for t ≤ 0.937 — no between-layer resolution (the early “mean AUC 0.88–0.92 / L7 slightly better” reading was a saturation-averaging artifact, discarded); the continuous Fréchet distance rises gently with depth and jumps at L27/29. Upstream's layer choice {15, 22, 29}: covers the mid and deep ranges; no evidence it is a bad choice, and no evidence it is uniquely optimal.", options: {} },
  ], { x: 0.7, y: 3.7, w: 12.0, h: 0.85, fontFace: FONT, fontSize: 11, color: INK, margin: 0 });
  s.addText([
    { text: "Honesty control (teachergen): ", options: { bold: true } },
    { text: "the teacher's own 50-step outputs separate from real data in the same cliff shape (and with larger FD) → feature separability mainly reflects generated-vs-real domain gaps plus prompt-domain differences; it cannot serve as a direct measure of distillation degradation.", options: {} },
  ], { x: 0.7, y: 4.7, w: 12.0, h: 0.75, fontFace: FONT, fontSize: 11, color: INK, margin: 0 });
  txt(s, "Protocol: 64 clips/side; feature path field-matched to the training-side discriminator; null-text conditioning; 5-fold linear-probe AUC + Fréchet distance. Status: to-our-knowledge observational evidence.", { x: 0.7, y: 5.7, w: 12.0, h: 0.5, fontSize: 9.5, color: GRAY });
  footer(s, " · Backup B2", "Source: research/E5_probe_results.md (corrected reading — authoritative)");
  s.addNotes("Open when asked about the discriminator mechanism / E5. Validity frame: n=64/side, single batch, null-text conditioning, no resolution where AUC saturates, FD sample-size sensitive.");
}

/* ============================================================ S12 | B3 design details */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Backup B3 | Relay-vs-direct comparison: design details");
  const row = (y, head, body) => {
    txt(s, head, { x: 0.7, y, w: 2.6, h: 0.6, fontSize: 12, bold: true, color: NAVY });
    txt(s, body, { x: 3.5, y, w: 9.3, h: 0.85, fontSize: 11, lineSpacingMultiple: 1.15 });
  };
  row(1.5, "Relay arm", "W5 (8-step, LR 1e-5 / batch 12, 2500 it, from the teacher) → W7 (4-step, LR 5e-6 / batch 16, 2500 it; inherits only W5@2500 generator weights — optimizer / fake score / discriminator all re-initialized). Total budget 5000 it.");
  row(2.55, "Direct arms", "E1a = relay stage-2 recipe (LR 5e-6 / batch 16); E1b = stage-1 recipe (LR 1e-5 / batch 12 — exactly the upstream FastGen default LR); 5000 it each, from the teacher. The two-arm bracket forestalls the “untuned direct baseline” objection.");
  row(3.6, "Invariants", "Data (OpenVid-1M) · 4-step t_list · discriminator architecture · all single-stage hyper-parameters (upstream published values) · evaluation protocol · checkpoint granularity (every 500 it).");
  row(4.55, "Selection", "All arms best-of-sweep (E1a/E1b: 10 checkpoints each; W7: 5; the 32-row table has no missing cells); champions re-checked at n=3 seeds.");
  s.addShape(RR, { x: 0.7, y: 5.55, w: 12.0, h: 0.95, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "Pre-registration discipline: before launch, the leading hypothesis was logged as “parity” (neutral); the observed result — direct slightly ahead — is reported exactly as pre-registered, with no after-the-fact adjustment.", { x: 0.95, y: 5.72, w: 11.5, h: 0.62, fontSize: 11.5, bold: true, color: NAVY });
  footer(s, " · Backup B3", "Source: research/thesis_ch2_draft.md §2.1");
  s.addNotes("Open when asked why the comparison is trustworthy / which objections were pre-empted. The path schematic is on the main slide '③ controlled comparison' (teacher fork → relay 50→8→4 with full reset / direct 50→4).");
}

/* ============================================================ S13 | B4 Quality Score */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Backup B4 | Quality Score synthesis and its blind spot");
  txt(s, "Official weights (Vchitect/VBench scripts/constant.py + cal_final_score.py @master): per-dimension min-max normalization; dynamic_degree weight 0.5, the other six weight 1; weighted sum ÷ 6.5.", { x: 0.6, y: 1.42, w: 12.1, h: 0.5, fontSize: 10.5, color: INK });
  s.addTable([
    [HC("Model"), HC("Quality Score (7-dim)"), HC("Remark")],
    [C("E2a@2000 (audit arm GAN=0, single ckpt)", { fontSize: 10 }), C("85.50", { align: "center", bold: true }), C("lowest diversity band of all (0.586–0.604); dynamics back at teacher level", { fontSize: 9 })],
    [C("W7@1000 (relay)", { fontSize: 10 }), C("84.47", { align: "center" }), C("numeric coincidence with CoDMD's 84.46 — never side by side (below)", { fontSize: 9 })],
    [C("E1b@500 (direct-B)", { fontSize: 10 }), C("83.62", { align: "center" }), C("high dynamics, low aesthetics", { fontSize: 9 })],
    [C("E1a@1000 (final model, pre-registered)", { fontSize: 10 }), C("82.80", { align: "center" }), C("dynamic degree 0.5806 enters at 0.5 weight; winning dims lose resolution after normalization", { fontSize: 9 })],
  ], { x: 0.6, y: 2.0, w: 8.6, colW: [3.3, 2.1, 3.2], rowH: [0.38, 0.44, 0.44, 0.44, 0.44], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  s.addShape(RR, { x: 0.6, y: 4.45, w: 8.6, h: 1.15, rectRadius: 0.05, fill: { color: LIGHT }, line: { color: BORDER, width: 0.75 } });
  txt(s, "Blind spot: the composite has no diversity term — it cannot reflect this project's confirmed main degradation axis (the lowest-diversity model ranks first, the final model last; the W4 lesson at benchmark scale, and exactly why our own protocol exists). Model selection follows the pre-registered protocol, unaffected by this composite.", { x: 0.85, y: 4.6, w: 8.1, h: 0.9, fontSize: 10.5, lineSpacingMultiple: 1.15 });
  s.addShape(RR, { x: 0.6, y: 5.85, w: 8.6, h: 0.85, rectRadius: 0.05, fill: { color: "FFFFFF" }, line: { color: DARKRED, width: 1.5 } });
  txt(s, "Warning: W7's 84.47 vs the published CoDMD 84.46 is a numeric coincidence — a 7-dimension quality composite vs a 16-dimension total; different scales. Never place the two side by side or compare them.", { x: 0.85, y: 6.0, w: 8.1, h: 0.6, fontSize: 10.5, bold: true, color: DARKRED });
  txt(s, "Demo clips (same prompt & seed, cyclist)", { x: 9.5, y: 2.0, w: 3.3, h: 0.22, fontSize: 9.5, bold: true, color: NAVY });
  video(s, "backup/backup_e2a_cyclist.mp4", 9.5, 2.28, 3.15, 1.817);
  txt(s, "Audit arm E2a@2000 (GAN=0): high static quality, dynamics fall back", { x: 9.5, y: 4.11, w: 3.3, h: 0.34, fontSize: 8.5, color: GRAY });
  video(s, "backup/backup_w7_cyclist.mp4", 9.5, 4.55, 3.15, 1.817);
  txt(s, "Relay W7@1000 (paired GAN): large dynamics, quality declines over iterations", { x: 9.5, y: 6.38, w: 3.3, h: 0.34, fontSize: 8.5, color: GRAY });
  footer(s, " · Backup B4", "Source: experiments/results/2026-07-26-e2a-vb946-fourth-row.md (QS section)");
  s.addNotes("Use with oral answers #6/#7: the final model scores lowest via the weight structure; the GAN-off arm scores highest yet has the lowest diversity band and teacher-level dynamics — the two things the composite cannot see; pre-registration forbids switching.");
}

/* ============================================================ S14 | B5 flow table */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Backup B5 | Motion magnitude multi-seed table (RAFT, dm40)");
  s.addTable([
    [HC("seed"), HC("Relay W7@1000"), HC("Direct E1a@1000"), HC("teacher 50-step")],
    [C("s0", { align: "center" }), C("4.44", { align: "center" }), C("2.15", { align: "center" }), C("2.75", { align: "center" })],
    [C("s1", { align: "center" }), C("3.28", { align: "center" }), C("1.83", { align: "center" }), C("2.17", { align: "center" })],
    [C("s2", { align: "center" }), C("1.27", { align: "center" }), C("0.46", { align: "center" }), C("0.86", { align: "center" })],
    [C("s3", { align: "center" }), C("4.44", { align: "center" }), C("2.80", { align: "center" }), C("2.41", { align: "center" })],
    [C("4-seed mean", { align: "center", bold: true }), C("3.36", { align: "center", bold: true }), C("1.81", { align: "center", bold: true }), C("2.05", { align: "center", bold: true })],
  ], { x: 0.9, y: 1.5, w: 7.6, colW: [1.6, 2.0, 2.0, 2.0], rowH: [0.4, 0.38, 0.38, 0.38, 0.38, 0.4], border: { type: "solid", color: BORDER, pt: 0.5 }, fontFace: FONT, valign: "middle" });
  s.addText([
    { text: "Seed-paired conclusions:", options: { bold: true, breakLine: true } },
    { text: "W7 > E1a: 4/4 seeds, same direction (≈1.9×)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "W7 > teacher: 4/4 seeds, same direction (by means, +64%)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "E1a not above the teacher (3/4 seeds; by means, −12%)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "seed 2 is low for all three models: the initial noise dominates part of the dynamics level — the empirical case for seed-paired design", options: { bullet: { code: "2022", indent: 10 }, breakLine: true, paraSpaceAfter: 5 } },
    { text: "Discipline: single-seed absolute percentages are never cited alone (per-seed medians can differ by 6×)", options: { bullet: { code: "2022", indent: 10 }, breakLine: true } },
  ], { x: 0.9, y: 4.35, w: 11.5, h: 2.2, fontFace: FONT, fontSize: 11, color: INK, valign: "top", margin: 0, lineSpacingMultiple: 1.15 });
  footer(s, " · Backup B5", "Source: experiments/results/2026-07-23-flow-multiseed-e1b946-e2a-eval.md §1/§4");
  s.addNotes("Open when asked how strong the motion-magnitude evidence is; the main slide '③ controlled comparison' reports only the 4-seed means and directions (its dashed reference 2.05 = this table's teacher 4-seed mean).");
}

/* ============================================================ S15 | B6 upstream */
{
  const s = pres.addSlide(); pageNo++;
  title(s, "Backup B6 | Relationship to upstream NVIDIA FastGen (statement)");
  s.addShape(RR, { x: 0.7, y: 1.5, w: 12.0, h: 2.95, rectRadius: 0.05, fill: { color: PANEL }, line: { color: BORDER, width: 0.75 } });
  txt(s, "All training builds on NVIDIA FastGen (NVlabs/FastGen, Apache-2.0), reusing its native DMD2 implementation and the official Wan2.1-T2V-1.3B configuration — teacher CFG=5; generator-side GAN weight 0.03; shared real/fake timestep and noise (gan_use_same_t_noise=True is the factory default); the multiscale MLP discriminator on frozen-teacher features from layers 15/22/29; two-time-scale updates with student_update_freq=5; and the 4-step t_list=[0.999, 0.937, 0.833, 0.624, 0.0].\n\nOn top of this, our recipe contribution is confined to the training-schedule layer: the official repository ships only single-stage 50→4/2-step configurations; we run a 50→8→4 step-count relay with an added 8-step intermediate student, where the 4-step stage inherits only the generator weights of the best 8-step checkpoint (optimizer / fake score / discriminator all re-initialized). On the data side we use OpenVid-1M (upstream is dataset-agnostic).", { x: 1.0, y: 1.72, w: 11.4, h: 2.62, fontSize: 10.5, lineSpacingMultiple: 1.22 });
  txt(s, "Further contributions of this report: the controlled audits (relay comparison, three-group discriminator ablation) and the portable degradation-evaluation protocol. Discriminator wording, everywhere: trainable multiscale MLP heads on frozen-teacher features, layers 15/22/29 (code-verified 2026-07-06).", { x: 0.7, y: 4.65, w: 12.0, h: 0.7, fontSize: 10.5, color: INK });
  footer(s, " · Backup B6", "Source: research/T3_novelty_adjudication.md §6.2");
  s.addNotes("Show verbatim when asked what is ours vs upstream. Never say we 'improved' FastGen; never present single-stage hyper-parameters as our design.");
}

const OUT = QA_IMAGES ? "DMD2_report_0728_en_V2_qa.pptx" : "DMD2_report_0728_en_V2.pptx";
fs.writeFileSync(path.join(__dirname, "qa", "ops_v2.json"), JSON.stringify(QA_LOG));
pres.writeFile({ fileName: path.join(__dirname, OUT) }).then(() => {
  console.log("deck written: " + OUT + " | slides: " + QA_LOG.length);
});
