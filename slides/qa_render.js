// HTML mirror renderer for deck QA: reads qa/ops.json (recorded pptxgenjs ops) and emits
// per-slide 1333.3x750 px HTML pages with identical geometry/fonts. Rendered to PNG via qlmanage.
const fs = require("fs");
const path = require("path");

const S = 73; // px per inch (973px page fits QuickLook 980px layout width)
const OPS = JSON.parse(fs.readFileSync(path.join(__dirname, "qa", "ops.json"), "utf8"));
const QACOV = path.join(__dirname, "qa_covers");

const px = (inch) => (inch * S).toFixed(1);
const fpx = (pt) => ((pt * S) / 72).toFixed(1);
const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function imgData(p) {
  const base = path.basename(p, path.extname(p));
  const small = path.join(QACOV, base + ".png");
  const f = fs.existsSync(small) ? small : p;
  return "data:image/png;base64," + fs.readFileSync(f).toString("base64");
}

function styleText(o) {
  const st = [];
  st.push(`font-size:${fpx(o.fontSize || 12)}px`);
  if (o.bold) st.push("font-weight:600");
  st.push(`color:#${o.color || "333333"}`);
  if (o.align) st.push(`text-align:${o.align}`);
  const m = o.lineSpacingMultiple || 1;
  st.push(`line-height:${(1.2 * m).toFixed(2)}em`);
  if (o.underline) st.push("text-decoration:underline");
  return st.join(";");
}

function renderRuns(arr) {
  let html = "";
  for (const r of arr) {
    const o = r.options || {};
    const st = [];
    if (o.bold) st.push("font-weight:600");
    if (o.color) st.push(`color:#${o.color}`);
    if (o.underline) st.push("text-decoration:underline");
    const inner = esc(r.text).replace(/\n/g, "<br>");
    if (o.bullet) {
      const mb = o.paraSpaceAfter ? `margin-bottom:${fpx(o.paraSpaceAfter)}px;` : "";
      html += `<div style="${mb}${st.join(";")};padding-left:1.1em;text-indent:-1.1em">•&nbsp;${inner}</div>`;
    } else if (o.breakLine) {
      html += `<div style="${st.join(";")}">${inner}</div>`;
    } else {
      html += `<span style="${st.join(";")}">${inner}</span>`;
    }
  }
  return html;
}

function renderText(args) {
  const [content, o] = args;
  const boxSt = [
    `left:${px(o.x)}px`, `top:${px(o.y)}px`, `width:${px(o.w)}px`, `height:${px(o.h)}px`,
    "position:absolute", "outline:1px dashed rgba(210,60,60,0.30)",
  ];
  let inner;
  if (Array.isArray(content)) inner = renderRuns(content);
  else inner = esc(content).replace(/\n/g, "<br>");
  const tst = styleText(o);
  if (o.valign === "middle") {
    return `<div style="${boxSt.join(";")};display:flex;align-items:center"><div style="${tst};width:100%">${inner}</div></div>`;
  }
  return `<div style="${boxSt.join(";")};${tst}">${inner}</div>`;
}

function renderShape(args, svgLines) {
  const [type, o] = args;
  if (type === "line") {
    const x1 = o.x * S, y1 = (o.flipV ? o.y + o.h : o.y) * S;
    const x2 = (o.x + o.w) * S, y2 = (o.flipV ? o.y : o.y + o.h) * S;
    const w = ((o.line && o.line.width) || 1) * S / 72;
    const col = "#" + ((o.line && o.line.color) || "1F3864");
    const dash = o.line && o.line.dashType === "dash" ? ' stroke-dasharray="7,5"' : "";
    const arrow = o.line && o.line.endArrowType ? ' marker-end="url(#arr)"' : "";
    svgLines.push(`<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="${w.toFixed(1)}"${dash}${arrow}/>`);
    return "";
  }
  const st = [`left:${px(o.x)}px`, `top:${px(o.y)}px`, `width:${px(o.w)}px`, `height:${px(o.h)}px`, "position:absolute", "box-sizing:border-box"];
  if (o.fill && o.fill.color) st.push(`background:#${o.fill.color}`);
  if (o.line && o.line.color && o.line.type !== "none" && o.line.width !== 0) st.push(`border:${Math.max(1, ((o.line.width || 1) * S) / 72).toFixed(1)}px solid #${o.line.color}`);
  if (type === "roundRect") st.push(`border-radius:${px(o.rectRadius || 0.05)}px`);
  if (type === "ellipse") st.push("border-radius:50%");
  return `<div style="${st.join(";")}"></div>`;
}

function renderImage(args) {
  const o = args[0];
  const src = imgData(o.path);
  return `<img src="${src}" style="position:absolute;left:${px(o.x)}px;top:${px(o.y)}px;width:${px(o.w)}px;height:${px(o.h)}px;object-fit:fill;outline:1px solid #ccc">`;
}

function renderTable(args) {
  const [rows, o] = args;
  const colW = o.colW;
  const rowH = Array.isArray(o.rowH) ? o.rowH : rows.map(() => o.rowH || 0.4);
  let html = `<table style="position:absolute;left:${px(o.x)}px;top:${px(o.y)}px;width:${px(o.w)}px;border-collapse:collapse;table-layout:fixed;font-family:'PingFang SC'">`;
  html += "<colgroup>" + colW.map((w) => `<col style="width:${px(w)}px">`).join("") + "</colgroup>";
  rows.forEach((row, ri) => {
    html += `<tr style="height:${px(rowH[ri] || 0.4)}px">`;
    for (const cell of row) {
      const c = cell.options || {};
      const st = [
        `font-size:${fpx(c.fontSize || 10)}px`,
        `color:#${c.color || "333333"}`,
        c.bold ? "font-weight:600" : "",
        c.underline ? "text-decoration:underline" : "",
        `text-align:${c.align || "left"}`,
        c.fill && c.fill.color ? `background:#${c.fill.color}` : "",
        "border:0.7px solid #D9D9D9", "padding:2px 8px", "vertical-align:middle", "line-height:1.15em",
      ].filter(Boolean);
      const span = c.colspan ? ` colspan="${c.colspan}"` : "";
      html += `<td${span} style="${st.join(";")}">${esc(cell.text)}</td>`;
    }
    html += "</tr>";
  });
  html += "</table>";
  return html;
}

OPS.forEach((rec, i) => {
  const svgLines = [];
  let body = "";
  for (const op of rec.ops) {
    if (op.op === "addText") body += renderText(op.args);
    else if (op.op === "addShape") body += renderShape(op.args, svgLines);
    else if (op.op === "addImage" || op.op === "addMedia") body += renderImage(op.args);
    else if (op.op === "addTable") body += renderTable(op.args);
  }
  const svg = `<svg style="position:absolute;left:0;top:0;pointer-events:none" width="973" height="547" xmlns="http://www.w3.org/2000/svg"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#767676"/></marker></defs>${svgLines.join("")}</svg>`;
  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=1205"><style>
  html,body{margin:0;padding:0}
  body{width:973px;height:547px;position:relative;background:#fff;font-family:'PingFang SC';overflow:hidden}
  div{box-sizing:border-box}
  </style></head><body>${body}${svg}</body></html>`;
  fs.writeFileSync(path.join(__dirname, "qa", `slide${String(i + 1).padStart(2, "0")}.html`), html);
});
console.log("mirror pages written:", OPS.length);
