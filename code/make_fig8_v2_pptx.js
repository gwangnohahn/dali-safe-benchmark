/* Fig 8 v2 (2026-09-29) -- editable PowerPoint version of the applicability map.
 *
 * 2 x 2 map (record clustered / uniform) x (optimum interior / near boundary), drawn natively.
 * Each row: [clustered cell bars] | shared method column (Fig 2 spine) | [uniform cell bars].
 * Bars = cell mean of violations per 100 experiments (class colours), dots = configurations,
 * symlog axis (linear below 1, log above).  Verdict badge + short summary above each cell.
 * The near-boundary row carries the two exploratory kNN-gate rows (dotted bars, clustered cell only).
 * Data: figures/Fig8_v2_data.json (make_fig8_v2_data.py).
 *
 * Usage: node make_fig8_v2_pptx.js [dataJson] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const dataPath = process.argv[2] || path.join(CODE_DIR, "..", "figures", "Fig8_v2_data.json");
const outPath = process.argv[3] || path.join(CODE_DIR, "..", "figures", "Fig8_applicability_map_v2.pptx");
const D = JSON.parse(fs.readFileSync(dataPath, "utf8"));

const FONT = "Arial", INK = "222222", MUTED = "595959", AXIS = "808080", GRID = "E3E3E3";
const methods = D.methods.map(m => m.key);
const LABEL = Object.fromEntries(D.methods.map(m => [m.key, m.label]));
const COLOR = Object.fromEntries(D.methods.map(m => [m.key, m.color.replace("#", "").toUpperCase()]));
const BAND = { DALIv2LE: "E4EEF8", SCBO: "E2F3F1", TuRBO: "EFE8F5", cTuRBOknown: "FFF4D6" };
const CLASSES = [["DALIv2LE", "DALIv2", "LEonly"], ["SCBO"], ["TuRBO", "GA", "BO", "Random"], ["cTuRBOknown"]];
const VERDICT = { Safe: "2E8B57", "Mostly safe": "C8912A", "Safety erodes": "C8912A", "Not safe": "B8322B" };

const pres = new pptxgen();
const W = 12.9, H = 8.75;
pres.defineLayout({ name: "FIG8", width: W, height: H });
pres.layout = "FIG8";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

function text(str, x, y, w, h, o = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 9, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" }, o));
}
function rect(x, y, w, h, fill, lineCol, lw) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: fill ? { color: fill } : { type: "none" }, line: { color: lineCol || fill, width: lw === undefined ? 0 : lw } });
}
function line(x1, y1, x2, y2, color, width, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color, width, dashType: dash || "solid" } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.ShapeType.line, o);
}
// symlog x (linear 0..1, log above), as Fig 3
const LINSCALE = 0.55;
const u = v => (v <= 1 ? v * LINSCALE : LINSCALE + Math.log10(v));
function xmap(x0, w, vmin, vmax) { const a = u(vmin), b = u(vmax); return v => x0 + (u(Math.max(v, vmin)) - a) / (b - a) * w; }

// ---------------------------------------------------------------- layout
const rowsA = methods;                                                  // interior row
const rowsB = ["DALIv2LE", "DALIv2LEgate", "DALIv2", "LEonly", "LEonlygate", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"];
const rowH = 0.29;
const colW = 1.45, midX = (W - colW) / 2;                               // shared column in the middle
const cellW = 4.05, gapC = 0.30;
const leftX0 = midX - gapC - cellW, rightX0 = midX + colW + gapC;
const xTicks = [0, 1, 3, 10, 30, 100];
const VMIN = -0.15, VMAX = 130;

function panelTitle(letter, title, x, y, w) {
  s.addText([{ text: letter, options: { bold: true, fontSize: 14 } }, { text: "   " + title, options: { fontSize: 11 } }],
    { x, y, w, h: 0.3, fontFace: FONT, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" });
}
function drawCell(x0, yTop, rows, key, letter, verdict, showGate) {
  const cell = D.cells[key];
  const X = xmap(x0, cellW - 0.55, VMIN, VMAX);
  const yAx = yTop + rows.length * rowH;
  panelTitle(letter, verdict, x0 - 0.05, yTop - 0.5, cellW + 0.3);
  // axis
  line(X(0), yAx, X(100), yAx, AXIS, 0.75);
  xTicks.forEach(v => {
    line(X(v), yTop, X(v), yAx, GRID, 0.5);
    line(X(v), yAx, X(v), yAx + 0.05, AXIS, 0.75);
    text(String(v), X(v) - 0.25, yAx + 0.06, 0.5, 0.15, { align: "center", fontSize: 7.5, color: MUTED });
  });
  text("Violations in 100 experiments", x0, yAx + 0.24, cellW - 0.55, 0.2, { align: "center", fontSize: 9 });
  // bars
  rows.forEach((m, i) => {
    const y = yTop + i * rowH, bh = rowH * 0.62;
    const isGate = m.endsWith("gate");
    let d = null, col;
    if (isGate) { if (!showGate) return; d = D.gate[m]; col = COLOR[d.base]; }
    else { d = cell.methods[m]; if (!d) return; col = COLOR[m]; }
    const w = Math.max(X(d.V_mean) - X(0), 0.004);
    s.addShape(pres.ShapeType.rect, { x: X(0), y: y + (rowH - bh) / 2, w, h: bh,
      fill: isGate ? { type: "none" } : { color: col }, line: { color: isGate ? col : INK, width: isGate ? 1.25 : 0.5, dashType: isGate ? "sysDot" : "solid" } });
    d.V_per_config.forEach((v, k) => {
      const r = 0.024;
      s.addShape(pres.ShapeType.ellipse, { x: X(v) - r, y: y + rowH / 2 + d.jitter[k] * rowH * 0.9 - r, w: 2 * r, h: 2 * r, fill: { color: "000000", transparency: 50 }, line: { color: "000000", width: 0 } });
    });
    const vmax = Math.max(d.V_mean, ...d.V_per_config);
    const lab = d.V_mean < 10 ? d.V_mean.toFixed(1) : d.V_mean.toFixed(0);
    text(lab, X(vmax) + 0.06, y, 0.5, rowH, { fontSize: 8.5, bold: ["DALIv2LE", "LEonly", "SCBO"].includes(m) });
  });
  return { yAx, X };
}

function drawColumn(yTop, rows, header) {
  text(header, midX - 0.2, yTop - 0.32, colW + 0.4, 0.24, { align: "center", bold: true, fontSize: 9 });
  line(midX + 0.02, yTop - 0.05, midX + colW - 0.02, yTop - 0.05, INK, 1.0);
  for (const grp of CLASSES) {
    const ks = grp.map(m => rows.indexOf(m));
    let hi = Math.max(...ks);
    // include the gate rows that follow a member of the class
    rows.forEach((m, i) => { if (m.endsWith("gate") && grp.includes(m.replace("gate", ""))) hi = Math.max(hi, i); });
    const yT = yTop + Math.min(...ks) * rowH + 0.02, yB = yTop + (hi + 1) * rowH - 0.02;
    rect(midX, yT, colW, yB - yT, BAND[grp[0]]);
  }
  rows.forEach((m, i) => {
    const yc = yTop + i * rowH + rowH / 2;
    const isGate = m.endsWith("gate");
    text(isGate ? "+ kNN gate" : LABEL[m], midX, yTop + i * rowH, colW, rowH, { align: "center", fontSize: isGate ? 8 : 8.5, italic: isGate, color: INK });
    line(leftX0 + cellW - 0.55 + 0.55 - 0.02 - 0.5, yc, midX - 0.02, yc, "BFBFBF", 0.75);     // tick towards the left cell (over the value-label zone)
    line(midX + colW + 0.02, yc, rightX0 - 0.02, yc, "BFBFBF", 0.75);                          // tick towards the right cell
  });
}

// ---------------------------------------------------------------- headers
text("Record clustered around the best condition", leftX0, 0.18, cellW, 0.3, { align: "center", bold: true, fontSize: 11.5 });
text("Record uniform over the feasible set (exploratory)", rightX0, 0.18, cellW, 0.3, { align: "center", bold: true, fontSize: 11.5 });
function rowHeader(str, yTop, n) {
  const hh = n * rowH;
  s.addText(str, { x: 0.05 - hh / 2 + 0.15, y: yTop + hh / 2 - 0.15, w: hh, h: 0.3, rotate: 270, fontFace: FONT, fontSize: 11.5, bold: true, color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
}

// ---------------------------------------------------------------- row 1: optimum interior
const y1 = 1.2;
rowHeader("Optimum interior", y1, rowsA.length);
drawCell(leftX0, y1, rowsA, "ci", "a", "Safe", false);
drawColumn(y1, rowsA, "Method");
drawCell(rightX0, y1, rowsA, "ui", "b", "Safety erodes", false);

// ---------------------------------------------------------------- row 2: optimum near boundary
const y2 = y1 + rowsA.length * rowH + 1.25;
rowHeader("Optimum near boundary", y2, rowsB.length);
drawCell(leftX0, y2, rowsB, "cb", "c", "Mostly safe — one failure mode (non-convex G2)", true);
drawColumn(y2, rowsB, "Method");
drawCell(rightX0, y2, rowsB, "ub", "d", "Not safe", false);

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
