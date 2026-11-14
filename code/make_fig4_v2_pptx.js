/* Fig 4 v2 (2026-09-28) -- PowerPoint version of Fig 4 (when violations occur).
 *
 * Panels (a) and (b) are placed as images of the plot area (curves, bands, ticks; made by
 * make_fig4_v2.py); everything else is editable: panel titles, axis titles, the direct
 * end labels with leader lines, panel (c) bars / dots / values, and the footnote.
 * Data: figures/Fig4_v2_data.json.  Style as Fig 1-3 v2 (class colours; line styles
 * distinguish methods within a class).
 *
 * Usage: node make_fig4_v2_pptx.js [figdir] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const figdir = process.argv[2] || path.join(CODE_DIR, "..", "figures");
const outPath = process.argv[3] || path.join(figdir, "Fig4_survival_v2.pptx");
const D = JSON.parse(fs.readFileSync(path.join(figdir, "Fig4_v2_data.json"), "utf8"));

const FONT = "Arial", INK = "222222", MUTED = "595959", AXIS = "808080";
const methods = D.methods.map(m => m.key);
const LABEL = Object.fromEntries(D.methods.map(m => [m.key, m.label]));
const COLOR = Object.fromEntries(D.methods.map(m => [m.key, m.color.replace("#", "").toUpperCase()]));
const hex = m => (m === "cTuRBOknown" ? "000000" : COLOR[m]);

const pres = new pptxgen();
const W = 12.9, H = 4.35;
pres.defineLayout({ name: "FIG4", width: W, height: H });
pres.layout = "FIG4";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

function text(str, x, y, w, h, o = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 9, color: INK, margin: 0,
    isTextBox: true, valign: "middle", align: "left" }, o));
}
function line(x1, y1, x2, y2, color, width, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color, width, dashType: dash || "solid" } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.ShapeType.line, o);
}
function panelTitle(letter, title, x, y, w) {
  s.addText([{ text: letter, options: { bold: true, fontSize: 14 } }, { text: "   " + title, options: { fontSize: 11 } }],
    { x, y, w, h: 0.32, fontFace: FONT, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" });
}
function ylabel(str, ax) {
  s.addText(str, { x: ax.x0 - 0.62 - ax.h / 2 + 0.1, y: ax.y0 + ax.h / 2 - 0.1, w: ax.h, h: 0.2, rotate: 270,
    fontFace: FONT, fontSize: 9.5, color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
}
// stack labels so that consecutive ones (sorted by y) are >= minSep apart (same rule as v1)
function stack(ys, minSep, yMin, yMax) {
  const idx = ys.map((y, i) => i).sort((a, b) => ys[a] - ys[b]);   // slide y grows downwards: sort ascending
  const pos = idx.map(i => ys[i]);
  for (let k = 1; k < pos.length; k++) if (pos[k] - pos[k - 1] < minSep) pos[k] = pos[k - 1] + minSep;
  if (yMax !== undefined && pos[pos.length - 1] > yMax) {          // overflow at the bottom: push up from the end only as far as needed
    pos[pos.length - 1] = yMax;
    for (let k = pos.length - 2; k >= 0; k--) pos[k] = Math.min(pos[k], pos[k + 1] - minSep);
  }
  if (yMin !== undefined && pos[0] < yMin) { const d = yMin - pos[0]; for (let k = 0; k < pos.length; k++) pos[k] += d; }
  const out = new Array(ys.length); idx.forEach((i, k) => out[i] = pos[k]); return out;
}

// ---------------------------------------------------------------- image panels (a), (b)
const SCALE = 0.8;
function placeImage(key, imgX, imgY) {
  const g = D.geom[key], [fw, fh] = g.fig, [ax0, ay0, aw, ah] = g.ax;
  s.addImage({ path: path.join(figdir, `Fig4_v2_panel_${key}.png`), x: imgX, y: imgY, w: fw * SCALE, h: fh * SCALE });
  const A = { x0: imgX + ax0 * SCALE, y0: imgY + (fh - ay0 - ah) * SCALE, w: aw * SCALE, h: ah * SCALE };
  const [xl, xh] = D[key].xlim, [yl, yh] = D[key].ylim;
  A.X = v => A.x0 + (v - xl) / (xh - xl) * A.w;
  A.Y = v => A.y0 + A.h - (v - yl) / (yh - yl) * A.h;
  return A;
}
function endLabels(A, key, valueKey, fmt, minSep) {
  const ys = methods.map(m => A.Y(D[key][m][valueKey]));
  const pos = stack(ys, minSep, A.y0 - 0.02, A.y0 + A.h + 0.02);
  const xEnd = A.X(D[key].xlim[1]), xText = xEnd + 0.10;
  methods.forEach((m, i) => {
    if (Math.abs(pos[i] - ys[i]) > 1e-3) line(xEnd + 0.01, ys[i], xText - 0.02, pos[i], "A6A6A6", 0.5);
    text(fmt(D[key][m][valueKey]), xText, pos[i] - 0.08, 0.6, 0.16, { fontSize: 8, color: hex(m) });
  });
}

// (a)
panelTitle("a", "Runs without a violation", 0.30, 0.12, 4.5);
const A = placeImage("a", 0.40, 0.48);
ylabel("Fraction of runs with no violation", A);
text("Experiment", A.x0, A.y0 + A.h + 0.22, A.w, 0.2, { align: "center", fontSize: 9.5 });
endLabels(A, "a", "S_end", v => `${Math.round(100 * v)} %`, 0.15);

// (b)
panelTitle("b", "Cumulative violations", 4.95, 0.12, 4.0);
const B = placeImage("b", 5.05, 0.48);
ylabel("Cumulative violations", B);
text("Experiment", B.x0, B.y0 + B.h + 0.22, B.w, 0.2, { align: "center", fontSize: 9.5 });
endLabels(B, "b", "V_end", v => v.toFixed(1), 0.15);

// ---------------------------------------------------------------- shared method column (legend for a, b + row labels of c), as the Fig 2 spine
const DASH = { solid: "solid", dashed: "dash", dotted: "sysDot", "dash-dot": "dashDot" };
const LSTYLE = Object.fromEntries(D.methods.map(m => [m.key, m.line]));
const CLASSES = [["DALIv2LE", "DALIv2", "LEonly"], ["SCBO"], ["TuRBO", "GA", "BO", "Random"], ["cTuRBOknown"]];
const BAND = { DALIv2LE: "E4EEF8", SCBO: "E2F3F1", TuRBO: "EFE8F5", cTuRBOknown: "FFF4D6" };
const colX = 9.35, colW = 1.5, cY0 = A.y0, rowH = A.h / methods.length;
text("Method (line style in a, b)", colX - 0.2, 0.12, colW + 0.4, 0.32, { align: "center", bold: true, fontSize: 9 });
line(colX + 0.02, 0.46, colX + colW - 0.02, 0.46, INK, 1.0);
for (const grp of CLASSES) {
  const ks = grp.map(m => methods.indexOf(m));
  const yTop = cY0 + Math.min(...ks) * rowH + 0.03, yBot = cY0 + (Math.max(...ks) + 1) * rowH - 0.03;
  s.addShape(pres.ShapeType.rect, { x: colX, y: yTop, w: colW, h: yBot - yTop, fill: { color: BAND[grp[0]] }, line: { color: BAND[grp[0]], width: 0 } });
}
methods.forEach((m, i) => {
  const yc = cY0 + i * rowH + rowH / 2;
  line(colX + 0.1, yc, colX + 0.42, yc, hex(m), m === "DALIv2LE" ? 1.75 : 1.25, DASH[LSTYLE[m]]);
  text(LABEL[m], colX + 0.5, cY0 + i * rowH, colW - 0.52, rowH, { fontSize: 8.5 });
});

// ---------------------------------------------------------------- (c) first 20 experiments, native (rows shared with the method column)
panelTitle("c", "First 20 experiments", colX + colW + 0.2, 0.12, 2.0);
const cX0 = colX + colW + 0.28, cW = 1.05;
const cx = v => cX0 + (v / 22) * cW;
line(cX0, cY0 + A.h, cx(22), cY0 + A.h, AXIS, 0.75);
[0, 5, 10, 15, 20].forEach(v => {
  line(cx(v), cY0 + A.h, cx(v), cY0 + A.h + 0.05, AXIS, 0.75);
  text(String(v), cx(v) - 0.2, cY0 + A.h + 0.06, 0.4, 0.15, { align: "center", fontSize: 8, color: MUTED });
});
text("Violations in first 20 exp.", cX0 - 0.45, cY0 + A.h + 0.22, cW + 0.9, 0.2, { align: "center", fontSize: 9.5 });
methods.forEach((m, i) => {
  const d = D.c[m], y = cY0 + i * rowH, bh = rowH * 0.62;
  line(colX + colW + 0.02, y + rowH / 2, cX0 - 0.02, y + rowH / 2, "BFBFBF", 0.75);   // connector tick, as Fig 2
  s.addShape(pres.ShapeType.rect, { x: cX0, y: y + (rowH - bh) / 2, w: Math.max(cx(d.mean) - cX0, 0.005), h: bh,
    fill: { color: hex(m) }, line: { color: INK, width: 0.5 } });
  d.per_config.forEach((v, k) => {
    const r = 0.022;
    s.addShape(pres.ShapeType.ellipse, { x: cx(v) - r, y: y + rowH / 2 + d.jitter[k] * rowH * 0.9 - r, w: 2 * r, h: 2 * r,
      fill: { color: "000000", transparency: 60 }, line: { color: "000000", width: 0 } });
  });
  const xv = cx(Math.max(d.mean, Math.max(...d.per_config))) + 0.06;
  text(d.mean.toFixed(1), xv, y, 0.45, rowH, { fontSize: 8.5 });
});

// ---------------------------------------------------------------- footnote
text("Main block: 16 configurations × 20 seeds = 320 runs per method. (a) Kaplan–Meier, censored at 100, band = Greenwood 95 % CI; " +
  "(b) mean over the 16 configurations, band = 95 % CI; (c) bars = mean over configurations, dots = configurations. Labels at the curve ends in (a, b): value at experiment 100; the method column gives the line style of each curve.",
  0.40, A.y0 + A.h + 0.55, W - 0.8, 0.34, { fontSize: 7.5, color: MUTED, valign: "top" });

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
