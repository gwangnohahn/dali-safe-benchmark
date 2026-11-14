/* Fig 7 v2 (2026-09-28) -- PowerPoint version of Fig 7 (real-process datasets).
 * v3 (2026-09-29): identical to v2 except the NOMC label "empirical f̂* at 1.26 (off scale)" (1.26 is the position of f̂* on the improvement axis, not f̂* itself).
 *
 * Row 1 (a-c): cumulative-violation curves per dataset (images from make_fig7_v2.py) with
 *   numbers at the curve ends (editable).
 * Row 2 (d-f): final improvement over the warm-start best per method, drawn natively as
 *   horizontal bars (mean, 95 % bootstrap CI over seeds) beside a shared method column
 *   (Fig 2 / 4 / 5 spine) whose line samples are the legend of row 1.  Dotted line = empirical
 *   f^* (off scale for NOMC: quoted as a number).
 * Usage: node make_fig7_v2_pptx.js [figdir] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const figdir = process.argv[2] || path.join(CODE_DIR, "..", "figures");
const outPath = process.argv[3] || path.join(figdir, "Fig7_realdata_v3.pptx");
const D = JSON.parse(fs.readFileSync(path.join(figdir, "Fig7_v2_data.json"), "utf8"));

const FONT = "Arial", INK = "222222", MUTED = "595959", AXIS = "808080", GRID = "D9D9D9";
const methods = D.methods.map(m => m.key);
const LABEL = Object.fromEntries(D.methods.map(m => [m.key, m.label]));
const COLOR = Object.fromEntries(D.methods.map(m => [m.key, m.color.replace("#", "").toUpperCase()]));
const LSTYLE = Object.fromEntries(D.methods.map(m => [m.key, m.line]));
const DASH = { solid: "solid", dashed: "dash", dotted: "sysDot", "dash-dot": "dashDot" };
const hex = m => (m === "cTuRBOknown" ? "000000" : COLOR[m]);
const BAND = { DALIv2LE: "E4EEF8", SCBO: "E2F3F1", TuRBO: "EFE8F5", cTuRBOknown: "FFF4D6" };
const CLASSES = [["DALIv2LE", "DALIv2", "LEonly"], ["SCBO"], ["TuRBO", "GA", "BO", "Random"], ["cTuRBOknown"]];

const pres = new pptxgen();
const W = 12.9, H = 7.75;
pres.defineLayout({ name: "FIG7", width: W, height: H });
pres.layout = "FIG7";
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
function panelTitle(letter, title, x, y, w) {
  s.addText([{ text: letter, options: { bold: true, fontSize: 14 } }, { text: "   " + title, options: { fontSize: 11 } }],
    { x, y, w, h: 0.3, fontFace: FONT, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" });
}
function ylabel(str, x, yc, len) {
  s.addText(str, { x: x - len / 2 + 0.1, y: yc - 0.1, w: len, h: 0.2, rotate: 270, fontFace: FONT, fontSize: 9, color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
}
function placeImage(name, key, imgX, imgY) {
  const g = D.geom[key], [fw, fh] = g.fig, [ax0, ay0, aw, ah] = g.ax;
  s.addImage({ path: path.join(figdir, name), x: imgX, y: imgY, w: fw, h: fh });
  return { x0: imgX + ax0, y0: imgY + (fh - ay0 - ah), w: aw, h: ah };
}
function stack(ys, minSep, yMin, yMax) {
  const idx = ys.map((y, i) => i).sort((a, b) => ys[a] - ys[b]);
  const pos = idx.map(i => ys[i]);
  for (let k = 1; k < pos.length; k++) if (pos[k] - pos[k - 1] < minSep) pos[k] = pos[k - 1] + minSep;
  if (yMax !== undefined && pos[pos.length - 1] > yMax) { pos[pos.length - 1] = yMax; for (let k = pos.length - 2; k >= 0; k--) pos[k] = Math.min(pos[k], pos[k + 1] - minSep); }
  if (yMin !== undefined && pos[0] < yMin) { const d = yMin - pos[0]; for (let k = 0; k < pos.length; k++) pos[k] += d; }
  const out = new Array(ys.length); idx.forEach((i, k) => out[i] = pos[k]); return out;
}

// ================================================================ row 1: violation curves
const r1Top = 0.15, r1Img = 0.55;
const slot = 4.15, x0s = [0.3, 0.3 + slot, 0.3 + 2 * slot];
const axesTop = [];
D.datasets.forEach((d, k) => {
  panelTitle("abc"[k], d.title, x0s[k] - 0.05, r1Top, 4.0);
  const A = placeImage(`Fig7_v2_panel_${d.key}.png`, "top", x0s[k], r1Img);
  axesTop.push(A);
  if (k === 0) ylabel("Cumulative violations", A.x0 - 0.42, A.y0 + A.h / 2, 2.0);
  text("Experiment", A.x0, A.y0 + A.h + 0.22, A.w, 0.2, { align: "center", fontSize: 9 });
  const Y = v => A.y0 + A.h - (v - d.ylim[0]) / (d.ylim[1] - d.ylim[0]) * A.h;
  const xEnd = A.x0 + A.w;
  const ys = methods.map(m => Y(d.V_end[m]));
  const pos = stack(ys, 0.135, A.y0 - 0.02, A.y0 + A.h + 0.02);
  methods.forEach((m, i) => {
    if (Math.abs(pos[i] - ys[i]) > 1e-3) line(xEnd + 0.02, ys[i], xEnd + 0.10, pos[i], "A6A6A6", 0.5);
    text(d.V_end[m].toFixed(1), xEnd + 0.12, pos[i] - 0.07, 0.45, 0.14, { fontSize: 7.5, color: hex(m) });
  });
});

// ================================================================ row 2: method column + improvement bars
const r2Top = axesTop[0].y0 + axesTop[0].h + 0.6, hdrY = r2Top + 0.42, rowsY = hdrY + 0.36;
const rowH = 0.27, nrow = methods.length, rowsH = rowH * nrow;
// shared method column (legend of row 1: line samples)
const colX = 0.32, colW = 1.5;
text("Method (line style in a–c)", colX - 0.2, hdrY, colW + 0.4, 0.24, { align: "center", bold: true, fontSize: 9 });
line(colX + 0.02, hdrY + 0.28, colX + colW - 0.02, hdrY + 0.28, INK, 1.0);
for (const grp of CLASSES) {
  const ks = grp.map(m => methods.indexOf(m));
  const yTop = rowsY + Math.min(...ks) * rowH + 0.02, yBot = rowsY + (Math.max(...ks) + 1) * rowH - 0.02;
  rect(colX, yTop, colW, yBot - yTop, BAND[grp[0]]);
}
methods.forEach((m, i) => {
  const yc = rowsY + i * rowH + rowH / 2;
  line(colX + 0.1, yc, colX + 0.42, yc, hex(m), m === "DALIv2LE" ? 1.75 : 1.25, DASH[LSTYLE[m]]);
  text(LABEL[m], colX + 0.5, rowsY + i * rowH, colW - 0.52, rowH, { fontSize: 8.5 });
});
// bar panels, one per dataset, centred under the curve panels
const barW = 2.55;
D.datasets.forEach((d, k) => {
  const A = axesTop[k];
  const bx0 = (k === 0) ? colX + colW + 0.55 : A.x0 + (A.w - barW) / 2 + 0.15;
  panelTitle("def"[k], d.unit, bx0 - 0.45, r2Top, 3.2);
  const vals = methods.map(m => d.improvement[m]);
  const vmax = Math.max(...vals.map(v => v.hi), d.fstar_gap < 3 * Math.max(...vals.map(v => v.mean)) ? d.fstar_gap : 0);
  const xmax = vmax * 1.15;
  const bx = v => bx0 + Math.max(v, 0) / xmax * barW;
  // axis + grid
  const yAx = rowsY + rowsH;
  line(bx0, rowsY, bx0, yAx, AXIS, 0.75);
  line(bx0, yAx, bx0 + barW, yAx, AXIS, 0.75);
  const nice = xmax > 6 ? 2 : xmax > 3 ? 1 : xmax > 1 ? 0.5 : xmax > 0.3 ? 0.1 : 0.05;
  for (let t = 0; t <= xmax + 1e-9; t += nice) {
    if (t > 0) line(bx(t), rowsY, bx(t), yAx, GRID, 0.5);
    line(bx(t), yAx, bx(t), yAx + 0.05, AXIS, 0.75);
    text(nice < 1 ? t.toFixed(nice < 0.1 ? 2 : 1) : String(t), bx(t) - 0.25, yAx + 0.06, 0.5, 0.15, { align: "center", fontSize: 7.5, color: MUTED });
  }
  text("Improvement over warm-start best" + (k === 2 ? " (MW)" : ""), bx0 - 0.3, yAx + 0.24, barW + 0.6, 0.2, { align: "center", fontSize: 9 });
  // connector ticks from the column (k = 0 only, as in Fig 2)
  if (k === 0) methods.forEach((m, i) => line(colX + colW + 0.02, rowsY + i * rowH + rowH / 2, bx0 - 0.02, rowsY + i * rowH + rowH / 2, "BFBFBF", 0.75));
  // bars
  methods.forEach((m, i) => {
    const v = d.improvement[m], y = rowsY + i * rowH + rowH * 0.2, bh = rowH * 0.6;
    s.addShape(pres.ShapeType.rect, { x: bx0, y, w: Math.max(bx(v.mean) - bx0, 0.004), h: bh, fill: { color: COLOR[m] }, line: { color: INK, width: 0.5 } });
    if (v.hi - v.lo > 1e-9) {
      line(bx(v.lo), y + bh / 2, bx(v.hi), y + bh / 2, INK, 0.75);
      line(bx(v.lo), y + 0.02, bx(v.lo), y + bh - 0.02, INK, 0.75);
      line(bx(v.hi), y + 0.02, bx(v.hi), y + bh - 0.02, INK, 0.75);
    }
    const lab = xmax < 1 ? v.mean.toFixed(2) : v.mean.toFixed(1);
    text(lab, bx(Math.max(v.hi, v.mean)) + 0.05, y - 0.02, 0.5, bh + 0.04, { fontSize: 8 });
  });
  // empirical f* reference
  if (d.fstar_gap <= xmax) {
    line(bx(d.fstar_gap), rowsY - 0.05, bx(d.fstar_gap), yAx, "404040", 1.0, "sysDot");
    text("empirical f̂*", bx(d.fstar_gap) - 0.6, rowsY - 0.24, 1.2, 0.18, { align: "center", fontSize: 8, color: MUTED });
  } else {
    text(`empirical f̂* at ${d.fstar_gap.toFixed(2)} (off scale)`, bx0, rowsY - 0.24, barW + 0.6, 0.18, { fontSize: 8, color: MUTED });
  }
});

// ================================================================ footnote
text("Feasibility: kNN₅ density oracle at the 99th percentile (feasible fraction of the box: SFS 0.14, NOMC 0.14, CCPP 0.11); 10 seeds. (a–c) Mean cumulative violations, band = 95 % CI over seeds; numbers = value at experiment 100. " +
  "(d–f) Final feasible best relative to the best of the 200 warm-start points (post-hoc metric), mean and 95 % bootstrap CI over seeds; dotted line = empirical f̂* (Methods 4.3), off scale for NOMC. Random improves by 0 on all three datasets.",
  0.35, rowsY + rowsH + 0.6, W - 0.7, 0.45, { fontSize: 7.5, color: MUTED, valign: "top" });

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
