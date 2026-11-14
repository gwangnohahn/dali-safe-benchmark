/* Fig 5 v2 (2026-09-28) -- PowerPoint version of Fig 5 (mechanism).
 * v3 (2026-09-29): identical to v2 except the (d) footnote: seed counts (nine methods 20, two controls 10) and the three upper-left G2 points (DALI, DALI (no clip), DALI+LE).
 *
 * (a) bars drawn natively (Fig 2 column layout: G1-G4 x 50/20/5/20 near bnd.);
 * (b)+(c) one row block: shared method column (Fig 2 / Fig 4 spine) | four margin panels
 *     (images, horizontal boxes, rows = methods) | d_kNN panel (image);
 * (d) ablation scatter image + editable annotations and a class-grouped legend.
 * Data and images: figures/Fig5_v2_data.json, Fig5_v2_panel_*.png (make_fig5_v2.py).
 *
 * Usage: node make_fig5_v2_pptx.js [figdir] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const figdir = process.argv[2] || path.join(CODE_DIR, "..", "figures");
const outPath = process.argv[3] || path.join(figdir, "Fig5_mechanism_v3.pptx");
const D = JSON.parse(fs.readFileSync(path.join(figdir, "Fig5_v2_data.json"), "utf8"));

const FONT = "Arial", INK = "222222", MUTED = "595959", AXIS = "808080";
const methods = D.methods.map(m => m.key);
const LABEL = Object.fromEntries(D.methods.map(m => [m.key, m.label]));
const COLOR = Object.fromEntries(D.methods.map(m => [m.key, m.color.replace("#", "").toUpperCase()]));
const BAND = { DALIv2LE: "E4EEF8", SCBO: "E2F3F1", TuRBO: "EFE8F5", cTuRBOknown: "FFF4D6" };
const CLASSES = [["DALIv2LE", "DALIv2", "LEonly"], ["SCBO"], ["TuRBO", "GA", "BO", "Random"], ["cTuRBOknown"]];
const GEO_LABEL = { g1: "G1  Convex", g2: "G2  Non-convex", g3: "G3  Disconnected", g4: "G4  Thin band" };

const pres = new pptxgen();
const W = 12.9, H = 12.75;
pres.defineLayout({ name: "FIG5", width: W, height: H });
pres.layout = "FIG5";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

function text(str, x, y, w, h, o = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 9, color: INK, margin: 0,
    isTextBox: true, valign: "middle", align: "left" }, o));
}
function rect(x, y, w, h, fill, lineCol, lw) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: fill ? { color: fill } : { type: "none" },
    line: { color: lineCol || fill, width: lw === undefined ? 0 : lw } });
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
function ylabel(str, x, yc, len) {
  s.addText(str, { x: x - len / 2 + 0.1, y: yc - 0.1, w: len, h: 0.2, rotate: 270, fontFace: FONT, fontSize: 9.5,
    color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
}
function placeImage(name, key, imgX, imgY, scale) {
  const g = D.geom[key], [fw, fh] = g.fig, [ax0, ay0, aw, ah] = g.ax;
  s.addImage({ path: path.join(figdir, name), x: imgX, y: imgY, w: fw * scale, h: fh * scale });
  return { x0: imgX + ax0 * scale, y0: imgY + (fh - ay0 - ah) * scale, w: aw * scale, h: ah * scale, imgW: fw * scale, imgH: fh * scale };
}

// ================================================================ (a) candidate feasible fraction
panelTitle("a", "Feasible fraction of raw candidates (before clip and verifier)", 0.30, 0.12, 8.0);
const aX0 = 0.95, aY0 = 0.98, aH = 1.85, cw = 0.54, ggap = 0.30;
const cfg = D.a.configs;
const colX = j => aX0 + Math.floor(j / 4) * (4 * cw + ggap) + (j % 4) * cw;
const aRight = colX(15) + cw;
const ay = v => aY0 + aH - v * aH;
// axis
line(aX0 - 0.05, aY0, aX0 - 0.05, aY0 + aH, AXIS, 0.75);
[0, 0.5, 1.0].forEach(v => { line(aX0 - 0.1, ay(v), aX0 - 0.05, ay(v), AXIS, 0.75); text(v === 1 ? "1" : String(v), aX0 - 0.5, ay(v) - 0.08, 0.38, 0.16, { align: "right", fontSize: 8, color: MUTED }); });
line(aX0 - 0.05, ay(0), aRight, ay(0), AXIS, 0.75);
line(aX0, ay(1), aRight, ay(1), "BFBFBF", 0.5, "sysDot");
ylabel("Feasible fraction of raw candidates", aX0 - 0.62, aY0 + aH / 2, 2.4);
// geometry headers
["g1", "g2", "g3", "g4"].forEach((g, gi) => {
  const x1 = colX(gi * 4), x2 = colX(gi * 4 + 3) + cw;
  text(GEO_LABEL[g], x1, aY0 - 0.42, x2 - x1, 0.24, { align: "center", bold: true, fontSize: 9.5 });
  line(x1 + 0.02, aY0 - 0.15, x2 - 0.02, aY0 - 0.15, INK, 1.0);
});
// bars: DALI (hollow blue), DALI+LE (blue), SCBO (teal); uniform reference = black tick
const bw = 0.14;
cfg.forEach((c, j) => {
  const xc = colX(j) + cw / 2;
  [["DALIv2", true], ["DALIv2LE", false], ["SCBO", false]].forEach(([m, hollow], k) => {
    const v = D.a.series[m][j]; if (v === null || v === undefined) return;
    const x = xc + (k - 1) * (bw + 0.02) - bw / 2;
    s.addShape(pres.ShapeType.rect, { x, y: ay(v), w: bw, h: Math.max(ay(0) - ay(v), 0.003),
      fill: { color: hollow ? "FFFFFF" : COLOR[m] }, line: { color: hollow ? COLOR[m] : INK, width: hollow ? 1.25 : 0.5 } });
  });
  line(xc - 0.21, ay(c.uniform), xc + 0.21, ay(c.uniform), "000000", 1.75);
  const bnd = c.location === "bnd";
  text(String(c.fraction_pct), colX(j), aY0 + aH + 0.05, cw, 0.18, { align: "center", fontSize: 9, bold: bnd });
  if (bnd) text("near\nbnd.", colX(j) - 0.05, aY0 + aH + 0.21, cw + 0.1, 0.3, { align: "center", fontSize: 7.5, bold: true });
});
text("Feasible fraction of the box (%)", aX0, aY0 + aH + 0.55, aRight - aX0, 0.2, { align: "center", fontSize: 9.5 });
// legend box for (a)
const laW = 1.9, laX = 12.6 - laW, laH = 1.12, laY = aY0 + (aH - laH) / 2;
rect(laX, laY, laW, laH, null, "BFBFBF", 0.5);
let ly = laY + 0.1;
[["DALIv2", true, "DALI"], ["DALIv2LE", false, "DALI+LE"], ["SCBO", false, "SCBO"]].forEach(([m, hollow, lab]) => {
  s.addShape(pres.ShapeType.rect, { x: laX + 0.15, y: ly + 0.03, w: 0.3, h: 0.14, fill: { color: hollow ? "FFFFFF" : COLOR[m] }, line: { color: hollow ? COLOR[m] : INK, width: hollow ? 1.25 : 0.5 } });
  text(lab, laX + 0.55, ly, laW - 0.6, 0.2, { fontSize: 8.5 }); ly += 0.22;
});
line(laX + 0.15, ly + 0.1, laX + 0.45, ly + 0.1, "000000", 1.75);
text("Uniform in box\n(= feasible fraction)", laX + 0.55, ly - 0.02, laW - 0.6, 0.38, { fontSize: 8.5, valign: "top" });

// ================================================================ (b)+(c) row block with shared method column
const rowTop = 3.85;
panelTitle("b", "Margin g(x) of proposals (interior · 20 %)", 1.85, rowTop, 6.0);
const SC = 1.09;
const bImgY = rowTop + 0.62;
const bX = k => 1.95 + k * (D.geom.b.fig[0] * SC + 0.08);
let Bax = null;
["g1", "g2", "g3", "g4"].forEach((g, k) => {
  const A = placeImage(`Fig5_v2_panel_b_${g}.png`, "b", bX(k), bImgY, SC);
  Bax = Bax || A;
  text(GEO_LABEL[g], A.x0, bImgY - 0.26, A.w, 0.22, { align: "center", bold: true, fontSize: 9 });
  line(A.x0 + 0.02, bImgY - 0.02, A.x0 + A.w - 0.02, bImgY - 0.02, INK, 1.0);
});
const cImgX = bX(4) + 0.12;
const Cax = placeImage("Fig5_v2_panel_c.png", "c", cImgX, bImgY, SC);
panelTitle("c", "Distance to record", cImgX, rowTop, 3.0);
text("d_kNN (log axis)", Cax.x0, bImgY - 0.26, Cax.w, 0.22, { align: "center", bold: true, fontSize: 9 });
line(Cax.x0 + 0.02, bImgY - 0.02, Cax.x0 + Cax.w - 0.02, bImgY - 0.02, INK, 1.0);
text("Margin g(x) of the 100 proposals (> 0 = feasible; dashed line = boundary)", bX(0), Bax.y0 + Bax.h + 0.24, bX(4) - bX(0) - 0.08, 0.2, { align: "center", fontSize: 9.5 });
text("one point per run (12 interior configs.)", Cax.x0 - 0.15, Cax.y0 + Cax.h + 0.24, Cax.w + 0.3, 0.2, { align: "center", fontSize: 9 });
// shared method column (rows aligned with the image rows)
const rowH = Bax.h / methods.length, colX0 = 0.32, colW = 1.35;
text("Method", colX0, bImgY - 0.26, colW, 0.22, { align: "center", bold: true, fontSize: 9 });
line(colX0 + 0.02, bImgY - 0.02, colX0 + colW - 0.02, bImgY - 0.02, INK, 1.0);
for (const grp of CLASSES) {
  const ks = grp.map(m => methods.indexOf(m));
  const yTop = Bax.y0 + Math.min(...ks) * rowH + 0.02, yBot = Bax.y0 + (Math.max(...ks) + 1) * rowH - 0.02;
  rect(colX0, yTop, colW, yBot - yTop, BAND[grp[0]]);
}
methods.forEach((m, i) => {
  const yc = Bax.y0 + i * rowH + rowH / 2;
  text(LABEL[m], colX0, Bax.y0 + i * rowH, colW, rowH, { align: "center", fontSize: 8.5 });
  line(colX0 + colW + 0.02, yc, Bax.x0 - 0.02, yc, "BFBFBF", 0.75);
});

// ================================================================ (d) ablation
const dTop = Bax.y0 + Bax.h + 0.75;
panelTitle("d", "Ablation: verifier or clip cannot replace the generator (5 configurations)", 0.30, dTop, 7.5);
const Dax = placeImage("Fig5_v2_panel_d.png", "d", 0.55, dTop + 0.4, 1.0);
const [dxl, dxh] = D.d.xlim, [dyl, dyh] = D.d.ylim;
const DX = v => Dax.x0 + (Math.log10(v) - Math.log10(dxl)) / (Math.log10(dxh) - Math.log10(dxl)) * Dax.w;
const DY = v => Dax.y0 + Dax.h - (v - dyl) / (dyh - dyl) * Dax.h;
ylabel("Violations in 100 experiments", Dax.x0 - 0.5, Dax.y0 + Dax.h / 2, 2.4);
text("Distance to record, d_kNN (log)", Dax.x0, Dax.y0 + Dax.h + 0.24, Dax.w, 0.2, { align: "center", fontSize: 9.5 });
// annotations
const pt = (m, c) => D.d.points.find(p => p.method === m && p.config === c);
for (const m of ["DALIv2", "DALIv2LE"]) {   // the only two failures of the record-based methods: keep this label
  const p = pt(m, "g2_f020_bnd");
  text("G2, near bnd.", DX(p.dknn) + 0.14, DY(p.V) - 0.09, 1.0, 0.18, { fontSize: 8, color: MUTED });
}
const vu = D.d.verif_uniform_mean;
text(`Spearman ρ = ${D.d.spearman_rho.toFixed(2)}`, Dax.x0 + Dax.w - 1.5, Dax.y0 + Dax.h - 0.5, 1.45, 0.2, { align: "right", fontSize: 9 });
// legend for (d): class-grouped, two columns, dotted separators (Fig 3 style)
const MK = { DALIv2LE: "ellipse", DALIv2: "ellipse", LEonly: "rect", DALInoclip: "x", VerifRandom: "hexagon", SCBO: "diamond",
  TuRBO: "triangle", GA: "triangleDown", BO: "mathPlus", Random: "mathMultiply", cTuRBOknown: "star5" };
const HOLLOW = new Set(["DALIv2", "cTuRBOknown"]);
function marker(m, cx, cy, size) {
  const col = D.d.colors[m].replace("#", "").toUpperCase(), sz = size * (m === "cTuRBOknown" ? 1.5 : 1);
  if (m === "DALInoclip") { line(cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2, col, 1.75); line(cx - sz / 2, cy + sz / 2, cx + sz / 2, cy - sz / 2, col, 1.75); return; }
  const o = { x: cx - sz / 2, y: cy - sz / 2, w: sz, h: sz, fill: { color: HOLLOW.has(m) ? "FFFFFF" : col },
    line: { color: HOLLOW.has(m) ? col : INK, width: HOLLOW.has(m) ? 1.5 : 0.5 } };
  let shape = MK[m]; if (shape === "triangleDown") { shape = "triangle"; o.flipV = true; }
  s.addShape(pres.ShapeType[shape], o);
}
const LG = ["DALIv2LE", "DALIv2", "LEonly", "DALInoclip", null, "VerifRandom", null, "SCBO", null, "TuRBO", "GA", "BO", "Random", null, "cTuRBOknown"];
const lgW = 1.75, lgX = 12.6 - lgW, lrow = 0.2, lsep = 0.08;
const lgH = 0.14 + 11 * lrow + 4 * lsep + 0.02, lgY = Dax.y0 + (Dax.h - lgH) / 2;
rect(lgX, lgY, lgW, lgH, "FFFFFF", "D9D9D9", 0.5);
{
  let y = lgY + 0.08; const x = lgX + 0.12;
  for (const m of LG) {
    if (m === null) { line(x, y + lsep / 2, x + lgW - 0.24, y + lsep / 2, "BFBFBF", 0.5, "sysDot"); y += lsep; continue; }
    marker(m, x + 0.12, y + lrow / 2, 0.13);
    text(D.d.labels[m], x + 0.32, y, lgW - 0.5, lrow, { fontSize: 8.5 });
    y += lrow;
  }
}

// ================================================================ footnote
const fnY = Dax.y0 + Dax.h + 0.55;
text("(a) Mean over cycles and seeds; the black tick is the feasible fraction of the box, i.e. the value expected for uniform candidates. " +
  "(b) All 100 proposals × 20 seeds pooled: box = quartiles, whiskers = 5–95 %, line = median. (c) Mean kNN₅ distance of a run's proposals to the warm start (standardized coordinates), 12 interior configurations × 20 seeds; boxes as in (b). " +
  "(d) One point per (configuration, method), interior 20 % × 4 + G2 near boundary; 20 seeds for the nine methods of the main block, 10 seeds for the two controls. Controls: verifier + uniform keeps DALI's verifier and selection but draws the raw candidates uniformly in the box " +
  `(${Math.round(vu.cand_feasible * 100)} % of them feasible = the feasible fraction of the box); DALI (no clip) proposes the raw candidates without clipping. The upper-left region (close to the record yet many violations) is empty except for the G2 near-boundary points of DALI, DALI (no clip) and DALI+LE.`,
  0.40, fnY, W - 0.8, 0.7, { fontSize: 7.5, color: MUTED, valign: "top" });

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
