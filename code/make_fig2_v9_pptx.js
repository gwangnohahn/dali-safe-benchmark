/* Fig 2 v8 (2026-09-28) -- editable PowerPoint version of Fig 2.
 * v9 (2026-09-29): identical to v8 except the SI cross-reference ("values in Fig. S1") after the SI renumbering.
 *
 * Every element is a native shape or text box (no images), so the figure can be
 * edited in PowerPoint.  Data come from figures/Fig2_v8_data.json written by
 * make_fig2_v8_data.py (same numbers as Fig 2 v7).
 *
 * Layout follows the user's Fig 1 restyle (2026-09-28): panel letter + short title,
 * method classes marked with the same colour bands (Record-based / Constraint
 * learning / Constraint-unaware / Reference), wording "near boundary".
 * v8b: the method-name column sits BETWEEN the heat map (a) and the bars (b) and is
 * shared by both; the bars are coloured by class (same colours as the bands and Fig 1).
 *
 * Usage: node make_fig2_v8_pptx.js [dataJson] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const dataPath = process.argv[2] || path.join(CODE_DIR, "..", "figures", "Fig2_v8_data.json");
const outPath = process.argv[3] || path.join(CODE_DIR, "..", "figures", "Fig2_heatmap_v9.pptx");
const D = JSON.parse(fs.readFileSync(dataPath, "utf8"));

// ---------------------------------------------------------------- style
const FONT = "Arial";
const INK = "222222", MUTED = "595959", GRID = "D9D9D9";
const CLASSES = [ // same colours as the user's Fig 1
  { name: "Record-based", methods: ["DALIv2LE", "DALIv2", "LEonly"], text: "2E75B6", band: "E4EEF8" },
  { name: "Constraint learning", methods: ["SCBO"], text: "1E8C8C", band: "E2F3F1" },
  { name: "Constraint-unaware", methods: ["TuRBO", "GA", "BO", "Random"], text: "7B4FA6", band: "EFE8F5" },
  { name: "Reference (constraint given)", methods: ["cTuRBOknown"], text: "8A6D00", band: "FFF4D6" },
];
const classOf = m => CLASSES.find(c => c.methods.includes(m));
// colour ramp of Fig 2 v7 (CMAP_V): white -> fcbba1 -> cb181d -> 67000d
const RAMP = [[255, 255, 255], [252, 187, 161], [203, 24, 29], [103, 0, 13]];
function ramp(v) { // v in [0, 100]
  const t = Math.max(0, Math.min(1, v / 100)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(t), RAMP.length - 2), f = t - i;
  const c = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * f));
  return c.map(x => x.toString(16).padStart(2, "0")).join("").toUpperCase();
}
const fmt = v => (v >= 10 ? v.toFixed(0) : v.toFixed(1));

// ---------------------------------------------------------------- geometry (inches)
const W = 12.6, H = 6.05;
const rowH = 0.36, y0 = 1.02;                  // shared rows of (a) and (b)
const methods = D.methods.map(m => m.key);
const rowY = k => y0 + k * rowH;
// (a) heat map, left
const aX = 0.30;
const hmX = aX + 0.05;
const cw = 0.34, gap = 0.12;
const geoms = ["g1", "g2", "g3", "g4"];
const GEO_LABEL = { g1: "G1  Convex", g2: "G2  Non-convex", g3: "G3  Disconnected", g4: "G4  Thin band" };
const colX = j => hmX + Math.floor(j / 4) * (4 * cw + gap) + (j % 4) * cw;
const hmRight = colX(15) + cw;                              // 6.15
// shared method-name column, middle
const labelW = 1.16, bandPad = 0.06;
const midX = hmRight + 0.24;                                // band left edge
const nameX = midX + bandPad;                               // names
const bandRight = nameX + labelW + bandPad;
// (b) bars, right
const bX = bandRight + 0.20;
const bAxisX0 = bX, bAxisW = 3.55;                          // 0..100 violations
const bx = v => bAxisX0 + (v / 100) * bAxisW;

const pres = new pptxgen();
pres.defineLayout({ name: "FIG2", width: W, height: H });
pres.layout = "FIG2";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

function text(str, x, y, w, h, o = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 9, color: INK, margin: 0,
    isTextBox: true, valign: "middle", align: "left" }, o));
}
function rect(x, y, w, h, fill, line) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: fill },
    line: line ? { color: line.color, width: line.width } : { color: fill, width: 0 } });
}
function hline(x1, y, x2, color, width) { s.addShape(pres.ShapeType.line, { x: x1, y, w: x2 - x1, h: 0, line: { color, width } }); }
function vline(x, y1, y2, color, width) { s.addShape(pres.ShapeType.line, { x, y: y1, w: 0, h: y2 - y1, line: { color, width } }); }
function panelTitle(letter, title, x, y, w) {
  s.addText([{ text: letter, options: { bold: true, fontSize: 14 } }, { text: "   " + title, options: { fontSize: 11 } }],
    { x, y, w, h: 0.32, fontFace: FONT, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" });
}

// ---------------------------------------------------------------- panel titles
panelTitle("a", "Violations per configuration (mean of 20 seeds)", aX, 0.18, 6.0);
panelTitle("b", "Interior vs near-boundary mean", bX, 0.18, 4.0);

// ---------------------------------------------------------------- shared method column (class bands + names)
// The column is drawn as a spine shared by (a) and (b): a "Method" header like the
// geometry headers, class-tinted bands, and light connector ticks on BOTH sides of
// every row (into the heat map and into the bar axis).
text("Method", midX, 0.58, bandRight - midX, 0.26, { align: "center", bold: true, fontSize: 9.5 });
hline(midX + 0.02, 0.88, bandRight - 0.02, INK, 1.0);
for (const c of CLASSES) {
  const ks = c.methods.map(m => methods.indexOf(m)).filter(k => k >= 0);
  if (!ks.length) continue;
  const yTop = rowY(Math.min(...ks)) + 0.03, yBot = rowY(Math.max(...ks)) + rowH - 0.03;
  rect(midX, yTop, bandRight - midX, yBot - yTop, c.band);
}
methods.forEach((m, k) => {
  const yc = rowY(k) + rowH / 2;
  hline(hmRight + 0.03, yc, midX - 0.02, "BFBFBF", 0.75);          // tick towards (a)
  hline(bandRight + 0.02, yc, bX - 0.02, "BFBFBF", 0.75);          // tick towards (b)
  text(D.methods[k].label, nameX, rowY(k), labelW, rowH, { align: "center", fontSize: 9.5, color: INK });
});

// ---------------------------------------------------------------- (a) heat map
geoms.forEach((g, gi) => {
  const x1 = colX(gi * 4), x2 = colX(gi * 4 + 3) + cw;
  text(GEO_LABEL[g], x1, 0.58, x2 - x1, 0.26, { align: "center", bold: true, fontSize: 9.5 });
  hline(x1 + 0.02, 0.88, x2 - 0.02, INK, 1.0);
});
D.configs.forEach((cfg, j) => {
  methods.forEach((m, k) => {
    const v = D.V_mean[j][k], mk = D.marks[j][k];
    const x = colX(j), y = rowY(k);
    rect(x + 0.015, y + 0.015, cw - 0.03, rowH - 0.03, ramp(v));
    if (mk) {
      const dark = v > 60, d = 0.10;
      const opts = { x: x + cw / 2 - d / 2, y: y + rowH / 2 - d / 2, w: d, h: d };
      if (mk === "*") s.addShape(pres.ShapeType.ellipse, Object.assign(opts, { fill: { color: dark ? "FFFFFF" : INK }, line: { color: dark ? "FFFFFF" : INK, width: 0.5 } }));
      else s.addShape(pres.ShapeType.ellipse, Object.assign(opts, { fill: { type: "none" }, line: { color: dark ? "FFFFFF" : INK, width: 1.25 } }));
    }
  });
});
const yCol = rowY(methods.length) + 0.04;
D.configs.forEach((cfg, j) => {
  const bnd = cfg.location === "bnd";
  text(String(cfg.fraction_pct), colX(j), yCol, cw, 0.2, { align: "center", fontSize: 9, bold: bnd });
  if (bnd) text("near\nbnd.", colX(j) - 0.06, yCol + 0.18, cw + 0.12, 0.3, { align: "center", fontSize: 7.5, bold: true });
});
text("Feasible fraction of the box (%)", hmX, yCol + 0.50, hmRight - hmX, 0.22, { align: "center", fontSize: 9.5 });
// colour bar (stepped rectangles, editable) + significance key
const cbY = yCol + 0.88, cbX = hmX + 0.3, cbW = 2.0, cbH = 0.13, nStep = 24;
for (let i = 0; i < nStep; i++) rect(cbX + (i / nStep) * cbW, cbY, cbW / nStep + 0.004, cbH, ramp(((i + 0.5) / nStep) * 100));
s.addShape(pres.ShapeType.rect, { x: cbX, y: cbY, w: cbW, h: cbH, fill: { type: "none" }, line: { color: "BFBFBF", width: 0.5 } });
[0, 50, 100].forEach(v => text(String(v), cbX + (v / 100) * cbW - 0.2, cbY + cbH + 0.01, 0.4, 0.16, { align: "center", fontSize: 8, color: MUTED }));
text("Violations in 100 experiments", cbX, cbY + cbH + 0.17, cbW, 0.18, { align: "center", fontSize: 8.5, color: MUTED });
const kx = cbX + cbW + 0.45, ky = cbY - 0.02;
s.addShape(pres.ShapeType.ellipse, { x: kx, y: ky + 0.03, w: 0.10, h: 0.10, fill: { color: INK }, line: { color: INK, width: 0.5 } });
text("more than DALI+LE", kx + 0.15, ky, 1.3, 0.16, { fontSize: 8.5, color: MUTED });
s.addShape(pres.ShapeType.ellipse, { x: kx + 1.45, y: ky + 0.03, w: 0.10, h: 0.10, fill: { type: "none" }, line: { color: INK, width: 1.25 } });
text("fewer than DALI+LE", kx + 1.60, ky, 1.3, 0.16, { fontSize: 8.5, color: MUTED });
text("Holm-adjusted p < 0.05; values in Fig. S1", kx, ky + 0.20, 3.0, 0.16, { fontSize: 8, color: MUTED });

// ---------------------------------------------------------------- (b) bars, coloured by class
const yTopB = rowY(0), yBotB = rowY(methods.length);
[25, 50, 75, 100].forEach(v => vline(bx(v), yTopB, yBotB, GRID, 0.5));
vline(bx(0), yTopB, yBotB, MUTED, 0.75);
const barH = 0.13;
methods.forEach((m, k) => {
  const col = classOf(m).text;
  [["int", false], ["bnd", true]].forEach(([loc, filled], i) => {
    const b = D.bars[loc][m];
    const y = rowY(k) + 0.03 + i * (barH + 0.03);
    const w = Math.max(bx(b.mean) - bx(0), 0.01);
    s.addShape(pres.ShapeType.rect, { x: bx(0), y, w, h: barH,
      fill: filled ? { color: col } : { color: "FFFFFF" },
      line: { color: col, width: filled ? 0.5 : 1.5 } });
    if (b.hi - b.lo > 0.5) {
      hline(bx(b.lo), y + barH / 2, bx(b.hi), INK, 0.75);
      vline(bx(b.lo), y + 0.02, y + barH - 0.02, INK, 0.75);
      vline(bx(b.hi), y + 0.02, y + barH - 0.02, INK, 0.75);
    }
    text(fmt(b.mean), bx(Math.max(b.hi, b.mean)) + 0.05, y - 0.02, 0.5, barH + 0.04, { fontSize: 8.5 });
  });
});
hline(bx(0), yBotB, bx(100), MUTED, 0.75);
[0, 25, 50, 75, 100].forEach(v => {
  vline(bx(v), yBotB, yBotB + 0.05, MUTED, 0.75);
  text(String(v), bx(v) - 0.25, yBotB + 0.06, 0.5, 0.16, { align: "center", fontSize: 8, color: MUTED });
});
text("Violations in 100 experiments", bAxisX0, yBotB + 0.26, bAxisW, 0.22, { align: "center", fontSize: 9.5 });

// legend: interior / near-boundary key + class colours (shared with the bands and Fig 1)
const lgX = bAxisX0, lgY = yBotB + 0.60, lgW = 3.95, lgH = 0.98, lh = 0.2;
s.addShape(pres.ShapeType.rect, { x: lgX, y: lgY, w: lgW, h: lgH, fill: { type: "none" }, line: { color: "BFBFBF", width: 0.5 } });
let ly = lgY + 0.08;
s.addShape(pres.ShapeType.rect, { x: lgX + 0.15, y: ly + 0.03, w: 0.32, h: 0.13, fill: { color: "FFFFFF" }, line: { color: MUTED, width: 1.5 } });
text("Interior (12 config.)", lgX + 0.55, ly, 1.45, lh, { fontSize: 8.5 });
s.addShape(pres.ShapeType.rect, { x: lgX + 2.05, y: ly + 0.03, w: 0.32, h: 0.13, fill: { color: "8C8C8C" }, line: { color: "8C8C8C", width: 0.5 } });
text("Near boundary (4 config.)", lgX + 2.45, ly, 1.5, lh, { fontSize: 8.5 });
ly += lh + 0.06;
hline(lgX + 0.15, ly, lgX + lgW - 0.15, "E0E0E0", 0.5);
ly += 0.05;
CLASSES.forEach((c, i) => {
  const cx = lgX + 0.15 + (i % 2) * 1.9, cy = ly + Math.floor(i / 2) * (lh + 0.04);
  rect(cx, cy + 0.045, 0.13, 0.13, c.text);
  text(c.name, cx + 0.2, cy, 1.75, lh, { fontSize: 8.5, color: c.text, bold: true });
});
text("Bars: mean over configurations; error bars: 95 % bootstrap CI over seeds", lgX, lgY + lgH + 0.05, lgW + 0.3, 0.18, { fontSize: 7.5, color: MUTED });

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
