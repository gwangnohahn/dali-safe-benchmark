/* Fig 3 v2 (2026-09-28) -- editable PowerPoint version of Fig 3 (safety-performance plane).
 *
 * All elements are native shapes / text boxes.  Data: figures/Fig3_v2_data.json from
 * make_fig3_v2_data.py (same numbers as Fig 3 v1).  Style follows Fig 1 / Fig 2 v8:
 * panel letter + short title, four class colours (Record-based / Constraint learning /
 * Constraint-unaware / Reference) with marker shapes distinguishing the methods
 * within a class, "near boundary" wording, legend grouped by class.
 * (a) 16-configuration mean per method with 95 % bootstrap CIs and the non-dominated
 *     frontier, plus a zoom inset of the low-regret cluster; (b) the four conditions.
 *
 * Usage: node make_fig3_v2_pptx.js [dataJson] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const dataPath = process.argv[2] || path.join(CODE_DIR, "..", "figures", "Fig3_v2_data.json");
const outPath = process.argv[3] || path.join(CODE_DIR, "..", "figures", "Fig3_frontier_v2.pptx");
const D = JSON.parse(fs.readFileSync(dataPath, "utf8"));

// ---------------------------------------------------------------- style
const FONT = "Arial";
const INK = "222222", MUTED = "595959", AXIS = "808080";
const CLASSES = [
  { name: "Record-based", methods: ["DALIv2LE", "DALIv2", "LEonly"], color: "2E75B6" },
  { name: "Constraint learning", methods: ["SCBO"], color: "1E8C8C" },
  { name: "Constraint-unaware", methods: ["TuRBO", "GA", "BO", "Random"], color: "7B4FA6" },
  { name: "Reference (constraint given)", methods: ["cTuRBOknown"], color: "8A6D00" },
];
const classOf = m => CLASSES.find(c => c.methods.includes(m));
const LABEL = Object.fromEntries(D.methods.map(m => [m.key, m.label]));
// marker shape per method (hollow = white fill, class-colour outline)
const MARK = {
  DALIv2LE: { shape: "ellipse" }, DALIv2: { shape: "ellipse", hollow: true }, LEonly: { shape: "rect" },
  SCBO: { shape: "diamond" }, TuRBO: { shape: "triangle" }, GA: { shape: "triangle", flipV: true },
  BO: { shape: "mathPlus" }, Random: { shape: "mathMultiply" }, cTuRBOknown: { shape: "star5", hollow: true, big: 1.5 },
};

const pres = new pptxgen();
const W = 12.6, H = 5.6;
pres.defineLayout({ name: "FIG3", width: W, height: H });
pres.layout = "FIG3";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

function text(str, x, y, w, h, o = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 9, color: INK, margin: 0,
    isTextBox: true, valign: "middle", align: "left" }, o));
}
function line(x1, y1, x2, y2, color, width, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color, width, dashType: dash || "solid" } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;      // rising line
  s.addShape(pres.ShapeType.line, o);
}
function panelTitle(letter, title, x, y, w) {
  s.addText([{ text: letter, options: { bold: true, fontSize: 14 } }, { text: "   " + title, options: { fontSize: 11 } }],
    { x, y, w, h: 0.32, fontFace: FONT, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" });
}
function marker(m, cx, cy, size) {
  const mk = MARK[m], col = classOf(m).color, sz = size * (mk.big || 1);
  const o = { x: cx - sz / 2, y: cy - sz / 2, w: sz, h: sz,
    fill: { color: mk.hollow ? "FFFFFF" : col },
    line: { color: mk.hollow ? col : INK, width: mk.hollow ? 1.5 : 0.5 } };
  if (mk.flipV) o.flipV = true;
  s.addShape(pres.ShapeType[mk.shape], o);
}

// ---------------------------------------------------------------- axes helpers (symlog x: linear 0..1, log above)
const LINSCALE = 0.55;
function makeAxes(x0, y0, w, h, xlim, ylim) {
  // xlim = [vmin, vmax] in V units (vmin may be slightly negative), ylim in regret units
  const u = v => (v <= 1 ? v * LINSCALE : LINSCALE + Math.log10(v));       // symlog coordinate
  const umin = u(xlim[0]), umax = u(xlim[1]);
  const X = v => x0 + (u(v) - umin) / (umax - umin) * w;
  const Y = r => y0 + h - (r - ylim[0]) / (ylim[1] - ylim[0]) * h;
  return { x0, y0, w, h, X, Y, xlim, ylim };
}
function drawFrame(A, xticks, yticks, opt = {}) {
  line(A.x0, A.y0 + A.h, A.x0 + A.w, A.y0 + A.h, AXIS, 0.75);
  line(A.x0, A.y0, A.x0, A.y0 + A.h, AXIS, 0.75);
  const fs = opt.fontSize || 8;
  xticks.forEach(v => {
    const x = A.X(v);
    line(x, A.y0 + A.h, x, A.y0 + A.h + 0.05, AXIS, 0.75);
    if (opt.xlabels !== false) text(String(v), x - 0.25, A.y0 + A.h + 0.06, 0.5, 0.15, { align: "center", fontSize: fs, color: MUTED });
  });
  yticks.forEach(r => {
    const y = A.Y(r);
    line(A.x0 - 0.05, y, A.x0, y, AXIS, 0.75);
    if (opt.ylabels !== false) text(String(r), A.x0 - 0.5, y - 0.08, 0.43, 0.16, { align: "right", fontSize: fs, color: MUTED });
  });
  if (opt.xlabel) text(opt.xlabel, A.x0, A.y0 + A.h + 0.24, A.w, 0.2, { align: "center", fontSize: 9.5 });
  if (opt.ylabel) s.addText(opt.ylabel, { x: A.x0 - 0.62 - A.h / 2 + 0.1, y: A.y0 + A.h / 2 - 0.1, w: A.h, h: 0.2, rotate: 270,
    fontFace: FONT, fontSize: 9.5, color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
}
function drawPanel(A, P, opt = {}) {
  const size = opt.size || 0.14;
  // frontier first (behind)
  const fr = P.frontier;
  for (let i = 0; i + 1 < fr.length; i++)
    line(A.X(fr[i].V), A.Y(fr[i].r), A.X(fr[i + 1].V), A.Y(fr[i + 1].r), "8C8C8C", 1.0, "dash");
  const order = ["Random", "BO", "GA", "TuRBO", "DALIv2", "SCBO", "LEonly", "DALIv2LE", "cTuRBOknown"];
  for (const m of order) {
    const d = P.methods[m]; if (!d) continue;
    if (opt.clip && (d.V > A.xlim[1] || d.r > A.ylim[1])) continue;
    const col = classOf(m).color;
    if (opt.ci !== false) {
      line(A.X(Math.max(d.V_lo, A.xlim[0])), A.Y(d.r), A.X(Math.min(d.V_hi, A.xlim[1])), A.Y(d.r), col, 1.0);
      line(A.X(d.V), A.Y(Math.max(d.r_lo, A.ylim[0])), A.X(d.V), A.Y(Math.min(d.r_hi, A.ylim[1])), col, 1.0);
    }
    marker(m, A.X(d.V), A.Y(d.r), size);
  }
}

// ---------------------------------------------------------------- (a)
panelTitle("a", "All 16 configurations", 0.30, 0.18, 5.5);
const A = makeAxes(0.95, 0.75, 4.75, 4.05, [-0.3, 140], [-0.004, 0.09]);
drawFrame(A, [0, 1, 3, 10, 30, 100], [0, 0.02, 0.04, 0.06, 0.08],
  { xlabel: "Violations in 100 experiments", ylabel: "Feasible regret r", fontSize: 8.5 });
const Pa = D.panels.all16;
drawPanel(A, Pa, { size: 0.17 });
// direct labels (offsets in inches)
const OFF = { DALIv2LE: [0.14, -0.16, "left"], DALIv2: [0.16, 0, "left"], LEonly: [-0.05, -0.22, "center"], SCBO: [0.02, 0.2, "center"],
  TuRBO: [0.14, -0.05, "left"], GA: [-0.14, -0.12, "right"], BO: [0.14, 0, "left"], Random: [-0.16, -0.12, "right"], cTuRBOknown: [0.18, 0.2, "left"] };
for (const m of Object.keys(OFF)) {
  const d = Pa.methods[m], [dx, dy, al] = OFF[m];
  const x = A.X(d.V) + dx, y = A.Y(d.r) + dy;
  text(LABEL[m], al === "right" ? x - 1.2 : al === "center" ? x - 0.6 : x, y - 0.09, 1.2, 0.18, { align: al, fontSize: 9 });
}
// ---------------------------------------------------------------- (b) 2 x 2 by condition
panelTitle("b", "By condition", 6.45, 0.18, 5.5);
const cells = [["int50", "Interior · 50 %", 0, 0], ["int20", "Interior · 20 %", 0, 1], ["int05", "Interior · 5 %", 1, 0], ["bnd20", "Near boundary · 20 %", 1, 1]];
const bx0 = 7.05, bw = 2.35, bgap = 0.45, by0 = 0.75, bh = 1.65, bvgap = 0.75;
for (const [key, title, r, c] of cells) {
  const x0 = bx0 + c * (bw + bgap), y0 = by0 + r * (bh + bvgap);
  const B = makeAxes(x0, y0, bw, bh, [-0.3, 140], [-0.004, 0.09]);
  text(title, x0, y0 - 0.24, bw, 0.2, { fontSize: 9, bold: true });
  drawFrame(B, [0, 1, 3, 10, 30, 100], [0, 0.04, 0.08], {
    xlabels: r === 1, ylabels: c === 0, fontSize: 8,
    xlabel: r === 1 ? "Violations in 100 experiments" : null, ylabel: c === 0 ? "Feasible regret r" : null });
  drawPanel(B, D.panels[key], { size: 0.12 });
}

// ---------------------------------------------------------------- legend inside (a), upper-left empty region
// (final, 2026-09-28: no class names -- the class colours are introduced in Fig 1 / Fig 2;
//  a thin dotted line separates the classes)
const lgX = 1.12, lgY = 1.0, rowH = 0.215, gapC = 0.10;
const nRows = CLASSES.reduce((n, c) => n + c.methods.length, 0) + 1;
const lgH = nRows * rowH + (CLASSES.length) * gapC + 0.16, lgW = 1.55;
s.addShape(pres.ShapeType.rect, { x: lgX, y: lgY, w: lgW, h: lgH, fill: { color: "FFFFFF" }, line: { color: "D9D9D9", width: 0.5 } });
let ly = lgY + 0.08;
for (const c of CLASSES) {
  for (const m of c.methods) {
    marker(m, lgX + 0.22, ly + rowH / 2, 0.13);
    text(LABEL[m], lgX + 0.42, ly, lgW - 0.45, rowH, { fontSize: 8.5 });
    ly += rowH;
  }
  line(lgX + 0.08, ly + gapC / 2, lgX + lgW - 0.08, ly + gapC / 2, "BFBFBF", 0.5, "sysDot");
  ly += gapC;
}
line(lgX + 0.1, ly + rowH / 2, lgX + 0.36, ly + rowH / 2, "8C8C8C", 1.0, "dash");
text("non-dominated\nfrontier", lgX + 0.42, ly - 0.02, lgW - 0.45, rowH + 0.06, { fontSize: 8, color: MUTED });
text("Points: mean over configurations; bars: 95 % bootstrap CI over seeds (B = 5,000). Horizontal axis linear below 1, logarithmic above.",
  0.95, 5.32, 10.9, 0.18, { fontSize: 7.5, color: MUTED });

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
