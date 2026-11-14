/* Fig 6 v2 (2026-09-28) -- PowerPoint version of Fig 6 (limits).
 * v3 (2026-10-06): identical to v2 except the wording of panel (a) (the feasible region is shifted, not the optimum; author comment 35).
 *
 * Plot areas are images (make_fig6_v2.py); titles, subtitles, axis titles, the numbers at
 * the line ends of (d), the shared legend and the footnote are editable shapes.
 * Layout: row 1  (a) | (b) | (c) | legend ;  row 2  (d) x 4 ;  row 3  (e) | (f).  Width 12.9 in.
 *
 * Usage: node make_fig6_v2_pptx.js [figdir] [outPptx]
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const CODE_DIR = __dirname;
const figdir = process.argv[2] || path.join(CODE_DIR, "..", "figures");
const outPath = process.argv[3] || path.join(figdir, "Fig6_limits_v3.pptx");
const D = JSON.parse(fs.readFileSync(path.join(figdir, "Fig6_v2_data.json"), "utf8"));

const FONT = "Arial", INK = "222222", MUTED = "595959";
const methods = D.methods.map(m => m.key);
const LABEL = Object.fromEntries(D.methods.map(m => [m.key, m.label]));
const COLOR = Object.fromEntries(D.methods.map(m => [m.key, m.color.replace("#", "").toUpperCase()]));
const LSTYLE = Object.fromEntries(D.methods.map(m => [m.key, m.line]));
const DASH = { solid: "solid", dashed: "dash", dotted: "sysDot", "dash-dot": "dashDot" };
const hex = m => (m === "cTuRBOknown" ? "000000" : COLOR[m]);
const GEO_LABEL = { g1: "G1  Convex", g2: "G2  Non-convex", g3: "G3  Disconnected", g4: "G4  Thin band" };

const pres = new pptxgen();
const W = 12.9, H = 11.3;
pres.defineLayout({ name: "FIG6", width: W, height: H });
pres.layout = "FIG6";
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

function text(str, x, y, w, h, o = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: FONT, fontSize: 9, color: INK, margin: 0,
    isTextBox: true, valign: "middle", align: "left" }, o));
}
function rect(x, y, w, h, fill, lineCol, lw) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: fill ? { color: fill } : { type: "none" }, line: { color: lineCol || fill, width: lw === undefined ? 0 : lw } });
}
function line(x1, y1, x2, y2, color, width, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color, width, dashType: dash || "solid" } };
  if ((x2 - x1) * (y2 - y1) < 0) o.flipV = true;
  s.addShape(pres.ShapeType.line, o);
}
function panelTitle(letter, title, x, y, w, sub) {
  s.addText([{ text: letter, options: { bold: true, fontSize: 14 } }, { text: "   " + title, options: { fontSize: 11 } }],
    { x, y, w, h: 0.3, fontFace: FONT, color: INK, margin: 0, isTextBox: true, valign: "middle", align: "left" });
  if (sub) text(sub, x + 0.28, y + 0.3, w, 0.2, { fontSize: 8.5, color: MUTED });
}
function ylabel(str, x, yc, len) {
  s.addText(str, { x: x - len / 2 + 0.1, y: yc - 0.1, w: len, h: 0.2, rotate: 270, fontFace: FONT, fontSize: 9, color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
}
function placeImage(name, key, imgX, imgY, scale = 1.0) {
  const g = D.geom[key], [fw, fh] = g.fig, [ax0, ay0, aw, ah] = g.ax;
  s.addImage({ path: path.join(figdir, name), x: imgX, y: imgY, w: fw * scale, h: fh * scale });
  return { x0: imgX + ax0 * scale, y0: imgY + (fh - ay0 - ah) * scale, w: aw * scale, h: ah * scale, imgW: fw * scale, imgH: fh * scale };
}
function stack(ys, minSep, yMin, yMax) {
  const idx = ys.map((y, i) => i).sort((a, b) => ys[a] - ys[b]);
  const pos = idx.map(i => ys[i]);
  for (let k = 1; k < pos.length; k++) if (pos[k] - pos[k - 1] < minSep) pos[k] = pos[k - 1] + minSep;
  if (yMax !== undefined && pos[pos.length - 1] > yMax) { pos[pos.length - 1] = yMax; for (let k = pos.length - 2; k >= 0; k--) pos[k] = Math.min(pos[k], pos[k + 1] - minSep); }
  if (yMin !== undefined && pos[0] < yMin) { const d = yMin - pos[0]; for (let k = 0; k < pos.length; k++) pos[k] += d; }
  const out = new Array(ys.length); idx.forEach((i, k) => out[i] = pos[k]); return out;
}

// ================================================================ row 1
const r1Top = 0.15, r1Img = 0.58;
panelTitle("a", "Feasible region shifted: optimum near the boundary", 0.30, r1Top, 4.5);
const A = placeImage("Fig6_v2_panel_a.png", "a", 0.35, r1Img);
ylabel("Violations in 100 experiments", A.x0 - 0.42, A.y0 + A.h / 2, 2.2);

panelTitle("b", "Contaminated record", 4.85, r1Top, 3.0);
const B = placeImage("Fig6_v2_panel_b.png", "b", 4.85, r1Img);
ylabel("Violations in 100 experiments", B.x0 - 0.42, B.y0 + B.h / 2, 2.2);
text("Contamination (%)", B.x0, B.y0 + B.h + 0.22, B.w, 0.2, { align: "center", fontSize: 9 });

panelTitle("c", "Record 200 → 50 points", 7.75, r1Top, 2.5);
const C = placeImage("Fig6_v2_panel_c.png", "c", 7.75, r1Img);
ylabel("Violations in 100 experiments", C.x0 - 0.42, C.y0 + C.h / 2, 2.2);

// shared legend (right of row 1): marker + line sample + name, grouped by class; encoding key
const lgX = 10.3, lgW = 2.3, lgY = r1Img + 0.05, lrow = 0.2, lsep = 0.08;
const LG = ["DALIv2LE", "DALIv2", "LEonly", null, "SCBO", null, "TuRBO", "GA", "BO", "Random", null, "cTuRBOknown", null, "__key__"];
const lgH = 0.14 + 10 * lrow + 4 * lsep + 0.34;
rect(lgX, lgY, lgW, lgH, "FFFFFF", "D9D9D9", 0.5);
const MK = { DALIv2LE: "ellipse", DALIv2: "ellipse", LEonly: "rect", SCBO: "diamond", TuRBO: "triangle", GA: "triangleDown", BO: "mathPlus", Random: "mathMultiply", cTuRBOknown: "star5" };
const HOLLOW = new Set(["DALIv2", "cTuRBOknown"]);
function marker(m, cx, cy, size, fillCol) {
  const col = hex(m), sz = size * (m === "cTuRBOknown" ? 1.5 : 1);
  const o = { x: cx - sz / 2, y: cy - sz / 2, w: sz, h: sz, fill: { color: HOLLOW.has(m) ? "FFFFFF" : (fillCol || col) },
    line: { color: HOLLOW.has(m) ? (fillCol || col) : INK, width: HOLLOW.has(m) ? 1.5 : 0.5 } };
  let shape = MK[m]; if (shape === "triangleDown") { shape = "triangle"; o.flipV = true; }
  s.addShape(pres.ShapeType[shape], o);
}
{
  let y = lgY + 0.08; const x = lgX + 0.12;
  for (const m of LG) {
    if (m === null) { line(x, y + lsep / 2, x + lgW - 0.24, y + lsep / 2, "BFBFBF", 0.5, "sysDot"); y += lsep; continue; }
    if (m === "__key__") {   // dumbbell key: light -> full colour
      const cy = y + 0.17;
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.02, y: cy - 0.06, w: 0.12, h: 0.12, fill: { color: "A9C4E3" }, line: { color: "A9C4E3", width: 0.5 } });
      line(x + 0.16, cy, x + 0.42, cy, "2E75B6", 1.25);
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.44, y: cy - 0.06, w: 0.12, h: 0.12, fill: { color: "2E75B6" }, line: { color: INK, width: 0.5 } });
      text("light → full colour:\nreference → changed condition", x + 0.64, y, lgW - 0.8, 0.36, { fontSize: 7.5, color: MUTED });
      y += 0.36; continue;
    }
    marker(m, x + 0.1, y + lrow / 2, 0.13);
    line(x + 0.24, y + lrow / 2, x + 0.52, y + lrow / 2, hex(m), m === "DALIv2LE" ? 1.75 : 1.25, DASH[LSTYLE[m]]);
    text(LABEL[m], x + 0.6, y, lgW - 0.75, lrow, { fontSize: 8.5 });
    y += lrow;
  }
}

// ================================================================ row 2: (d)
const r2Top = A.y0 + A.h + 0.55, r2Img = r2Top + 0.68;
panelTitle("d", "Near-boundary optimum, feasible fraction 50 / 20 / 5 %", 0.30, r2Top, 8.0);
const dImgW = D.geom.d.fig[0], dSlot = 3.06;
["g1", "g2", "g3", "g4"].forEach((g, k) => {
  const x = 0.35 + k * dSlot;
  const P = placeImage(`Fig6_v2_panel_d_${g}.png`, "d", x, r2Img);
  text(GEO_LABEL[g], P.x0, r2Img - 0.28, P.w, 0.22, { align: "center", bold: true, fontSize: 9 });
  line(P.x0 + 0.02, r2Img - 0.04, P.x0 + P.w - 0.02, r2Img - 0.04, INK, 1.0);
  text("Feasible fraction of the box (%)", P.x0 - 0.2, P.y0 + P.h + 0.22, P.w + 0.4, 0.2, { align: "center", fontSize: 8.5 });
  if (k === 0) ylabel("Violations in 100 experiments", P.x0 - 0.4, P.y0 + P.h / 2, 2.0);
  // numbers at the right end (value at 5 %), stacked, class colour
  const [yl, yh] = D.d.ylim, [xl, xh] = D.d.xlim;
  const Y = v => P.y0 + P.h - (v - yl) / (yh - yl) * P.h;
  const xEnd = P.x0 + (2 - xl) / (xh - xl) * P.w;
  const ys = methods.map(m => Y(D.d[g][m][2]));
  const pos = stack(ys, 0.135, P.y0 - 0.02, P.y0 + P.h + 0.02);
  methods.forEach((m, i) => {
    if (Math.abs(pos[i] - ys[i]) > 1e-3) line(xEnd + 0.02, ys[i], xEnd + 0.12, pos[i], "A6A6A6", 0.5);
    text(D.d[g][m][2].toFixed(1), xEnd + 0.14, pos[i] - 0.07, 0.45, 0.14, { fontSize: 7.5, color: hex(m) });
  });
});

// ================================================================ row 3: (e) (f)
const r3Top = r2Img + D.geom.d.fig[1] + 0.45, r3Img = r3Top + 0.42;
panelTitle("e", "Uniform record: violations (exploratory)", 0.30, r3Top, 6.0);
const E = placeImage("Fig6_v2_panel_e.png", "e", 0.35, r3Img);
ylabel("Violations in 100 experiments", E.x0 - 0.42, E.y0 + E.h / 2, 2.2);
panelTitle("f", "Uniform record: regret (exploratory)", 6.65, r3Top, 6.0);
const F = placeImage("Fig6_v2_panel_f.png", "e", 6.65, r3Img);
ylabel("Feasible regret r", F.x0 - 0.42, F.y0 + F.h / 2, 2.2);

// ================================================================ footnote
text("All panels: interior 20 % configurations unless stated; violations = mean over seeds. (a, c, e, f) Dumbbells: light marker = reference condition → full-colour marker with arrow head = changed condition; DALI has a dashed connector. " +
  "(a) Feasible region shifted so that the optimum moves from the interior to just inside the boundary. (b) 0 / 10 / 30 % of the warm start replaced by infeasible points; line = mean of the four geometries, grey dots = geometries. (c) Warm start 200 → 50 points; the regret change is given in Results 2.6. " +
  "(d) Near-boundary optimum at three feasible fractions; numbers at the line ends = violations at 5 %. (e, f) Record clustered around the optimum → record uniform over the feasible set (exploratory block); (f) uses the same runs as (e).",
  0.35, E.y0 + E.h + 0.5, W - 0.7, 0.6, { fontSize: 7.5, color: MUTED, valign: "top" });

pres.writeFile({ fileName: outPath }).then(f => console.log("wrote", f));
