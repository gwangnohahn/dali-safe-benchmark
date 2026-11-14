"""Fig 3 -- performance-violation frontier (H2).

(a) 16-configuration mean per method: x = cumulative violations V (symlog axis:
    linear 0-1, log above), y = feasible regret r; bars = 95 % bootstrap CI over seeds
    (resample seeds, average over configurations); non-dominated frontier.
(b) the same for the four conditions of the suite (interior 50 / 20 / 5 %, boundary 20 %;
    4 geometries each) = the column groups of Fig 2(b).
The per-configuration safety cost (former panel c) is written as the SI
figure FigS9_safety_cost.

Data: tier1/summary.json (per-seed V and regret, safety_cost).
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import figstyle as FS

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
MARK = {"DALIv2LE": "o", "DALIv2": "o", "LEonly": "s", "SCBO": "D", "TuRBO": "^", "GA": "v",
        "BO": "P", "Random": "X", "cTuRBOknown": "*"}
B, SEED = 5000, 0


def seed_matrix(S, probs, m, key):
    """(n_seeds, n_probs) matrix of per-seed values (seeds aligned by order)."""
    cols = [S["per_problem"][p]["methods"][m][key] for p in probs]
    n = min(len(c) for c in cols)
    return np.array([c[:n] for c in cols]).T


def boot_mean(Vm, Rm, rng):
    n = Vm.shape[0]
    idx = rng.randint(0, n, size=(B, n))
    v = Vm[idx].mean(axis=(1, 2))
    r = Rm[idx].mean(axis=(1, 2))
    return v, r


def ellipse(ax, x, y, color):
    if x.std() == 0 and y.std() == 0:
        return
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    w, h = 2 * np.sqrt(np.maximum(vals, 0) * 5.991)   # 95 % chi2(2)
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, facecolor=color, alpha=0.18,
                         edgecolor=color, lw=1.0))


def xplot(v):
    """identity: the x axis is symlog (linear 0-1, log above), so V = 0 is drawn at 0"""
    return np.asarray(v, float)


def frontier(points):
    """non-dominated set (min V, min r) sorted by V"""
    pts = sorted(points, key=lambda t: (t[0], t[1]))
    out, best_r = [], np.inf
    for v, r, m in pts:
        if r < best_r:
            out.append((v, r, m))
            best_r = r
    return out


LABEL_OFFSET = {  # (dx, dy) in points, per method, for the annotated panel (a)
    "DALIv2LE": (14, 10), "DALIv2": (16, 0), "LEonly": (0, 18), "SCBO": (0, -26), "TuRBO": (16, 4),
    "GA": (-18, 8), "BO": (16, 0), "Random": (-20, 10), "cTuRBOknown": (18, -18)}
LABEL_HA = {"GA": "right", "Random": "right", "LEonly": "center", "SCBO": "center"}
ZORDER = {"DALIv2LE": 9, "LEonly": 8, "SCBO": 7, "cTuRBOknown": 6}   # draw order where points coincide
LABEL_FS = 17


def draw_point(ax, m, v, r, size=1.0):
    """Marker for method m.  DALI = hollow blue; cTuRBO (known) = hollow black star so
    that points underneath it (V = 0 cluster) stay visible."""
    hollow = m in ("DALIv2", "cTuRBOknown")
    s_ = (170 if m != "cTuRBOknown" else 420) * size
    ax.scatter(xplot(np.array([v])), [r], marker=MARK[m], s=s_,
               facecolors="white" if hollow else FS.COLORS[m],
               edgecolors=FS.COLORS[m] if hollow else "black",
               linewidths=2.6 if hollow else 0.8, zorder=ZORDER.get(m, 5), label=FS.LABELS[m])


def scatter_panel(ax, S, probs, methods, with_ci=True, labels=True, size=1.0, frontier_label=False):
    rng = np.random.RandomState(SEED)
    pts = []
    for m in methods:
        Vm = seed_matrix(S, probs, m, "V_per_seed")
        Rm = seed_matrix(S, probs, m, "regret_per_seed")
        v0, r0 = Vm.mean(), Rm.mean()
        pts.append((v0, r0, m))
        if with_ci:
            bv, br = boot_mean(Vm, Rm, rng)
            vlo, vhi = np.percentile(bv, [2.5, 97.5])
            rlo, rhi = np.percentile(br, [2.5, 97.5])
            ax.plot(xplot(np.array([vlo, vhi])), [r0, r0], color=FS.COLORS[m], lw=1.5, alpha=0.85, zorder=3)
            ax.plot(xplot(np.array([v0, v0])), [rlo, rhi], color=FS.COLORS[m], lw=1.5, alpha=0.85, zorder=3)
        draw_point(ax, m, v0, r0, size)
        if labels:
            dx, dy = LABEL_OFFSET.get(m, (8, 6))
            ax.annotate(FS.LABELS[m], (xplot(np.array([v0]))[0], r0), textcoords="offset points", xytext=(dx, dy),
                        fontsize=LABEL_FS, ha=LABEL_HA.get(m, "left"), va="center", zorder=12)
    fr = frontier(pts)
    ax.plot(xplot(np.array([p[0] for p in fr])), [p[1] for p in fr], color="0.45", lw=1.4, ls="--", zorder=2)
    if frontier_label:
        vx, ry, _ = fr[-1]
        ax.annotate("non-dominated", (xplot(np.array([vx]))[0], ry), textcoords="offset points",
                    xytext=(26, -30), fontsize=LABEL_FS - 2, color="0.4", style="italic", ha="left")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.55)
    ax.set_xlim(-0.3, 140)
    ax.set_xticks([0, 1, 3, 10, 30, 100])
    ax.set_xticklabels(["0", "1", "3", "10", "30", "100"])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(width=1.1, length=6)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))


def safety_cost_panel(ax, S, probs):
    names = []
    costs = {"DALIv2LE": [], "SCBO": []}
    for p in probs:
        sc = S["per_problem"][p]["safety_cost"]
        names.append(p)
        for m in costs:
            c = sc.get(m, {}).get("cost_per_avoided_violation")
            costs[m].append(np.nan if c is None else c)
    x = np.arange(len(probs))
    w = 0.38
    for k, m in enumerate(("DALIv2LE", "SCBO")):
        vals = np.array(costs[m], float)
        ax.barh(x + (k - 0.5) * w, np.nan_to_num(vals, nan=0.0), height=w, color=FS.COLORS[m],
                edgecolor="black", lw=0.5, label=FS.LABELS[m])
        for i, v in enumerate(vals):
            if np.isnan(v):
                ax.text(0, x[i] + (k - 0.5) * w, " n/a", va="center", fontsize=10, color="0.3")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([p.replace("_", " ") for p in probs], fontsize=12)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("regret cost per avoided violation (symlog)\n(r − r$_{best}$)/(V$_{best}$ − V), best unaware baseline as reference")
    ax.set_title("safety cost per configuration", fontsize=14)
    ax.legend(fontsize=12, loc="lower right", frameon=True, edgecolor="0.7")
    ax.grid(axis="x", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier1", "summary.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig3_frontier")
    args = ap.parse_args()
    S = json.load(open(args.summary))
    probs = list(S["per_problem"])
    methods = [m for m in METHODS if all(m in S["per_problem"][p]["methods"] for p in probs)]
    FS.main_rc()

    fig = plt.figure(figsize=(21.0, 10.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.0, 1.0], hspace=0.45, wspace=0.30,
                          left=0.06, right=0.985, top=0.93, bottom=0.16)
    ax_a = fig.add_subplot(gs[:, 0])
    scatter_panel(ax_a, S, probs, methods, size=1.4)
    ax_a.set_ylim(-0.004, 0.09)
    ax_a.set_xlabel("Violations in 100 experiments")
    ax_a.set_ylabel("Feasible regret, r")
    FS.subtitle(ax_a, "all16")
    FS.panel_label(ax_a, "a", dx=-0.13, dy=1.02)
    # (b) the four conditions of the suite = the four column groups of Fig 2(b)
    cells = [("int50", lambda p: "_f050_int" in p, (0, 1)), ("int20", lambda p: "_f020_int" in p, (0, 2)),
             ("int05", lambda p: "_f005_int" in p, (1, 1)), ("bnd20", lambda p: "_f020_bnd" in p, (1, 2))]
    for key, sel, (r, c) in cells:
        ax = fig.add_subplot(gs[r, c])
        sub = [p for p in probs if sel(p)]
        scatter_panel(ax, S, sub, methods, labels=False, size=0.85)
        ax.set_ylim(-0.004, 0.09)
        ax.set_yticks([0, 0.04, 0.08])
        FS.subtitle(ax, key)
        if r == 1:
            ax.set_xlabel("Violations in 100 experiments")
        else:
            ax.tick_params(labelbottom=False)
        if c == 1:
            ax.set_ylabel("Feasible regret, r")
        else:
            ax.tick_params(labelleft=False)
        if (r, c) == (0, 1):
            FS.panel_label(ax, "b", dx=-0.30, dy=1.04)
    handles = [plt.Line2D([], [], marker=MARK[m], ls="", ms=15 if m == "cTuRBOknown" else 12,
                          markerfacecolor="white" if m in ("DALIv2", "cTuRBOknown") else FS.COLORS[m],
                          markeredgecolor=FS.COLORS[m] if m in ("DALIv2", "cTuRBOknown") else "black",
                          markeredgewidth=2.4 if m in ("DALIv2", "cTuRBOknown") else 0.8, color=FS.COLORS[m],
                          label=FS.LABELS[m]) for m in methods]
    handles.append(plt.Line2D([], [], color="0.45", ls="--", lw=1.6, label="non-dominated frontier"))
    fig.legend(handles=handles, loc="lower center", ncol=10, bbox_to_anchor=(0.5, 0.015), frameon=False,
               fontsize=17, columnspacing=1.2, handletextpad=0.5)
    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)

    # SI: safety cost per configuration (former panel c)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    safety_cost_panel(ax, S, probs)
    FS.savefig(fig, args.figdir, "FigS9_safety_cost", dpi=args.dpi)


if __name__ == "__main__":
    main()
