"""Fig. 1(d): the three real datasets as the benchmark sees them (2026-10-19, restyled to match the author's Fig. 1).

For each dataset the condition box (min..max of every condition column) is sampled uniformly (200,000
points), every sample is judged by the kNN-density oracle used in the runs (feasibility_oracle_real.py:
standardised coordinates, mean kNN_5 distance to the de-duplicated data rows, threshold = 99 % quantile of
the rows' leave-one-out kNN_5 distances; the thresholds are checked against realdata_params.json), and
everything is projected onto the first two principal components of the standardised data.  The projected
box is drawn in grey (as the G3 panel of Fig. 1b), the part of it where more than half of the samples are
feasible in light green with a green outline, the data rows as small grey dots and the warm start of seed 0
(200 rows, rng.choice as in the runs) as dark dots with the best row as a star.
Output: figures/Fig1/Fig1d_realdata_panel.{png,pdf,svg} + _data.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle_v2 as V2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/mnt/user-data/uploads/DALI_SAFE/code/dataset"
OUT = os.path.join(ROOT, "figures", "Fig1")
DATASETS = {   # from the paper-1 registry (run_so_experiments.py)
    "steel_fatigue": {"file": "steel_fatigue.xlsx", "cond_idx": [16, 17, 18, 21, 22], "short": "SFS", "title": "SFS", "sub": "Steel fatigue, 5-D"},
    "methane_conv": {"file": "methane_conv.xlsx", "cond_idx": [0, 1, 3, 4], "short": "NOMC", "title": "NOMC", "sub": "Methane conversion, 4-D"},
    "power_plant": {"file": "power_plant.xlsx", "cond_idx": [0, 1, 2, 3], "short": "CCPP", "title": "CCPP", "sub": "Power plant, 4-D"},
}
PARAMS = json.load(open(os.path.join(ROOT, "code", "realdata_params.json")))
K, Q, N_WARM, N_MC, NG = 5, 0.99, 200, 200_000, 90
FEAS_FILL, FEAS_EDGE, BOX_FILL, FRAME = "#DDEFDC", "#2E8B57", "#E6E6E6", "#808080"
ROW_DOT, REC_DOT = "#B8B8B8", "#6E6E6E"


def oracle(X):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    R = np.unique(((X - mu) / sd).round(10), axis=0)
    nn = NearestNeighbors(n_neighbors=K).fit(R)
    loo = NearestNeighbors(n_neighbors=K + 1).fit(R)
    d, _ = loo.kneighbors(R)
    tau = float(np.quantile(d[:, 1:].mean(axis=1), Q))

    def g(Z):
        dd, _ = nn.kneighbors((np.atleast_2d(Z) - mu) / sd)
        return tau - dd.mean(axis=1)
    return g, tau, len(R), mu, sd


def draw_panel(ax, name, cfg, labels=True):
    """Draw one dataset panel on `ax`; returns the info dict (used by main() and by make_fig1_pptx.py)."""
    df = pd.read_excel(os.path.join(DATA, cfg["file"]))
    X = df.iloc[:, cfg["cond_idx"]].values.astype(float)
    g, tau, n_ref, mu, sd = oracle(X)
    key = f"real_{cfg['short'].lower()}_q99"
    assert abs(tau - PARAMS[key]["tau"]) < 1e-6 and n_ref == PARAMS[key]["n_ref"], (name, tau, PARAMS[key]["tau"])
    lb, ub = X.min(axis=0), X.max(axis=0)
    rng = np.random.RandomState(0)
    Umc = lb + rng.rand(N_MC, X.shape[1]) * (ub - lb)
    feas = g(Umc) >= 0
    Z = (X - mu) / sd
    zc = Z.mean(axis=0)
    _, sv, Vt = np.linalg.svd(Z - zc, full_matrices=False)
    P = Vt[:2].T
    proj = lambda A: ((A - mu) / sd - zc) @ P
    pu, px = proj(Umc), proj(X)
    # grid in PCA space: box coverage and feasible fraction
    x0, x1 = pu[:, 0].min(), pu[:, 0].max(); y0, y1 = pu[:, 1].min(), pu[:, 1].max()
    padx, pady = 0.04 * (x1 - x0), 0.04 * (y1 - y0)
    xs = np.linspace(x0 - padx, x1 + padx, NG + 1); ys = np.linspace(y0 - pady, y1 + pady, NG + 1)
    H_all, _, _ = np.histogram2d(pu[:, 0], pu[:, 1], bins=[xs, ys])
    H_f, _, _ = np.histogram2d(pu[feas, 0], pu[feas, 1], bins=[xs, ys])
    cover = gaussian_filter((H_all > 0).astype(float), 0.8)
    # shadow (projection) of the feasible set: cells that contain feasible samples; smoothed, the
    # outline is drawn where the feasible-sample density exceeds 1.5 samples per cell
    dens = gaussian_filter(H_f, 1.0)
    Xc, Yc = np.meshgrid(0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1]), indexing="ij")
    ax.contourf(Xc, Yc, cover, levels=[0.5, 2], colors=[BOX_FILL], zorder=1)
    ax.contourf(Xc, Yc, dens, levels=[1.5, dens.max() + 1], colors=[FEAS_FILL], zorder=2)
    ax.contour(Xc, Yc, dens, levels=[1.5], colors=[FEAS_EDGE], linewidths=0.9, zorder=3)
    warm = np.random.RandomState(0).choice(len(X), N_WARM, replace=False)
    ax.scatter(px[:, 0], px[:, 1], s=1.6 if len(X) < 2000 else 0.35, c=ROW_DOT, linewidths=0, zorder=4, rasterized=len(X) > 2000)
    ax.scatter(px[warm, 0], px[warm, 1], s=3.2, c=REC_DOT, linewidths=0, zorder=5)
    # best row of the warm start (the record's best, as the star in Fig. 1c): the runs minimise
    # f = -objective, with objective = direction-weighted target (paper-1 registry)
    if name == "steel_fatigue":
        ib = warm[np.argmin(df.iloc[warm, 25].values)]     # direction -1 on the last target
    elif name == "methane_conv":
        ib = warm[np.argmax(df.iloc[warm, 12].values)]
    else:
        ib = warm[np.argmax(df.iloc[warm, 4].values)]
    ax.scatter([px[ib, 0]], [px[ib, 1]], marker="*", s=42, c="#111111", linewidths=0, zorder=6)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(FRAME); sp.set_linewidth(0.8)
    pf = PARAMS[key]["box_feasible_frac"]
    if labels:   # labels=False: plot only (titles, axis labels and the feasible-fraction text are added natively in the pptx)
        ax.set_xlabel("PC1", fontsize=7.5, labelpad=2); ax.set_ylabel("PC2", fontsize=7.5, labelpad=2)
        ax.text(0.5, 1.16, cfg["title"], transform=ax.transAxes, fontsize=9, fontweight="bold", ha="center", va="center")
        ax.text(0.5, 1.065, cfg["sub"], transform=ax.transAxes, fontsize=7.5, color=V2.MUTED, ha="center", va="center")
        ax.text(0.03, 0.04, f"feasible {100 * pf:.0f} % of box", transform=ax.transAxes, fontsize=6.6, color=V2.MUTED, ha="left", va="bottom", zorder=7)
    return {"n_rows": int(len(X)), "n_ref_unique": n_ref, "tau_q99": tau, "box_feasible_frac_mc": float(feas.mean()),
                "box_feasible_frac_params": pf, "dim": int(X.shape[1]),
                "pc_explained": [float(v) for v in (sv ** 2 / (sv ** 2).sum())[:2]]}


def main():
    os.makedirs(OUT, exist_ok=True)
    V2.rc(8.0)
    fig = plt.figure(figsize=(7.0, 2.95))
    axes = [fig.add_axes([x, 0.27, 0.255, 0.50]) for x in (0.075, 0.395, 0.715)]
    info = {}
    for ax, (name, cfg) in zip(axes, DATASETS.items()):
        info[cfg["short"]] = draw_panel(ax, name, cfg)
    # panel label + title (Fig. 1 convention: bold letter, plain title)
    fig.text(0.028, 0.975, "d", fontsize=13, fontweight="bold", va="center", ha="left")
    fig.text(0.062, 0.975, "Real datasets (PCA projection of the condition box)", fontsize=10, va="center", ha="left")
    # legend box in the style of Fig. 1b/c
    lax = fig.add_axes([0.075, 0.035, 0.895, 0.11]); lax.axis("off"); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    lax.add_patch(Rectangle((0.002, 0.03), 0.996, 0.94, facecolor="white", edgecolor="#BFBFBF", lw=0.8))
    items = [("box", FEAS_FILL, FEAS_EDGE, "Feasible set (density oracle)"), ("box", BOX_FILL, "#BFBFBF", "Infeasible part of box"),
             ("dot", ROW_DOT, None, "Data rows"), ("dot", REC_DOT, None, "Record (200)"), ("star", "#111111", None, "Best in record")]
    x = 0.02
    for kind, fc, ec, lab in items:
        if kind == "box":
            lax.add_patch(Rectangle((x, 0.33), 0.035, 0.34, facecolor=fc, edgecolor=ec, lw=0.8))
            x += 0.045
        elif kind == "dot":
            lax.scatter([x + 0.012], [0.5], s=12, c=fc, linewidths=0); x += 0.03
        else:
            lax.scatter([x + 0.012], [0.5], marker="*", s=40, c=fc, linewidths=0); x += 0.03
        lax.text(x, 0.5, lab, fontsize=7.5, va="center", ha="left")
        x += V2.text_width(lab, 7.5) / 6.27 + 0.025
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"Fig1d_realdata_panel.{ext}"), dpi=400, bbox_inches="tight", facecolor="white")
    json.dump(info, open(os.path.join(OUT, "Fig1d_realdata_panel_data.json"), "w"), indent=1)
    print(json.dumps({k: {"mc": v["box_feasible_frac_mc"], "params": v["box_feasible_frac_params"]} for k, v in info.items()}))


if __name__ == "__main__":
    main()
