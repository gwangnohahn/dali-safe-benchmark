"""Fig 4 -- when do violations happen (H1b).  All three panels use the same runs
as Fig 3: the 16 Tier-1 configurations x 20 seeds (320 runs per method).

(a) Kaplan-Meier "no violation yet" curve vs experiment index, pooled over the
    320 runs; censored at 100; band = Greenwood 95 % CI.  Curves are labelled directly with the fraction
    of runs that never violate (pooled log-rank p vs DALI+LE -> caption / Table S1).
(b) cumulative violations vs experiment index, mean over the 16 configurations
    (band = 95 % CI of that mean from the per-configuration seed SDs).
(c) violations within the first 20 experiments (4 cycles): bar = mean over the
    16 configurations, dots = the configurations.

Data: tier1/summary.json (T1 per seed, V_curve_mean/sd).
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import figstyle as FS
from analyze_constraint_suite import km_curve

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ["DALIv2LE", "DALIv2", "LEonly", "SCBO", "TuRBO", "GA", "BO", "Random", "cTuRBOknown"]
LABEL_FS = 17


def km_greenwood(times, events, t_max):
    """Kaplan-Meier S(t) with Greenwood 95 % CI (log-log transform), t = 1..t_max."""
    times = np.asarray(times, float)
    events = np.asarray(events, bool)
    S, gw, surv, lo, hi = 1.0, 0.0, [], [], []
    for t in range(1, t_max + 1):
        n = np.sum(times >= t)
        d = np.sum((times == t) & events)
        if n > 0 and d > 0:
            S *= 1.0 - d / n
            gw += d / (n * (n - d)) if n > d else 0.0
        surv.append(S)
        if 0 < S < 1 and gw > 0:
            se = np.sqrt(gw) / abs(np.log(S))
            lo.append(S ** np.exp(1.96 * se))
            hi.append(S ** np.exp(-1.96 * se))
        else:
            lo.append(S)
            hi.append(S)
    return surv, lo, hi


def line_kw(m):
    """figstyle line style without the DALI+LE markers (curves are labelled directly)."""
    kw = FS.style_line(m)
    for k in ("marker", "markevery", "ms", "markerfacecolor", "markeredgecolor", "markeredgewidth"):
        kw.pop(k, None)
    kw["lw"] = {"DALIv2LE": 3.2, "DALIv2": 2.2}.get(m, 2.0)
    kw.pop("label", None)
    return kw


def stack_labels(ys, min_sep):
    """Shift label positions so that consecutive labels (sorted by y) are at least
    min_sep apart; returns positions in the original order."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < min_sep:
            pos[i] = pos[i - 1] + min_sep
    # push back down if the stack overshoots the top of the axis is handled by caller (ylim)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def end_labels(ax, x_end, ys, texts, colors, min_sep, x_text, y_max=None, fs=LABEL_FS):
    """Direct labels at the right end of curves with thin leader lines where a
    label had to be moved."""
    pos = stack_labels(ys, min_sep)
    if y_max is not None and pos.max() > y_max:
        pos -= pos.max() - y_max
    for y, yp, t, c in zip(ys, pos, texts, colors):
        if abs(yp - y) > 1e-9:
            ax.plot([x_end, x_text - 0.4 * (x_text - x_end)], [y, yp], color="0.6", lw=0.8, clip_on=False, zorder=1)
        ax.text(x_text, yp, t, color=c, fontsize=fs, va="center", ha="left", clip_on=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(os.path.dirname(CODE_DIR), "results", "constraint_suite", "tier1", "summary.json"))
    ap.add_argument("--figdir", default=os.path.join(os.path.dirname(CODE_DIR), "figures"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--stem", default="Fig4_survival")
    args = ap.parse_args()
    S = json.load(open(args.summary))
    probs = list(S["per_problem"])
    methods = [m for m in METHODS if all(m in S["per_problem"][p]["methods"] for p in probs)]
    t_max = 100
    FS.main_rc()

    fig = plt.figure(figsize=(26.0, 7.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.45, 1.0], wspace=0.28,
                          left=0.045, right=0.985, top=0.90, bottom=0.14)
    X_MAX = 150          # data axis ends at 100; the rest of the panel holds the direct labels

    clean = FS.clean_axes

    # ------------------------------------------------------------------ (a) pooled KM
    ax = fig.add_subplot(gs[0, 0])
    ends, txts, cols = [], [], []
    for m in methods:
        T, E = [], []
        for p in probs:
            r = S["per_problem"][p]["methods"][m]
            T += r["T1_per_seed"]
            E += r["T1_event_per_seed"]
        s, lo, hi = km_greenwood(np.array(T), np.array(E, bool), t_max)
        assert np.allclose(s, km_curve(np.array(T), np.array(E, bool), t_max))
        ax.step(np.arange(0, t_max + 1), [1.0] + s, where="post", **line_kw(m))
        ax.fill_between(np.arange(0, t_max + 1), [1.0] + lo, [1.0] + hi, step="post",
                        color=FS.COLORS[m], alpha=0.10, lw=0)
        ends.append(s[-1])
        txts.append(f"{FS.LABELS[m]}  {100 * s[-1]:.0f} %")
        cols.append(FS.COLORS[m])
    end_labels(ax, t_max, ends, txts, cols, min_sep=0.062, x_text=t_max + 4, y_max=1.0)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Experiments")
    ax.set_ylabel("Runs with no violation yet")
    FS.subtitle(ax, "all16", pad=10)
    clean(ax, xbound=(0, t_max))
    FS.panel_label(ax, "a", dx=-0.14, dy=1.04)

    # ------------------------------------------------------------------ (b) cumulative violations, all 16
    ax = fig.add_subplot(gs[0, 1])
    cyc = np.arange(1, 21) * 5
    ends, txts, cols = [], [], []
    for m in methods:
        M = np.array([S["per_problem"][p]["methods"][m]["V_curve_mean"] for p in probs])
        SD = np.array([S["per_problem"][p]["methods"][m]["V_curve_sd"] or [0] * 20 for p in probs])
        n = np.array([S["per_problem"][p]["methods"][m]["n_seeds"] for p in probs])
        mu = M.mean(axis=0)
        se = np.sqrt((SD ** 2 / n[:, None]).sum(axis=0)) / len(probs)   # SE of the mean of config means
        ax.plot(np.r_[0, cyc], np.r_[0, mu], **line_kw(m))
        ax.fill_between(cyc, mu - 1.96 * se, mu + 1.96 * se, color=FS.COLORS[m], alpha=0.13, lw=0)
        ends.append(mu[-1])
        txts.append(f"{FS.LABELS[m]}  {mu[-1]:.1f}")
        cols.append(FS.COLORS[m])
    end_labels(ax, t_max, ends, txts, cols, min_sep=5.4, x_text=t_max + 4, y_max=100)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Experiments")
    ax.set_ylabel("Cumulative violations")
    FS.subtitle(ax, "all16", pad=10)
    clean(ax, xbound=(0, t_max))
    FS.panel_label(ax, "b", dx=-0.16, dy=1.04)

    # ------------------------------------------------------------------ (c) first 20 experiments
    ax = fig.add_subplot(gs[0, 2])
    vals = [np.array([S["per_problem"][p]["methods"][m]["V_curve_mean"][3] for p in probs]) for m in methods]
    means = np.array([v.mean() for v in vals])
    y = np.arange(len(methods))
    rng = np.random.RandomState(0)
    for i, m in enumerate(methods):
        hollow = m == "DALIv2"
        ax.barh(i, means[i], height=0.68, color="white" if hollow else FS.COLORS[m],
                edgecolor=FS.COLORS[m] if hollow else "black", lw=1.8 if hollow else 0.6,
                hatch="////" if hollow else None, zorder=2)
        jit = rng.uniform(-0.22, 0.22, size=len(vals[i]))
        ax.scatter(vals[i], i + jit, s=22, color="black", alpha=0.55, zorder=3, lw=0)
        ax.text(max(means[i], vals[i].max()) + 0.5, i, f"{means[i]:.1f}", va="center", ha="left", fontsize=LABEL_FS)
    ax.set_yticks(y)
    ax.set_yticklabels([FS.LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Violations in first 20 experiments")
    FS.subtitle(ax, "All 16 configurations (dots)", pad=10)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    FS.panel_label(ax, "c", dx=-0.55, dy=1.04)

    FS.savefig(fig, args.figdir, args.stem, dpi=args.dpi)


if __name__ == "__main__":
    main()
