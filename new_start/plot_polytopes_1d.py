"""1D polytope / best-response-region visualizations for n=2.

With n=2 targets and R=1 resource, the strategy space is
    P = { (p1, p2) : p1 + p2 = 1, p1, p2 >= 0 }
which we parameterise by p1 in [0, 1].  For each attacker type alpha, the
best-response boundary U_alpha(1, p) = U_alpha(2, p) reduces to a single
threshold p1 = tau_alpha.  The partition of P becomes a sequence of intervals.

Outputs (in presentation/figures/):
  simplex_1d.pdf               bare strategy space
  partition_1d_1.pdf           partition with 1 attacker (2 intervals)
  partition_1d_2.pdf           partition with 2 attackers (3 intervals)
  partition_1d_3.pdf           partition with 3 attackers (4 intervals)
  extreme_points_1d.pdf        endpoints marked
  eset_growth_1d.pdf           E(C) across |C| = 1, 2, 3
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = "presentation/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# New palette -- should match the presentation color scheme
DARKGREEN = "#006857"
CREAM = "#FDF7E7"
ACCENT = "#B08642"
ACCENT_LIGHT = "#E3C689"
REGION_COLORS = ["#CFE3DB", "#EAD6B0", "#C7D9B5", "#EDC9A6", "#D6C6E0"]


def _line_axes(ax, title=""):
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.6, 0.9)
    ax.axis("off")
    ax.set_title(title, fontsize=12, color=DARKGREEN, pad=10)
    # main axis line
    ax.hlines(0, 0, 1, colors="black", lw=2.2, zorder=5)
    # tick marks at 0 and 1 with labels below
    for x, lbl in [(0, r"$p_1=0\;\;(p_2=1)$"), (1, r"$p_1=1\;\;(p_2=0)$")]:
        ax.plot([x, x], [-0.08, 0.08], color="black", lw=2.2, zorder=6)
        ax.text(x, -0.28, lbl, ha="center", va="top", fontsize=10)


def _shade_intervals(ax, thresholds, labels_per_region=None):
    """Shade the intervals between successive thresholds with region colors."""
    bounds = [0.0] + sorted(thresholds) + [1.0]
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        ax.add_patch(plt.Rectangle((a, -0.10), b - a, 0.20,
                                   facecolor=REGION_COLORS[k % len(REGION_COLORS)],
                                   edgecolor="none", zorder=3))
    for tau in thresholds:
        ax.plot([tau, tau], [-0.14, 0.14], color=ACCENT, lw=2.0, ls="--", zorder=6)
        ax.plot(tau, 0, "o", color=ACCENT, markersize=7, zorder=7)
    if labels_per_region:
        for k, lbl in enumerate(labels_per_region):
            a = bounds[k]
            b = bounds[k + 1]
            ax.text((a + b) / 2, 0.30, lbl,
                    ha="center", va="bottom", fontsize=10, color="#333")


def _label_thresholds(ax, thresholds, names):
    for tau, name in zip(thresholds, names):
        ax.text(tau, -0.18, name, ha="center", va="top",
                fontsize=10, color=ACCENT)


def _mark_extreme_points(ax, thresholds):
    pts = sorted(set([0.0] + list(thresholds) + [1.0]))
    for x in pts:
        ax.plot(x, 0, "o", color="#222", markersize=9, zorder=11)
        ax.plot(x, 0, "o", color="#D72631", markersize=5, zorder=12)


# ----------------------------------------------------------------------------
# Attacker thresholds (picked so the regions are visibly distinct)
# ----------------------------------------------------------------------------
TAU_1 = 0.35   # alpha_1
TAU_2 = 0.65   # alpha_2
TAU_3 = 0.50   # alpha_3


def fig_simplex_1d():
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    _line_axes(ax, r"Strategy space $\mathcal{P}=\{(p_1,p_2):p_1+p_2=1,\;p_i\geq 0\}$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "simplex_1d.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "simplex_1d.png"), dpi=160)
    plt.close(fig)


def fig_partition_1d_1():
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    _line_axes(ax, r"Catalogue $C=\{\alpha_1\}$: 2 best-response regions")
    _shade_intervals(ax, [TAU_1],
                     [r"$b_{\alpha_1}=1$", r"$b_{\alpha_1}=2$"])
    _label_thresholds(ax, [TAU_1], [r"$\tau_{\alpha_1}$"])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_1d_1.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_1d_1.png"), dpi=160)
    plt.close(fig)


def fig_partition_1d_2():
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    _line_axes(ax, r"Catalogue $C=\{\alpha_1,\alpha_2\}$: 3 refined regions")
    _shade_intervals(ax, [TAU_1, TAU_2],
                     [r"$(1,1)$", r"$(2,1)$", r"$(2,2)$"])
    _label_thresholds(ax, [TAU_1, TAU_2],
                      [r"$\tau_{\alpha_1}$", r"$\tau_{\alpha_2}$"])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_1d_2.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_1d_2.png"), dpi=160)
    plt.close(fig)


def fig_partition_1d_3():
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    _line_axes(ax, r"Catalogue $C=\{\alpha_1,\alpha_2,\alpha_3\}$: 4 refined regions")
    _shade_intervals(ax, [TAU_1, TAU_3, TAU_2],
                     [r"$(1,1,1)$", r"$(2,1,1)$", r"$(2,1,2)$", r"$(2,2,2)$"])
    _label_thresholds(ax, [TAU_1, TAU_3, TAU_2],
                      [r"$\tau_{\alpha_1}$", r"$\tau_{\alpha_3}$", r"$\tau_{\alpha_2}$"])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_1d_3.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_1d_3.png"), dpi=160)
    plt.close(fig)


def fig_extreme_points_1d():
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    _line_axes(ax, r"Extreme point set $\mathcal{E}(C)$ for $C=\{\alpha_1,\alpha_2\}$")
    _shade_intervals(ax, [TAU_1, TAU_2])
    _label_thresholds(ax, [TAU_1, TAU_2],
                      [r"$\tau_{\alpha_1}$", r"$\tau_{\alpha_2}$"])
    _mark_extreme_points(ax, [TAU_1, TAU_2])
    ax.text(0.5, 0.55,
            r"$\mathcal{E}(C)=\{0,\;\tau_{\alpha_1},\;\tau_{\alpha_2},\;1\}$",
            ha="center", va="bottom", fontsize=11, color=DARKGREEN)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "extreme_points_1d.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "extreme_points_1d.png"), dpi=160)
    plt.close(fig)


def fig_eset_growth_1d():
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    cases = [
        ([TAU_1], r"$|C|=1,\;|\mathcal{E}(C)|=3$"),
        ([TAU_1, TAU_2], r"$|C|=2,\;|\mathcal{E}(C)|=4$"),
        ([TAU_1, TAU_3, TAU_2], r"$|C|=3,\;|\mathcal{E}(C)|=5$"),
    ]
    for ax, (thr, title) in zip(axes, cases):
        _line_axes(ax, title)
        _shade_intervals(ax, thr)
        _mark_extreme_points(ax, thr)
    fig.suptitle(r"$\mathcal{E}(C)$ grows as new attacker types are discovered",
                 fontsize=13, color=DARKGREEN, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eset_growth_1d.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "eset_growth_1d.png"), dpi=160)
    plt.close(fig)


def main():
    fig_simplex_1d()
    fig_partition_1d_1()
    fig_partition_1d_2()
    fig_partition_1d_3()
    fig_extreme_points_1d()
    fig_eset_growth_1d()
    print("Saved 1D polytope figures in", FIG_DIR)


if __name__ == "__main__":
    main()
