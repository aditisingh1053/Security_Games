"""Polytope / best-response-region visualizations for the presentation.

We work in the n=3 simplex projected to 2D:
    P = { (p1, p2) : p1 >= 0, p2 >= 0, p1 + p2 <= 1 },    p3 = 1 - p1 - p2

For each attacker type alpha with utilities (u^c, u^u), the boundary
between "attack target i" and "attack target j" in coverage space is the
line  U_alpha(i, p) = U_alpha(j, p)  which becomes linear in (p1, p2).
We draw these lines, shade the regions by which target is attacked, and
mark the extreme points of the induced partition.

Outputs (all in presentation/figures/):
  simplex.pdf                  bare simplex with target labels
  partition_1.pdf              partition for catalogue {alpha_1}
  partition_2.pdf              partition for catalogue {alpha_1, alpha_2}
  partition_3.pdf              partition for catalogue {alpha_1, alpha_2, alpha_3}
  extreme_points.pdf           extreme points overlaid on 2-attacker partition
  eset_growth.pdf              E-set size as new types are added
  sequential_vs_simultaneous.pdf  conceptual diagram
"""

from __future__ import annotations

import os
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

FIG_DIR = "presentation/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Simplex plotting utilities
# ----------------------------------------------------------------------------


SIMPLEX_VERTS = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # p1e1, p2e2, p3e3 in (p1, p2)

# Pleasant target colors
REGION_COLORS = {
    1: "#A8DADC",
    2: "#F6BD60",
    3: "#B5C99A",
}
TARGET_LABEL_COLORS = {
    1: "#1D3557",
    2: "#A0522D",
    3: "#386641",
}


def _draw_simplex(ax, fill=False):
    tri = MplPolygon(SIMPLEX_VERTS, closed=True,
                     facecolor=("white" if fill else "none"),
                     edgecolor="black", lw=1.8, zorder=5)
    ax.add_patch(tri)
    # Corner labels
    ax.text(1.02, -0.04, r"$p=(1,0,0)$", fontsize=10)
    ax.text(-0.03, 1.03, r"$p=(0,1,0)$", fontsize=10, ha="right")
    ax.text(-0.03, -0.04, r"$p=(0,0,1)$", fontsize=10, ha="right")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")


def _best_response_lines(alpha, i, j, n=3):
    """Return (a_vec, b_scalar) for line a . (p1, p2) = b representing U_alpha(i) = U_alpha(j)
    in the projected 2D coordinates (p1, p2).
    Equation in full n=3 coords: Delta_i p_i - Delta_j p_j = u^u(j) - u^u(i)
    with p3 = 1 - p1 - p2.
    """
    u_c, u_u = alpha
    Delta_i = u_c[i] - u_u[i]
    Delta_j = u_c[j] - u_u[j]
    rhs_p = u_u[j] - u_u[i]
    coeff_p = np.zeros(n)
    coeff_p[i] = Delta_i
    coeff_p[j] = -Delta_j
    # Substitute p3 = 1 - p1 - p2
    a_x = coeff_p[0] - coeff_p[2]
    a_y = coeff_p[1] - coeff_p[2]
    rhs_x = rhs_p - coeff_p[2]
    return np.array([a_x, a_y]), rhs_x


def _best_response_at(alpha, p):
    """Return argmax target (0/1/2) for alpha at p=(p1,p2,p3)."""
    u_c, u_u = alpha
    utils = u_c * p + u_u * (1 - p)
    return int(np.argmax(utils))


def _shade_best_response_regions(ax, alphas, grid_res=600):
    """Shade the n^|alphas| regions by (sum of target indices across attackers)
    using a fine grid. We color each cell by the FIRST attacker's best response
    so that visualizations focus on partition structure.
    """
    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    X, Y = np.meshgrid(xs, ys)
    mask_simplex = (X >= 0) & (Y >= 0) & (X + Y <= 1)
    # Use alpha_1's best-response to color
    alpha_1 = alphas[0]
    u_c, u_u = alpha_1
    p_grid = np.stack([X, Y, 1 - X - Y], axis=-1)  # shape H, W, 3
    utils = u_c[None, None, :] * p_grid + u_u[None, None, :] * (1 - p_grid)
    br = np.argmax(utils, axis=-1)
    br = np.where(mask_simplex, br, -1)
    color_img = np.ones((*br.shape, 4), dtype=float)
    from matplotlib.colors import to_rgba
    for t in range(3):
        rgba = to_rgba(REGION_COLORS[t + 1], alpha=0.60)
        color_img[br == t] = rgba
    color_img[~mask_simplex] = (1, 1, 1, 0)
    ax.imshow(color_img, extent=[0, 1, 0, 1], origin="lower")


def _draw_boundary_lines(ax, alphas, linestyles=None, colors=None):
    """Draw all pairwise best-response boundaries for each attacker in alphas."""
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"]
    if colors is None:
        colors = ["#1D3557", "#E63946", "#457B9D", "#2A9D8F"]
    for k, alpha in enumerate(alphas):
        ls = linestyles[k % len(linestyles)]
        col = colors[k % len(colors)]
        for (i, j) in itertools.combinations(range(3), 2):
            a, b = _best_response_lines(alpha, i, j)
            # Sample points on the line intersected with simplex
            xs = np.linspace(-0.2, 1.2, 500)
            if abs(a[1]) > 1e-8:
                ys = (b - a[0] * xs) / a[1]
            else:
                xs = np.full(500, b / a[0] if abs(a[0]) > 1e-8 else 0)
                ys = np.linspace(-0.2, 1.2, 500)
            valid = (xs >= 0) & (ys >= 0) & (xs + ys <= 1)
            if valid.sum() < 2:
                continue
            ax.plot(xs[valid], ys[valid], ls, color=col, lw=1.6,
                    label=rf"$\alpha_{{{k+1}}}$ boundary" if (i == 0 and j == 1) else None)


# ----------------------------------------------------------------------------
# Extreme point computation (matches algorithm.py; restricted to n=3 for plots)
# ----------------------------------------------------------------------------


def compute_extreme_points_2d(alphas):
    """Enumerate all pairs of hyperplanes (simplex + best-response), solve for
    intersection in 2D, keep those inside the simplex. Returns list of (x, y).
    """
    # Simplex hyperplanes in 2D:
    #   p1 = 0,  p2 = 0,  p1 + p2 = 1
    hps = [
        (np.array([1.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0),
        (np.array([1.0, 1.0]), 1.0),
    ]
    for alpha in alphas:
        for (i, j) in itertools.combinations(range(3), 2):
            a, b = _best_response_lines(alpha, i, j)
            hps.append((a, b))
    pts = []
    for s in itertools.combinations(range(len(hps)), 2):
        A = np.array([hps[s[0]][0], hps[s[1]][0]])
        rhs = np.array([hps[s[0]][1], hps[s[1]][1]])
        if abs(np.linalg.det(A)) < 1e-10:
            continue
        try:
            x = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        if x[0] < -1e-8 or x[1] < -1e-8 or x[0] + x[1] > 1 + 1e-8:
            continue
        pts.append(tuple(np.round(x, 6)))
    return sorted(set(pts))


# ----------------------------------------------------------------------------
# Concrete attacker types used for visualization
# ----------------------------------------------------------------------------

# alpha_1: cares most about target 1  (u_u high at target 1)
ALPHA_1 = (np.array([-0.4, -0.2, -0.6]), np.array([0.9, 0.3, 0.5]))
# alpha_2: cares most about target 2
ALPHA_2 = (np.array([-0.3, -0.5, -0.2]), np.array([0.4, 0.85, 0.5]))
# alpha_3: cares most about target 3
ALPHA_3 = (np.array([-0.2, -0.3, -0.5]), np.array([0.35, 0.45, 0.9]))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def fig_simplex():
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _draw_simplex(ax, fill=True)
    ax.set_title(r"Strategy space $\mathcal{P}$ for $n=3$ targets, $R=1$ resource", fontsize=11)
    # Annotate: each vertex = "cover target X with probability 1"
    ax.annotate("cover target 1", xy=(1, 0), xytext=(1.1, 0.12),
                fontsize=9, color=TARGET_LABEL_COLORS[1],
                arrowprops=dict(arrowstyle="->", color=TARGET_LABEL_COLORS[1]))
    ax.annotate("cover target 2", xy=(0, 1), xytext=(-0.21, 1.08),
                fontsize=9, color=TARGET_LABEL_COLORS[2],
                arrowprops=dict(arrowstyle="->", color=TARGET_LABEL_COLORS[2]))
    ax.annotate("cover target 3", xy=(0, 0), xytext=(-0.24, -0.10),
                fontsize=9, color=TARGET_LABEL_COLORS[3],
                arrowprops=dict(arrowstyle="->", color=TARGET_LABEL_COLORS[3]))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "simplex.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "simplex.png"), dpi=150)
    plt.close(fig)


def fig_partition_1():
    alphas = [ALPHA_1]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _draw_simplex(ax)
    _shade_best_response_regions(ax, alphas)
    _draw_boundary_lines(ax, alphas)
    # Label regions by attacker's best response target
    # region 1: attack target 1, region 2: attack target 2, region 3: attack target 3
    ax.text(0.62, 0.08, r"$\mathcal{P}^{\alpha_1}_1$", fontsize=12, color=TARGET_LABEL_COLORS[1])
    ax.text(0.10, 0.60, r"$\mathcal{P}^{\alpha_1}_2$", fontsize=12, color=TARGET_LABEL_COLORS[2])
    ax.text(0.15, 0.12, r"$\mathcal{P}^{\alpha_1}_3$", fontsize=12, color=TARGET_LABEL_COLORS[3])
    ax.set_title(r"Catalogue $C=\{\alpha_1\}$: 3 best-response regions", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_1.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_1.png"), dpi=150)
    plt.close(fig)


def fig_partition_2():
    alphas = [ALPHA_1, ALPHA_2]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _draw_simplex(ax)
    _shade_best_response_regions(ax, alphas)
    _draw_boundary_lines(ax, alphas)
    ax.set_title(r"Catalogue $C=\{\alpha_1,\alpha_2\}$: refined partition $\mathcal{P}_\sigma$",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_2.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_2.png"), dpi=150)
    plt.close(fig)


def fig_partition_3():
    alphas = [ALPHA_1, ALPHA_2, ALPHA_3]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _draw_simplex(ax)
    _shade_best_response_regions(ax, alphas)
    _draw_boundary_lines(ax, alphas)
    ax.set_title(r"Catalogue $C=\{\alpha_1,\alpha_2,\alpha_3\}$: further refined",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_3.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_3.png"), dpi=150)
    plt.close(fig)


def fig_extreme_points():
    alphas = [ALPHA_1, ALPHA_2]
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    _draw_simplex(ax)
    _shade_best_response_regions(ax, alphas)
    _draw_boundary_lines(ax, alphas)
    pts = compute_extreme_points_2d(alphas)
    for (x, y) in pts:
        ax.plot(x, y, "o", color="black", markersize=7, zorder=10)
        ax.plot(x, y, "o", color="#E63946", markersize=4, zorder=11)
    ax.set_title(r"Extreme point set $\mathcal{E}(C)$: vertices of refined partition",
                 fontsize=11)
    ax.text(1.02, 1.05, r"$\bullet\;$ element of $\mathcal{E}(C)$",
            transform=ax.transAxes, ha="right", fontsize=10, color="#E63946")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "extreme_points.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "extreme_points.png"), dpi=150)
    plt.close(fig)


def fig_eset_growth():
    catalogues = [
        [ALPHA_1],
        [ALPHA_1, ALPHA_2],
        [ALPHA_1, ALPHA_2, ALPHA_3],
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, alphas, label in zip(axes, catalogues, [r"$|C|=1$", r"$|C|=2$", r"$|C|=3$"]):
        _draw_simplex(ax)
        _shade_best_response_regions(ax, alphas)
        _draw_boundary_lines(ax, alphas)
        pts = compute_extreme_points_2d(alphas)
        for (x, y) in pts:
            ax.plot(x, y, "o", color="black", markersize=6, zorder=10)
            ax.plot(x, y, "o", color="#E63946", markersize=3.2, zorder=11)
        ax.set_title(rf"{label},  $|\mathcal{{E}}(C)|={len(pts)}$", fontsize=12)
    fig.suptitle(r"$\mathcal{E}(C)$ grows (and refines) as new attacker types are discovered",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eset_growth.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "eset_growth.png"), dpi=150)
    plt.close(fig)


def fig_seq_vs_simultaneous():
    """Conceptual diagram: sequential (defender commits first, attacker sees and
    best-responds) vs simultaneous (both act without seeing each other)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    def _panel(ax, title, arrows):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold")
        # defender and attacker boxes
        ax.add_patch(plt.Rectangle((0.8, 1.6), 2.8, 1.8, facecolor="#A8DADC",
                                   edgecolor="black", lw=1.3))
        ax.text(2.2, 2.5, "Defender", ha="center", va="center", fontsize=12)
        ax.add_patch(plt.Rectangle((6.4, 1.6), 2.8, 1.8, facecolor="#F6BD60",
                                   edgecolor="black", lw=1.3))
        ax.text(7.8, 2.5, "Attacker\n(type $\\alpha_t$)", ha="center", va="center", fontsize=11)
        for (txt, yshift, arrstyle) in arrows:
            ax.annotate(txt, xy=(6.3, 2.5 + yshift), xytext=(3.7, 2.5 + yshift),
                        fontsize=10, ha="left", va="center",
                        arrowprops=dict(arrowstyle=arrstyle, lw=1.6, color="black"))

    _panel(
        axes[0],
        "Sequential (Stackelberg)\n — defender commits, attacker sees $\\mathbf{p}_t$ and best-responds",
        [("commits $\\mathbf{p}_t$", 0.8, "->"),
         ("attacker sees $\\mathbf{p}_t$\nand best-responds", -0.8, "<-")],
    )
    _panel(
        axes[1],
        "Simultaneous\n — both act without observing the other",
        [("plays $\\mathbf{p}_t$ blind", 0.8, "->"),
         ("plays target blind", -0.8, "<-")],
    )
    fig.suptitle(
        "Defender is worse off under sequential $\\Longrightarrow$ worst-case bound lives here",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sequential_vs_simultaneous.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "sequential_vs_simultaneous.png"), dpi=150)
    plt.close(fig)


def fig_epoch_decomposition():
    """Conceptual time-line with epochs and discovery rounds."""
    fig, ax = plt.subplots(figsize=(11, 2.6))
    T = 10.0
    K = 4
    tau = [0, 2.5, 5.0, 7.5]  # discovery rounds
    labels = [r"$\tau_1$", r"$\tau_2$", r"$\tau_3$", r"$\tau_4$"]
    # draw time axis
    ax.hlines(0, 0, T, colors="black", lw=2)
    ax.plot([0, T], [0, 0], "k>", markersize=10)
    # epochs
    cols = ["#A8DADC", "#F6BD60", "#B5C99A", "#F28482"]
    for i in range(K):
        s = tau[i]
        e = tau[i + 1] if i + 1 < K else T
        ax.add_patch(plt.Rectangle((s, -0.3), e - s, 0.6,
                                   facecolor=cols[i], edgecolor="black",
                                   alpha=0.8))
        ax.text((s + e) / 2, 0, f"Epoch {i+1}\n$|C|={i+1}$", ha="center", va="center",
                fontsize=10)
        ax.plot(s, 0, "o", color="#E63946", markersize=8, zorder=10)
        ax.text(s, 0.55, labels[i], ha="center", color="#E63946", fontsize=11)
    ax.set_xlim(-0.5, T + 0.5)
    ax.set_ylim(-1, 1.3)
    ax.axis("off")
    ax.set_title("Time-line: an 'epoch' begins each time a new attacker type is observed",
                 fontsize=12)
    ax.text(T, -0.7, r"$T$", ha="right", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "epoch_decomposition.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "epoch_decomposition.png"), dpi=150)
    plt.close(fig)


def main():
    fig_simplex()
    fig_partition_1()
    fig_partition_2()
    fig_partition_3()
    fig_extreme_points()
    fig_eset_growth()
    fig_seq_vs_simultaneous()
    fig_epoch_decomposition()
    print("Saved all polytope figures in", FIG_DIR)


if __name__ == "__main__":
    main()
