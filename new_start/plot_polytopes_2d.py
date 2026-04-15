"""2D triangle polytope / best-response-region visualizations for n=2.

Following Balcan et al. (EC 2015) Figure 1: with n=2 targets and one resource
that may also be idle, the coverage space is the 2D triangle
    P = { (p_1, p_2) : p_1 >= 0, p_2 >= 0, p_1 + p_2 <= 1 }.
Each attacker type alpha has ONE best-response boundary (the single
inequality U_alpha(1, p) = U_alpha(2, p)), which cuts the triangle with a line.

Outputs (in presentation/figures/):
  simplex_2t.pdf               bare 2-target triangle
  partition_2t_1.pdf           partition with 1 attacker (2 regions)
  partition_2t_2.pdf           partition with 2 attackers (refined)
  partition_2t_3.pdf           partition with 3 attackers (refined)
  extreme_points_2t.pdf        endpoints + best-response boundary intersections marked
  eset_growth_2t.pdf           |C|=1,2,3 side-by-side growth panel
"""

from __future__ import annotations

import os
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import to_rgba

FIG_DIR = "presentation/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Palette -- matches the presentation slides (green + cream + blue accents).
DARKGREEN = "#006857"
CREAM = "#FDF7E7"
BLUE = "#1C4587"
BLUE_SOFT = "#4F78B3"
REGION_COLORS = [
    "#C8D7EC",
    "#FCE3B4",
    "#CFE6D4",
    "#F4CCC0",
    "#DDD2E8",
    "#FFE9A8",
    "#D5E7D9",
]

# 2D triangle  { (p1, p2) : p1 >= 0, p2 >= 0, p1 + p2 <= 1 }
TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


# ----------------------------------------------------------------------------
# Attacker types (concrete utilities chosen for visually clear partitions)
# ----------------------------------------------------------------------------
#
# For n=2 the attacker boundary in (p_1, p_2) space is the line
#    Delta_1 p_1 - Delta_2 p_2 = u^u(2) - u^u(1)
# where Delta_i = u^c(i) - u^u(i).
# We pick utilities so the boundaries cut the triangle at distinct places.

ALPHA_1 = (np.array([-0.5, -0.2]), np.array([0.6, 0.9]))  # tends to attack 2
ALPHA_2 = (np.array([-0.3, -0.6]), np.array([0.9, 0.3]))  # tends to attack 1
ALPHA_3 = (np.array([-0.1, -0.4]), np.array([0.5, 0.8]))  # intermediate


def _br_line(alpha):
    """Return (a_x, a_y, b) with a_x * p_1 + a_y * p_2 = b for the line
    U_alpha(1, p) = U_alpha(2, p)."""
    u_c, u_u = alpha
    D1 = u_c[0] - u_u[0]
    D2 = u_c[1] - u_u[1]
    rhs = u_u[1] - u_u[0]
    return D1, -D2, rhs


def _best_response(alpha, p):
    """0 -> attacks target 1, 1 -> attacks target 2."""
    u_c, u_u = alpha
    utils = u_c * p + u_u * (1 - p)
    return int(np.argmax(utils))


# ----------------------------------------------------------------------------
# Axes helpers
# ----------------------------------------------------------------------------


def _draw_triangle(ax, title=""):
    tri = MplPolygon(TRI_VERTS, closed=True, facecolor="none",
                     edgecolor="black", lw=2.0, zorder=5)
    ax.add_patch(tri)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.22, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, color=DARKGREEN, pad=8)
    # Vertex annotations
    ax.annotate(r"$(p_1,p_2)=(0,0)$  idle", xy=(0, 0), xytext=(-0.05, -0.16),
                fontsize=9, ha="right", color="#555",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate(r"$(1,0)$: always target 1", xy=(1, 0), xytext=(1.05, -0.1),
                fontsize=9, ha="left", color="#555",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate(r"$(0,1)$: always target 2", xy=(0, 1), xytext=(-0.05, 1.1),
                fontsize=9, ha="right", color="#555",
                arrowprops=dict(arrowstyle="->", color="#555"))


def _shade_regions(ax, alphas, grid_res=600):
    """Color each grid cell by which (target-for-alpha_1, target-for-alpha_2, ...) tuple it falls in."""
    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    X, Y = np.meshgrid(xs, ys)
    inside = (X >= 0) & (Y >= 0) & (X + Y <= 1)
    # Compose an integer region id encoding the best-response of each alpha.
    region_id = np.zeros_like(X, dtype=int)
    for k, alpha in enumerate(alphas):
        u_c, u_u = alpha
        p_stack = np.stack([X, Y], axis=-1)                        # H, W, 2
        utils = u_c[None, None, :] * p_stack + u_u[None, None, :] * (1 - p_stack)
        br = np.argmax(utils, axis=-1)                             # 0 or 1
        region_id = region_id * 2 + br
    region_id = np.where(inside, region_id, -1)
    # Map each distinct id to a color
    unique_ids = sorted(set(region_id[inside].tolist()))
    color_img = np.ones((*region_id.shape, 4), dtype=float)
    color_img[..., 3] = 0.0  # transparent outside triangle
    for k, uid in enumerate(unique_ids):
        rgba = to_rgba(REGION_COLORS[k % len(REGION_COLORS)], alpha=0.85)
        color_img[region_id == uid] = rgba
    ax.imshow(color_img, extent=[0, 1, 0, 1], origin="lower", zorder=2)


def _draw_br_boundaries(ax, alphas):
    colors = [BLUE, "#B8000C", "#2A7A3E", "#7A3FB8"]
    styles = ["-", "--", "-.", ":"]
    for k, alpha in enumerate(alphas):
        a_x, a_y, b = _br_line(alpha)
        col = colors[k % len(colors)]
        sty = styles[k % len(styles)]
        # Parameterise the line: find two points where it intersects the triangle boundary.
        segs = []
        # Intersect with the triangle edges
        edges = [((0, 0), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (0, 0))]
        for (p0, p1) in edges:
            p0 = np.array(p0); p1 = np.array(p1)
            d = p1 - p0
            denom = a_x * d[0] + a_y * d[1]
            if abs(denom) < 1e-10:
                continue
            t = (b - a_x * p0[0] - a_y * p0[1]) / denom
            if -1e-8 <= t <= 1 + 1e-8:
                segs.append(p0 + t * d)
        if len(segs) >= 2:
            pt1, pt2 = segs[0], segs[1]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                    sty, color=col, lw=2.0, zorder=8,
                    label=rf"$\alpha_{{{k+1}}}$ boundary" if k < 3 else None)


# ----------------------------------------------------------------------------
# Extreme point computation for the 2D case
# ----------------------------------------------------------------------------


def _triangle_halfspaces():
    # a_x p_1 + a_y p_2 = b    (we enumerate equalities for vertex extraction)
    return [
        (np.array([1.0, 0.0]), 0.0),   # p_1 = 0
        (np.array([0.0, 1.0]), 0.0),   # p_2 = 0
        (np.array([1.0, 1.0]), 1.0),   # p_1 + p_2 = 1
    ]


def compute_extreme_points(alphas):
    hps = _triangle_halfspaces()
    for alpha in alphas:
        a_x, a_y, b = _br_line(alpha)
        hps.append((np.array([a_x, a_y]), b))
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
# Figures
# ----------------------------------------------------------------------------


def fig_simplex_2t():
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    _draw_triangle(ax, r"Strategy space $\mathcal{P}$ for $n=2$: triangle $\{p_1,p_2\geq 0,\;p_1{+}p_2\leq 1\}$")
    # White fill
    tri = MplPolygon(TRI_VERTS, closed=True, facecolor="white",
                     edgecolor="none", lw=0, zorder=4)
    ax.add_patch(tri)
    _draw_triangle(ax, r"Strategy space $\mathcal{P}$ for $n=2$: triangle $\{p_1,p_2\geq 0,\;p_1{+}p_2\leq 1\}$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "simplex_2t.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "simplex_2t.png"), dpi=150)
    plt.close(fig)


def _make_panel(alphas, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    _draw_triangle(ax, title)
    _shade_regions(ax, alphas)
    # redraw triangle border on top
    tri = MplPolygon(TRI_VERTS, closed=True, facecolor="none",
                     edgecolor="black", lw=2.0, zorder=7)
    ax.add_patch(tri)
    _draw_br_boundaries(ax, alphas)
    return fig, ax


def fig_partition_2t_1():
    fig, ax = _make_panel([ALPHA_1], r"Catalogue $C=\{\alpha_1\}$: 2 best-response regions")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_2t_1.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_2t_1.png"), dpi=150)
    plt.close(fig)


def fig_partition_2t_2():
    fig, ax = _make_panel([ALPHA_1, ALPHA_2],
                          r"Catalogue $C=\{\alpha_1,\alpha_2\}$: refined partition")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_2t_2.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_2t_2.png"), dpi=150)
    plt.close(fig)


def fig_partition_2t_3():
    fig, ax = _make_panel([ALPHA_1, ALPHA_2, ALPHA_3],
                          r"Catalogue $C=\{\alpha_1,\alpha_2,\alpha_3\}$: further refined")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "partition_2t_3.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "partition_2t_3.png"), dpi=150)
    plt.close(fig)


def fig_extreme_points_2t():
    fig, ax = _make_panel([ALPHA_1, ALPHA_2],
                          r"Extreme point set $\mathcal{E}(C)$ for $C=\{\alpha_1,\alpha_2\}$")
    pts = compute_extreme_points([ALPHA_1, ALPHA_2])
    for (x, y) in pts:
        ax.plot(x, y, "o", color="black", markersize=8, zorder=10)
        ax.plot(x, y, "o", color="#B3213A", markersize=4.5, zorder=11)
    ax.text(1.12, 1.08, rf"$|\mathcal{{E}}(C)|={len(pts)}$",
            transform=ax.transAxes, ha="right", fontsize=10,
            color="#B3213A")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "extreme_points_2t.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "extreme_points_2t.png"), dpi=150)
    plt.close(fig)


def fig_eset_growth_2t():
    catalogues = [[ALPHA_1], [ALPHA_1, ALPHA_2], [ALPHA_1, ALPHA_2, ALPHA_3]]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    for ax, alphas, label in zip(axes, catalogues,
                                 [r"$|C|=1$", r"$|C|=2$", r"$|C|=3$"]):
        _draw_triangle(ax, "")
        _shade_regions(ax, alphas)
        tri = MplPolygon(TRI_VERTS, closed=True, facecolor="none",
                         edgecolor="black", lw=2.0, zorder=7)
        ax.add_patch(tri)
        _draw_br_boundaries(ax, alphas)
        pts = compute_extreme_points(alphas)
        for (x, y) in pts:
            ax.plot(x, y, "o", color="black", markersize=7, zorder=10)
            ax.plot(x, y, "o", color="#B3213A", markersize=3.8, zorder=11)
        ax.set_title(rf"{label},  $|\mathcal{{E}}(C)|={len(pts)}$", fontsize=13,
                     color=DARKGREEN)
    fig.suptitle(r"$\mathcal{E}(C)$ grows (and refines) as new attacker types are discovered",
                 fontsize=14, color=DARKGREEN, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eset_growth_2t.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "eset_growth_2t.png"), dpi=150)
    plt.close(fig)


def main():
    fig_simplex_2t()
    fig_partition_2t_1()
    fig_partition_2t_2()
    fig_partition_2t_3()
    fig_extreme_points_2t()
    fig_eset_growth_2t()
    print("Saved 2D triangle polytope figures in", FIG_DIR)


if __name__ == "__main__":
    main()
