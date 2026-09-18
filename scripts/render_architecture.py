"""
scripts/render_architecture.py — LEVEL 3.2 architecture diagram (pure matplotlib, no external CLI).

    python scripts/render_architecture.py --out docs/assets/architecture.png

Fast loop (every frame) vs slow loop (event-driven relocalization), with badges carrying numbers
measured in Levels 0-2. The Mermaid source of the same structure lives in docs/architecture.md.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch             # noqa: E402

FAST, FAST_BG = "#1b7f3b", "#e8f5ec"
SLOW, SLOW_BG = "#b36b00", "#fdf2e1"
IN, IN_BG = "#3a5a8c", "#eaf0f8"
INK, MUTED = "#1f2328", "#57606a"
BADGE_BG = "#ffffff"


def box(ax, x, y, w, h, title, sub, edge, fill="#ffffff", title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fill, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=title_size, color=INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.2, color=MUTED)
    return (x, y, w, h)


def badge(ax, x, y, text, edge):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.8, color=edge, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=BADGE_BG, ec=edge, lw=1.0))


def arrow(ax, p, q, color=INK, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2))


def lane(ax, x, y, w, h, title, subtitle, edge, fill, title_at_bottom=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02", fc=fill, ec=edge, lw=2.0))
    ty, sy = (y + 0.050, y + 0.020) if title_at_bottom else (y + h - 0.03, y + h - 0.062)
    ax.text(x + 0.012, ty, title, ha="left", va="center", fontsize=13, color=edge, weight="bold")
    ax.text(x + 0.012, sy, subtitle, ha="left", va="center", fontsize=9, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/architecture.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(16, 9.4), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.965, "UAV-VisLoc — fast loop (continuation) vs slow loop (relocalization)", ha="center",
            fontsize=17, weight="bold", color=INK)
    ax.text(0.5, 0.935, "Badges: measured on GeoTest1 (Levels 0–2). Ablation deltas = RMSE change when the component is removed.",
            ha="center", fontsize=9.5, color=MUTED)

    # ---- inputs column -------------------------------------------------------------------------
    lane(ax, 0.015, 0.20, 0.155, 0.70, "INPUTS", "", IN, IN_BG)
    cam = box(ax, 0.027, 0.70, 0.131, 0.11, "Camera frame", "nadir, north-up\n1920×1080", IN)
    mapb = box(ax, 0.027, 0.45, 0.131, 0.12, "Satellite map", "landmark catalogue\n(SAHI YOLO, offline)", IN)
    nav = box(ax, 0.027, 0.235, 0.131, 0.15, "NavSource",
              "leg heading (FC-log replay)\n→ clamp frame (fast loop)\n→ flashlight (slow loop)\nscale prior 0.5 → one-time lock", IN)

    # ---- fast loop lane (top) ------------------------------------------------------------------------
    lane(ax, 0.185, 0.58, 0.80, 0.32, "FAST LOOP  ·  every frame",
         "~58 ms/frame median on RTX 5060 (YOLO ≈ 35%)  ·  386 of 401 frames are continuation only", FAST, FAST_BG)
    fy, fh, fw = 0.625, 0.13, 0.135
    xs = [0.200, 0.357, 0.514, 0.671, 0.828]
    y1 = box(ax, xs[0], fy, fw, fh, "YOLO landmarks", "building · tree ·\ninfrastructure", FAST)
    t1 = box(ax, xs[1], fy, fw, fh, "Pixel tracker", "persistent IDs,\nvelocity prediction, coasting", FAST)
    b1 = box(ax, xs[2], fy, fw, fh, "Bound pairs", "track ↔ map landmark\n(set at last commit)", FAST)
    r1 = box(ax, xs[3], fy, fw, fh, "Deterministic RANSAC", "translation consensus\n@ locked scale", FAST)
    k1 = box(ax, xs[4], fy, fw, fh, "Clamp + estimate", "offset absorption; clamp 15/8/4 px\nper frame in NavSource leg frame", FAST)
    for a, b in ((y1, t1), (t1, b1), (b1, r1), (r1, k1)):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2), FAST, lw=2.0)
    badge(ax, xs[1] + fw / 2, fy + fh + 0.022, "A_coast no coasted votes: +40%", FAST)
    badge(ax, xs[3] + fw / 2, fy + fh + 0.022, "A1 no RANSAC: +68%", FAST)
    badge(ax, xs[4] + fw / 2, fy + fh + 0.022, "A_prior static heading: +138%", IN)

    # ---- slow loop lane (bottom) ----------------------------------------------------------------------
    lane(ax, 0.185, 0.20, 0.80, 0.27, "SLOW LOOP  ·  event-driven relocalization",
         "15 of 401 frames (all pre-emptive handoffs)  ·  candidate generation ~355 ms median when it runs", SLOW, SLOW_BG,
         title_at_bottom=True)
    sy, sh, sw = 0.30, 0.13, 0.135
    g1 = box(ax, xs[0], sy, sw, sh, "Video route graph", "landmarks as a path\n(≤10 vertices)", SLOW)
    v1 = box(ax, xs[1], sy, sw, sh, "Viewport + A*", "map routes in camera\nfootprint, class-matched", SLOW)
    c1 = box(ax, xs[2], sy, sw, sh, "Graph comparison", "internal angles +\nedge-length RATIOS", SLOW)
    tp = box(ax, xs[3], sy, sw, sh, "TOPSIS arbitration", "shape · ratio ·\nconstellation affinity", SLOW)
    cm = box(ax, xs[4], sy, sw, sh, "Commit", "new bindings;\none-time scale lock at boot", SLOW)
    for a, b in ((g1, v1), (v1, c1), (c1, tp), (tp, cm)):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2), SLOW, lw=2.0)
    badge(ax, xs[2] + sw / 2, sy + sh + 0.020, "A3b no edges: +199% (wrong boot scale)", SLOW)
    badge(ax, xs[3] + sw / 2, sy - 0.020, "A2 single criterion: +26% / +50%", SLOW)

    # ---- edges between inputs and lanes ----------------------------------------------------------------
    arrow(ax, (cam[0] + cam[2], cam[1] + cam[3] / 2), (y1[0], y1[1] + y1[3] * 0.55), IN, lw=1.6)
    arrow(ax, (mapb[0] + mapb[2], mapb[1] + mapb[3] * 0.35), (v1[0] + 0.03, v1[1] + v1[3]), IN, lw=1.4, rad=0.12)
    arrow(ax, (nav[0] + nav[2], nav[1] + nav[3] * 0.5), (g1[0], g1[1] + g1[3] * 0.5), IN, lw=1.4, ls="--")

    # ---- trigger / rebind, in the gap between the lanes ---------------------------------------------------
    arrow(ax, (b1[0] + b1[2] * 0.25, b1[1]), (g1[0] + g1[2] * 0.80, g1[1] + g1[3]), SLOW, lw=2.8, rad=0.10)
    ax.text(0.262, 0.515, "TRIGGER\nboot · tracking lost\n≤2 bound pairs", fontsize=9.5, color=SLOW,
            weight="bold", ha="center", va="center")
    arrow(ax, (cm[0] + cm[2] * 0.40, cm[1] + cm[3]), (b1[0] + b1[2] * 0.80, b1[1]), FAST, lw=2.8, rad=0.10)
    ax.text(0.905, 0.515, "REBIND\nnew track ↔ map\npairs", fontsize=9.5, color=FAST,
            weight="bold", ha="center", va="center")

    # ---- known limits strip ----------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.015, 0.03), 0.97, 0.135, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc="#f6f8fa", ec="#d0d7de", lw=1.2))
    ax.text(0.03, 0.14, "Known limits (Level 2 evidence)", fontsize=11, weight="bold", color=INK)
    limits = [
        "• Trigger counts bound pairs, not their quality: 3 pairs with 1 inlier, or 8 stale pairs, never fire a search.",
        "• The 8 px/frame lateral clamp paces both drift and recovery; the heading prior (waypoint legs) disagrees with image motion at turns.",
        "• Translation-only solver: the camera must stay north-up nadir. Ground truth is an uncorrected 4-waypoint polyline; LSR stays 100% even at 70% dropout.",
    ]
    for i, t in enumerate(limits):
        ax.text(0.03, 0.108 - i * 0.03, t, fontsize=9, color=MUTED)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, facecolor="white")
    print(f"[ARCH] wrote {args.out}")


if __name__ == "__main__":
    main()
