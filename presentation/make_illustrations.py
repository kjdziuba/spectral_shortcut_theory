"""Generate custom illustrations used in the lab talk.

Run before build_lab_talk_v2.py. Saves PNGs into presentation/img/.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

OUT = Path("/home/u37314kd/Projects/spectral_shortcut_theory/presentation/img")
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1A365D"
BLUE = "#3D7DC7"
ORANGE = "#F58A1F"
GREEN = "#4DB6AC"
RED = "#E57373"
GREY = "#9E9E9E"
LIGHT = "#F2F2F2"
DARK = "#333333"


# ---------------------------------------------------------------- #
# 1. Two-moons + easy line vs right curve
# ---------------------------------------------------------------- #
def make_moons_figure():
    rng = np.random.default_rng(42)
    n = 200
    t = np.linspace(0, np.pi, n // 2)
    # upper moon
    x1 = np.cos(t) + 0.10 * rng.normal(size=n // 2)
    y1 = np.sin(t) + 0.10 * rng.normal(size=n // 2)
    # lower moon (shifted)
    x2 = 1 - np.cos(t) + 0.10 * rng.normal(size=n // 2)
    y2 = -np.sin(t) + 0.5 + 0.10 * rng.normal(size=n // 2)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.scatter(x1, y1, s=140, c=BLUE, edgecolor="white", linewidth=1.4,
               label="cancer cells", zorder=3)
    ax.scatter(x2, y2, s=140, c=ORANGE, edgecolor="white", linewidth=1.4,
               label="normal cells", zorder=3)

    # Easy line
    ax.axline((-0.6, 0.6), (2.0, 0.4), color=RED, lw=4,
              linestyle="--", zorder=2, label="easy line")
    # Right curve (the curved boundary that nails it)
    xs = np.linspace(-0.3, 1.7, 80)
    ys = 0.25 + 0.55 * np.sin(np.pi * (xs - 0.05) / 1.5)
    ax.plot(xs, ys, color=GREEN, lw=4, zorder=2, label="right curve")

    ax.set_xlim(-1.4, 2.4)
    ax.set_ylim(-1.4, 1.7)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95,
              edgecolor=GREY, fancybox=True)

    fig.tight_layout(pad=0.5)
    out = OUT / "moons.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- #
# 2. Pipeline with capacity asymmetry visualised by box size
# ---------------------------------------------------------------- #
def make_pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis("off")

    # Input (spectrum drawing)
    ax.text(0.6, 4.3, "spectrum", fontsize=15, color=DARK, weight="bold")
    xs = np.linspace(0, 1.6, 400)
    ys = 1.6 + 0.85 * (
        np.exp(-((xs - 0.3) ** 2) / 0.012) * 1.0
        + np.exp(-((xs - 0.7) ** 2) / 0.025) * 0.6
        + np.exp(-((xs - 1.1) ** 2) / 0.018) * 0.8
        + 0.05 * np.sin(40 * xs)
    )
    ax.plot(xs + 0.4, ys, color=BLUE, lw=2)
    ax.add_patch(Rectangle((0.4, 1.4), 1.6, 2.1, fill=False,
                            edgecolor=GREY, linewidth=1, linestyle=":"))

    # Arrow into f_theta
    ax.annotate("", xy=(3.0, 2.5), xytext=(2.1, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=NAVY))

    # f_theta — small
    fbox = FancyBboxPatch((3.0, 2.1), 1.0, 0.8,
                          boxstyle="round,pad=0.04,rounding_size=0.08",
                          facecolor=BLUE, edgecolor=NAVY, linewidth=2)
    ax.add_patch(fbox)
    ax.text(3.5, 2.5, r"$f_\theta$", fontsize=20, color="white",
            ha="center", va="center", weight="bold")
    ax.text(3.5, 1.85, "spectral\nreducer", fontsize=11, color=NAVY,
            ha="center", va="top")
    ax.text(3.5, 3.05, "40k params", fontsize=11, color=NAVY,
            ha="center", va="bottom")

    # Arrow
    ax.annotate("", xy=(5.4, 2.5), xytext=(4.2, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=NAVY))

    # g_phi — HUGE
    gbox = FancyBboxPatch((5.4, 0.6), 4.0, 3.8,
                          boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor=ORANGE, edgecolor="#B05C12", linewidth=2)
    ax.add_patch(gbox)
    ax.text(7.4, 2.7, r"$g_\phi$", fontsize=48, color="white",
            ha="center", va="center", weight="bold")
    ax.text(7.4, 1.0, "spatial model (CNN / ViT / …)",
            fontsize=12, color="#5D2F08",
            ha="center", va="center")
    ax.text(7.4, 4.05, "13,000,000 params", fontsize=13,
            color="#5D2F08", ha="center", va="center", weight="bold")

    # Arrow to output
    ax.annotate("", xy=(10.6, 2.5), xytext=(9.6, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=NAVY))

    # Output — tissue map (just a colored grid)
    grid = np.array([
        [0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    colors = np.array([[1, 1, 1, 0], [BLUE, BLUE, BLUE, 0.6],
                       [ORANGE, ORANGE, ORANGE, 0.8]], dtype=object)
    # quick render: just imshow in an inset
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", BLUE, ORANGE])
    ax_inset = ax.inset_axes([10.6 / 12, 0.2, 1.4 / 12, 4.6 / 5])
    ax_inset.imshow(grid, cmap=cmap, interpolation="nearest")
    ax_inset.axis("off")
    ax.text(11.3, 0.3, "prediction", fontsize=12, color=DARK,
            ha="center", va="center")

    # Caption: capacity ratio
    ax.text(6.0, 0.1,
            r"$C_g / C_f \approx 325 \times$  —  the spatial model is 300× bigger",
            fontsize=15, color=NAVY, weight="bold", ha="center", va="bottom")

    fig.tight_layout(pad=0.4)
    out = OUT / "pipeline.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- #
# 3. Chain rule "relay race" of three runners
# ---------------------------------------------------------------- #
def make_relay_figure():
    fig, ax = plt.subplots(figsize=(11, 4), dpi=140)
    ax.set_xlim(0, 11); ax.set_ylim(0, 4)
    ax.axis("off")

    runner_colors = [RED, ORANGE, GREEN]
    runner_labels = ["residual\n∂L/∂ŷ", "spatial Jacobian\n∂ŷ/∂Z",
                     "input\n∂Z/∂θ"]
    runner_centers = [1.6, 5.0, 8.4]

    for cx, color, label in zip(runner_centers, runner_colors, runner_labels):
        # head
        ax.add_patch(Circle((cx, 2.7), 0.32, facecolor=color,
                            edgecolor="white", linewidth=2, zorder=3))
        # body
        ax.add_patch(FancyBboxPatch((cx - 0.18, 1.4), 0.36, 1.0,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor="white",
                                    linewidth=2, zorder=2))
        # baton in hand
        ax.plot([cx + 0.18, cx + 0.65], [1.9, 2.4],
                color=NAVY, lw=4, solid_capstyle="round", zorder=4)
        # label
        ax.text(cx, 0.65, label, fontsize=13, ha="center", va="center",
                color=DARK, weight="bold")

    # Arrows between runners
    for i in range(2):
        x0 = runner_centers[i] + 0.7; x1 = runner_centers[i+1] - 0.7
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color=NAVY))

    # Final arrow to "θ updates"
    ax.annotate("", xy=(10.6, 2.0), xytext=(runner_centers[2] + 0.7, 2.0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=NAVY))
    ax.text(10.7, 2.0, r"$\theta$", fontsize=24, color=NAVY,
            ha="left", va="center", weight="bold")

    # Title
    ax.text(5.5, 3.7, "the gradient relay",
            fontsize=18, color=NAVY, ha="center", weight="bold")

    fig.tight_layout(pad=0.4)
    out = OUT / "relay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- #
# 4. Relay but the first runner has DROPPED the baton
# ---------------------------------------------------------------- #
def make_relay_drop_figure():
    fig, ax = plt.subplots(figsize=(11, 4), dpi=140)
    ax.set_xlim(0, 11); ax.set_ylim(0, 4)
    ax.axis("off")

    runner_colors = [RED, ORANGE, GREEN]
    runner_centers = [1.6, 5.0, 8.4]
    runner_labels = ["residual = 0\nthe error vanished",
                     "still ready\n(but no signal coming)",
                     "still ready\n(but no signal coming)"]

    for i, (cx, color, label) in enumerate(zip(runner_centers, runner_colors,
                                               runner_labels)):
        head_alpha = 0.4 if i > 0 else 1.0
        ax.add_patch(Circle((cx, 2.7), 0.32, facecolor=color,
                            edgecolor="white", linewidth=2, zorder=3,
                            alpha=head_alpha if i > 0 else 1.0))
        ax.add_patch(FancyBboxPatch((cx - 0.18, 1.4), 0.36, 1.0,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor="white",
                                    linewidth=2, zorder=2,
                                    alpha=head_alpha if i > 0 else 1.0))
        ax.text(cx, 0.65, label, fontsize=11, ha="center", va="center",
                color=DARK if i == 0 else GREY)

    # The first runner has DROPPED the baton on the floor
    ax.plot([runner_centers[0] - 0.3, runner_centers[0] + 0.3],
            [1.0, 1.1], color=NAVY, lw=4, solid_capstyle="round", zorder=4)
    # big red X over the baton
    ax.plot([runner_centers[0] - 0.4, runner_centers[0] + 0.4],
            [0.8, 1.3], color=RED, lw=3, zorder=5)
    ax.plot([runner_centers[0] - 0.4, runner_centers[0] + 0.4],
            [1.3, 0.8], color=RED, lw=3, zorder=5)
    ax.text(runner_centers[0], 0.0,
            "baton on the floor", fontsize=11, color=RED,
            ha="center", weight="bold")

    # No arrows between runners — dotted ghost arrows
    for i in range(2):
        x0 = runner_centers[i] + 0.7; x1 = runner_centers[i+1] - 0.7
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5,
                                    color=GREY, linestyle=":"))

    # Ghost arrow to theta with a giant red X
    ax.annotate("", xy=(10.6, 2.0), xytext=(runner_centers[2] + 0.7, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.5,
                                color=GREY, linestyle=":"))
    ax.text(10.7, 2.0, r"$\theta$", fontsize=24, color=GREY,
            ha="left", va="center", weight="bold", alpha=0.5)
    ax.text(10.95, 1.45, "no update", fontsize=11, color=RED,
            ha="center", weight="bold")

    # Title
    ax.text(5.5, 3.7, "what happens after a few epochs",
            fontsize=18, color=RED, ha="center", weight="bold")

    fig.tight_layout(pad=0.4)
    out = OUT / "relay_drop.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- #
# 5. The "F1 paradox" big numbers
# ---------------------------------------------------------------- #
def make_paradox_figure():
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis("off")

    # LEFT panel
    ax.add_patch(FancyBboxPatch((0.4, 0.5), 5.0, 4.0,
                                 boxstyle="round,pad=0.06,rounding_size=0.15",
                                 facecolor="#FFEEEE", edgecolor=RED, linewidth=2))
    ax.text(2.9, 4.05, "trained the normal way", fontsize=15,
            ha="center", color=DARK, weight="bold")
    ax.text(2.9, 3.55, "(end-to-end joint)", fontsize=12, ha="center", color=GREY)
    ax.text(2.9, 2.0, "0.67", fontsize=88, ha="center", va="center",
            color=RED, weight="bold")
    ax.text(2.9, 0.85, "test F1", fontsize=14, ha="center", color=DARK)

    # RIGHT panel
    ax.add_patch(FancyBboxPatch((6.6, 0.5), 5.0, 4.0,
                                 boxstyle="round,pad=0.06,rounding_size=0.15",
                                 facecolor="#ECF7EC", edgecolor=GREEN, linewidth=2))
    ax.text(9.1, 4.05, "froze half the model", fontsize=15,
            ha="center", color=DARK, weight="bold")
    ax.text(9.1, 3.55, "(spectral module locked)", fontsize=12,
            ha="center", color=GREY)
    ax.text(9.1, 2.0, "0.95", fontsize=88, ha="center", va="center",
            color=GREEN, weight="bold")
    ax.text(9.1, 0.85, "test F1", fontsize=14, ha="center", color=DARK)

    # Big question mark arrow between them
    ax.annotate("", xy=(6.5, 2.5), xytext=(5.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=4, color=NAVY))
    ax.text(6.0, 3.2, "???", fontsize=44, ha="center", va="center",
            color=NAVY, weight="bold")

    fig.tight_layout(pad=0.4)
    out = OUT / "paradox.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- #
# 6. The freeze prescription — three big icons in a row
# ---------------------------------------------------------------- #
def make_prescription_figure():
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=140)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    ax.axis("off")

    pos = [2.0, 6.0, 10.0]
    titles = ["freeze it", "or anchor it", "watch the meter"]
    captions = [
        "lock the spectral\nmodule outright",
        "inject a fixed\nbaseline (PCA, etc)",
        "track EGR during\ntraining as a signal",
    ]
    colors = [GREEN, BLUE, ORANGE]

    for cx, t, cap, c in zip(pos, titles, captions, colors):
        # icon circle
        ax.add_patch(Circle((cx, 3.3), 1.0, facecolor=c, edgecolor="white",
                            linewidth=4))

    # custom icon 1: padlock (rectangle + half-arc shackle)
    cx = pos[0]
    body = Rectangle((cx - 0.35, 2.8), 0.7, 0.55,
                     facecolor="white", edgecolor="white", linewidth=0)
    ax.add_patch(body)
    # keyhole
    ax.add_patch(Circle((cx, 3.1), 0.07, facecolor=colors[0]))
    ax.add_patch(Rectangle((cx - 0.04, 2.92), 0.08, 0.2, facecolor=colors[0]))
    # shackle (half circle)
    from matplotlib.patches import Arc
    ax.add_patch(Arc((cx, 3.45), 0.55, 0.55, theta1=0, theta2=180,
                     edgecolor="white", linewidth=6))
    # vertical legs of shackle
    ax.plot([cx - 0.275, cx - 0.275], [3.45, 3.35],
            color="white", lw=6, solid_capstyle="round")
    ax.plot([cx + 0.275, cx + 0.275], [3.45, 3.35],
            color="white", lw=6, solid_capstyle="round")

    # custom icon 2: anchor (vertical line + curved bottom + ring on top)
    cx = pos[1]
    # ring
    from matplotlib.patches import Wedge
    ax.add_patch(Circle((cx, 3.85), 0.13, facecolor="none",
                        edgecolor="white", linewidth=5))
    # vertical
    ax.plot([cx, cx], [3.72, 3.0], color="white", lw=6, solid_capstyle="round")
    # crossbar
    ax.plot([cx - 0.32, cx + 0.32], [3.55, 3.55],
            color="white", lw=6, solid_capstyle="round")
    # bottom arc
    ax.add_patch(Arc((cx, 3.0), 0.8, 0.5, theta1=180, theta2=360,
                     edgecolor="white", linewidth=5))
    # spikes on arc tips
    ax.plot([cx - 0.4, cx - 0.45], [3.0, 2.9],
            color="white", lw=5, solid_capstyle="round")
    ax.plot([cx + 0.4, cx + 0.45], [3.0, 2.9],
            color="white", lw=5, solid_capstyle="round")

    # custom icon 3: gauge / chart (three vertical bars)
    cx = pos[2]
    bar_x = [cx - 0.4, cx - 0.05, cx + 0.3]
    bar_h = [0.35, 0.65, 0.5]
    for bx, bh in zip(bar_x, bar_h):
        ax.add_patch(Rectangle((bx - 0.10, 2.85), 0.20, bh,
                                facecolor="white", edgecolor="white"))
    # baseline
    ax.plot([cx - 0.55, cx + 0.55], [2.85, 2.85],
            color="white", lw=4, solid_capstyle="round")

    for cx, t, cap in zip(pos, titles, captions):
        ax.text(cx, 1.7, t, fontsize=20, ha="center", va="center",
                weight="bold", color=DARK)
        ax.text(cx, 1.0, cap, fontsize=12, ha="center", va="center",
                color=GREY)

    fig.tight_layout(pad=0.4)
    out = OUT / "prescription.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- #
if __name__ == "__main__":
    paths = []
    paths.append(make_moons_figure())
    paths.append(make_pipeline_figure())
    paths.append(make_relay_figure())
    paths.append(make_relay_drop_figure())
    paths.append(make_paradox_figure())
    paths.append(make_prescription_figure())
    for p in paths:
        sz = p.stat().st_size / 1024
        print(f"wrote {p}  ({sz:.1f} KB)")
