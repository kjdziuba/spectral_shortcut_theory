"""Fig 1 (v2 paper): the serial model, the two cues and the three test conditions.

Schematic only (no data): a 3x3 neighbourhood of pixel spectra -> shared
per-pixel encoder W -> spatial head of width M reading the encoded
neighbourhood -> logit. Left: where the two cues live (spectral cue in the
centre pixel along an unknown direction u; contextual cue in the eight
neighbours along large-amplitude directions v_d). Right: the three paired
evaluation conditions (iid / reversed / context-random).

Output: paper/figures/fig1_model.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 7, "figure.dpi": 200})
fig, ax = plt.subplots(figsize=(5.6, 1.75))
ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")

C_SPEC, C_CTX, C_ENC, C_HEAD = "#d95f02", "#1b9e77", "#4575b4", "#7570b3"

# --- 3x3 neighbourhood of spectra ------------------------------------------
x0, y0, s = 2, 6, 5.2
for i in range(3):
    for j in range(3):
        centre = (i == 1 and j == 1)
        fc = "#fde0c8" if centre else "#d7efe6"
        ec = C_SPEC if centre else C_CTX
        ax.add_patch(Rectangle((x0 + j * s, y0 + (2 - i) * s), s * 0.92, s * 0.92, fc=fc, ec=ec, lw=1.1))
ax.text(x0 + 1.5 * s - 0.2, y0 + 3 * s + 1.0, r"pixel spectra $x_p\in\mathbb{R}^S$", ha="center", va="bottom", fontsize=6.5)
ax.text(x0 + 1.5 * s - 0.2, y0 - 1.2, "centre: spectral cue\n" + r"$\alpha\,y_p\,u$ + isotropic noise",
        ha="center", va="top", fontsize=5.8, color=C_SPEC)
ax.text(x0 + 3 * s + 1.0, y0 + 2.6 * s, "neighbours: contextual cue\n" + r"$\beta\,(y_p+\tau\eta)\,v_d$, large amplitude",
        ha="left", va="center", fontsize=5.8, color=C_CTX)

# --- encoder -----------------------------------------------------------------
ex = 40
ax.add_patch(FancyBboxPatch((ex, 9), 11, 10, boxstyle="round,pad=0.4", fc="#e3ecf7", ec=C_ENC, lw=1.2))
ax.text(ex + 5.5, 15.5, "encoder", ha="center", va="center", fontsize=6.5, color=C_ENC)
ax.text(ex + 5.5, 12.3, r"$W\in\mathbb{R}^{K\times S}$" + "\nshared, per pixel", ha="center", va="center", fontsize=5.8)
ax.add_patch(FancyArrowPatch((x0 + 3 * s + 0.5, y0 + 1.5 * s), (ex - 0.8, 14), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.3"))
ax.text((x0 + 3 * s + ex) / 2 + 0.5, 15.6, "each pixel", ha="center", fontsize=5.5, color="0.3")

# --- head ---------------------------------------------------------------------
hx = 58
ax.add_patch(FancyBboxPatch((hx, 7), 15, 14, boxstyle="round,pad=0.4", fc="#ece9f5", ec=C_HEAD, lw=1.2))
ax.text(hx + 7.5, 18.3, "spatial head", ha="center", va="center", fontsize=6.5, color=C_HEAD)
ax.text(hx + 7.5, 13.3, "width $M$, ReLU\nreads the encoded\n$3\\times3$ neighbourhood\nrate $\\kappa\\eta$", ha="center", va="center", fontsize=5.6)
ax.add_patch(FancyArrowPatch((ex + 11.8, 14), (hx - 0.8, 14), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.3"))
ax.text((ex + 12 + hx) / 2, 15.6, r"$z_p=Wx_p$", ha="center", fontsize=5.8, color="0.3")
ax.add_patch(FancyArrowPatch((hx + 15.8, 14), (hx + 21, 14), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.3"))
ax.text(hx + 21.5, 14, r"logit $F_p$", ha="left", va="center", fontsize=6.2)
ax.text(hx + 21.5, 10.8, r"$\mathcal{L}=\overline{\log(1+e^{-y_pF_p})}$", ha="left", va="center", fontsize=5.8)
ax.text(hx + 21.5, 8.2, "one global rate $\\eta$;\nstop at matched fit $L^\\ast$", ha="left", va="center", fontsize=5.6)

# --- test conditions ------------------------------------------------------------
ty = 27.5
ax.text(hx + 7.5, ty + 2.2, "paired test conditions (same centres, labels, noise)", ha="center", va="center", fontsize=6.2)
for k, (name, desc, col) in enumerate([("iid", "context = label", "0.25"),
                                        ("reversed", "context = $-$label", "#b2182b"),
                                        ("context-random", "context independent", "0.45")]):
    xx = hx - 14 + k * 20
    ax.text(xx, ty - 0.8, name, ha="center", va="center", fontsize=6, color=col, fontweight="bold")
    ax.text(xx, ty - 3.0, desc, ha="center", va="center", fontsize=5.5, color=col)

fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_model.{ext}", bbox_inches="tight")
print(f"saved {OUT / 'fig1_model.pdf'}")
