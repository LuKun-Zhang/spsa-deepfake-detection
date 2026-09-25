# -*- coding: utf-8 -*-
"""ICASSP mechanism evidence-board figure v1 (fig_mech_board).

Two-row board.  (A) real s2048 controlled trajectories showing WHY SWA alone
fails (its deployed average theta-bar falls below its own trajectory mean) and
SPSA fixes it (theta-bar rides the path).  (B) paired 7-pt-mean gains against
matched CTRL with the +/-0.95-pt pairing-noise band: only the structured
SPSA+SWA combination clears the band.  Full spec in fig_mech_board_spec.md.

Data (frame-level AUC, Celeb-DF-v2, controlled 4-cycle), verified line-by-line
from spsa1_ctrl_verified_data.md / controlled main table; no invented numbers.
  CTRL+SWA s2048: [0.7420,0.6991,0.7016,0.7150,0.7119,0.7031,0.6989]
                  mean 0.7102, theta-bar 0.6987 -> d = -1.15  (3-seed |d| 1.03)
  SPSA+SWA s2048: [0.7555,0.7162,0.7491,0.7086,0.7290,0.7275,0.7243]
                  mean 0.7300, theta-bar 0.7258 -> d = -0.42  (3-seed |d| 0.51)
  Panel B paired gains vs matched CTRL (pts): SPSA only -0.38 (3), SWA only
  -0.39 (3), SPSA+SWA +1.27 (3), SAM +0.28 (3-seed mean; per-seed
  +0.23 / -0.63 / +1.24, 993 probe).

Output: fig_mech_board.pdf (+ preview png).  Single column, width <= 0.92
\columnwidth, rendered height ~2.5 in (see fig_mech_board_spec.md).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORANGE = "#d97706"   # CTRL+SWA (no SPSA) side
BLUE = "#1f5fa8"     # SPSA+SWA side / protagonist
GREY_D = "#444"; GREY_M = "#888"

stages = ["E0e", "E1m", "E1e", "E2m", "E2e", "E3m", "E3e"]
X = np.arange(7)

panels = [
    dict(key="CTRL + SWA", sub="no SPSA",
         pts=[0.7420, 0.6991, 0.7016, 0.7150, 0.7119, 0.7031, 0.6989],
         mean=0.7102, thetabar=0.6987, d3="mean |d| = 1.03 (3 seeds)",
         color=ORANGE),
    dict(key="SPSA + SWA", sub="",
         pts=[0.7555, 0.7162, 0.7491, 0.7086, 0.7290, 0.7275, 0.7243],
         mean=0.7300, thetabar=0.7258, d3="mean |d| = 0.51 (3 seeds)",
         color=BLUE),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

fig = plt.figure(figsize=(3.4, 2.68))
# explicit axes: two small panels side-by-side on top, one full-width row below
axL = fig.add_axes([0.105, 0.575, 0.385, 0.395])
axR = fig.add_axes([0.545, 0.575, 0.385, 0.395])
axB = fig.add_axes([0.105, 0.205, 0.825, 0.310])

for ax in (axL, axR):
    ax.tick_params(axis="y", labelsize=5.2, pad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.6)

# ---------------- Panel A ----------------
for ax, g in zip((axL, axR), panels):
    col = g["color"]
    ax.axvspan(3.5, 6.0, color="#dddddd", alpha=0.55, zorder=0)
    ax.text(4.75, 0.797, "SWA collects\n(last cycle)", ha="center", va="top",
            fontsize=4.5, color="#666", linespacing=1.02)
    ax.plot(X, g["pts"], color=col, lw=1.4, marker="o", ms=3.0, zorder=4)
    ax.axhline(g["mean"], color=GREY_D, lw=0.8, ls=":", zorder=3)
    ax.text(6.16, g["mean"] - 0.0026, "7-pt mean", fontsize=4.8, color="#333",
            va="top", ha="left")
    ax.axhline(g["thetabar"], color=col, lw=1.5, ls="--", zorder=5)
    ax.annotate("", xy=(5.82, g["mean"]), xytext=(5.82, g["thetabar"]),
                arrowprops=dict(arrowstyle="<->", color="#111", lw=0.8))
    dpts = (g["thetabar"] - g["mean"]) * 100.0
    ax.text(5.90, (g["thetabar"] + g["mean"]) / 2, f"{dpts:+.2f} pts",
            fontsize=5.0, color="#111", va="center")
    ax.text(0.5, 1.012, g["key"], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.8, fontweight="bold", color="#111")
    if g["sub"]:
        ax.text(0.5, 0.975, g["sub"], transform=ax.transAxes, ha="center",
                va="top", fontsize=5.0, color=GREY_M)
    ax.text(0.03, 0.035, g["d3"], transform=ax.transAxes, ha="left",
            va="bottom", fontsize=4.7, color=col)
    ax.set_xticks(X)
    ax.set_xticklabels(stages, fontsize=4.7)
    ax.set_xlim(-0.55, 6.9)
    ax.set_ylim(0.668, 0.798)
axL.set_ylabel("frame-level AUC", fontsize=5.6, labelpad=1)
axL.yaxis.set_label_coords(-0.22, 0.5)

# ---------------- Panel B ----------------
SIG = 0.95
axB.axhspan(-SIG, SIG, color=BLUE, alpha=0.06, zorder=0)
axB.axhline(0.0, color="#000", lw=0.7, zorder=2)
for sgn in (-1, 1):
    axB.axhline(sgn * SIG, color=GREY_M, lw=0.5, ls="--", zorder=2)
axB.text(0.015, SIG + 0.10, "pairing-noise band $\\pm$0.95 pts",
         fontsize=4.7, color="#555", va="bottom")

ptsB = [(-0.38, "h", 0), (-0.39, "h", 1), (1.27, "s", 2)]
for val, kind, i in ptsB:
    if kind == "s":
        axB.plot(i, val, "o", ms=3.75, mfc=BLUE, mec=BLUE, zorder=5)
        axB.text(i, val + 0.15, f"+{val:.2f}", ha="center", fontsize=6.2,
                 color=BLUE, fontweight="bold")
    else:
        axB.plot(i, val, "o", ms=2.75, mfc="white", mec="#666", mew=1.0,
                 zorder=5)
        axB.text(i, val - 0.17, f"{val:.2f}", ha="center", fontsize=5.4,
                 color="#333")
# SAM: 3-seed mean +0.28 pts (in-band), same mean statistic as the others
axB.plot(3, 0.28, "o", ms=2.75, mfc="white", mec="#666", mew=1.0, zorder=5)
axB.text(3, 0.58, "+0.28", ha="center", fontsize=5.0, color="#333")

for i, lab in [(0, "n=3"), (1, "n=3"), (2, "n=3"), (3, "n=3")]:
    axB.text(i, -1.32, lab, ha="center", fontsize=4.6, color=GREY_M)
axB.set_xticks([0, 1, 2, 3])
axB.set_xticklabels(["SPSA\nonly", "SWA\nonly", "SPSA+SWA", "SAM"],
                    fontsize=5.4)
# below-axis note: GaussMix and misplaced/overscaled variants are NOT on this
# axis (no paired CTRL on the same machine), so we state them in words per spec 4.2.
axB.text(0.5, -1.80,
         "GaussMix+SWA / misplaced, overscaled: off this axis, no paired CTRL",
         ha="center", va="center", fontsize=4.4, color=GREY_M)
axB.set_ylabel("gain vs matched CTRL\n(pts)", fontsize=5.4, labelpad=1)
axB.set_ylim(-2.05, 1.95)
axB.set_xlim(-0.5, 3.55)
axB.tick_params(axis="y", labelsize=5.0, pad=1)
for s in axB.spines.values():
    s.set_linewidth(0.6)

# panel letters + bottom footnote
fig.text(0.012, 0.972, "A", fontsize=8.5, fontweight="bold")
fig.text(0.012, 0.505, "B", fontsize=8.5, fontweight="bold")
fig.text(0.5, 0.006,
         "A: s2048, controlled 4-cycle, Celeb-DF-v2.  Without SPSA the deployed "
         "$\\bar\\theta$ sits 1.15 pts below its own 7-pt trajectory mean; with "
         "SPSA it rides the path (0.42 pts).  "
         "B: only the structured SPSA+SWA combination clears the $\\pm0.95$-pt "
         "pairing-noise band; SPSA alone, SWA alone, and SAM (mean $+0.28$, "
         "3 seeds) do not.",
         ha="center", va="bottom", fontsize=5.0, color="#333")

fig.savefig("fig_mech_board.pdf")
fig.savefig("fig_mech_board_preview.png", dpi=220, bbox_inches="tight")
print("saved fig_mech_board.pdf / fig_mech_board_preview.png")
