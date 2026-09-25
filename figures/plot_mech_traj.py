# -*- coding: utf-8 -*-
"""ICASSP mechanism viz v2 (candidate): the fidelity-gap story drawn on a real
trajectory, so the 'deployed average falls off the path' is visible directly.
Shown seed = s2048 (controlled protocol, Celeb-DF frame AUC), the cleanest
illustration of the collapse. Per-seed numbers for all three seeds (incl. the
flat s4096) go in the caption / table, not fabricated here.
  CTRL+SWA s2048: 7-pt = [0.7420,0.6991,0.7016,0.7150,0.7119,0.7031,0.6989]
                  mean 0.7102, SWA-Final thetabar 0.6987  ->  d = -1.15
  SPSA+SWA s2048: 7-pt = [0.7555,0.7162,0.7491,0.7086,0.7290,0.7275,0.7243]
                  mean 0.7300, SWA-Final thetabar 0.7258  ->  d = -0.42
Source spsa1_ctrl_verified_data.md (s2048 CTRL+SWA / SPSA1+SWA rows).
Output: fig_mech_traj.pdf (+ preview png).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stages = ["E0e", "E1m", "E1e", "E2m", "E2e", "E3m", "E3e"]
X = list(range(7))

panels = [
    {
        "key": "CTRL + SWA",
        "sub": "no SPSA",
        "pts": [0.7420, 0.6991, 0.7016, 0.7150, 0.7119, 0.7031, 0.6989],
        "mean": 0.7102,
        "thetabar": 0.6987,
        "color": "#d97706",
    },
    {
        "key": "SPSA + SWA",
        "sub": "",
        "pts": [0.7555, 0.7162, 0.7491, 0.7086, 0.7290, 0.7275, 0.7243],
        "mean": 0.7300,
        "thetabar": 0.7258,
        "color": "#1f5fa8",
    },
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)

for ax, g in zip(axes, panels):
    col = g["color"]
    # SWA collection window = final cycle E3 (spans E2e -> E3e test points)
    ax.axvspan(3.5, 6.0, color="#dddddd", alpha=0.5, zorder=0)
    ax.text(4.75, 0.769, "SWA collects\n(last cycle)", ha="center",
            va="top", fontsize=5.8, color="#666", linespacing=1.1)
    # trajectory (the path being averaged)
    ax.plot(X, g["pts"], color=col, lw=1.5, marker="o", ms=3.4,
            zorder=4, label="trajectory")
    # 7-pt trajectory mean (expected level of the path)
    ax.axhline(g["mean"], color="#444", lw=0.9, ls=":", zorder=3)
    ax.text(6.15, g["mean"], " 7-pt mean", fontsize=6.4, color="#333",
            va="center", ha="left")
    # deployed average thetabar
    ax.axhline(g["thetabar"], color=col, lw=1.6, ls="--", zorder=5)
    ax.text(6.15, g["thetabar"], r" $\bar{\theta}$ (deployed)", fontsize=6.4,
            color=col, va="center", ha="left")
    # gap between deployed average and trajectory mean (per-seed d, points)
    d = (g["thetabar"] - g["mean"]) * 100.0
    ax.annotate("", xy=(5.55, g["mean"]), xytext=(5.55, g["thetabar"]),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=0.9))
    ax.text(5.62, (g["thetabar"] + g["mean"]) / 2, f"{d:+.2f} pts",
            fontsize=6.4, color="#111", va="center")
    # header
    ax.set_title(g["key"], fontsize=8.6, fontweight="bold", pad=3)
    if g["sub"]:
        ax.text(0.5, 0.965, g["sub"], transform=ax.transAxes, ha="center",
                fontsize=6.4, color="#666")
    ax.set_xticks(X)
    ax.set_xticklabels(stages, fontsize=6.2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(-0.4, 7.0)
    ax.set_ylim(0.652, 0.782)

axes[0].set_ylabel("frame-level AUC", fontsize=7.5)
fig.text(0.5, 0.015,
         "s2048, controlled 4-cycle protocol, Celeb-DF-v2.  "
         "Equal-weight average $\\bar{\\theta}$ lands 1.15 pts below its own "
         "trajectory mean without SPSA, and 0.42 pts with SPSA.",
         ha="center", fontsize=6.8, color="#333")

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("fig_mech_traj.pdf", bbox_inches="tight")
fig.savefig("fig_mech_traj_preview.png", dpi=200, bbox_inches="tight")
print("saved fig_mech_traj.pdf / fig_mech_traj_preview.png")
