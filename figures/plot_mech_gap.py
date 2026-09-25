# -*- coding: utf-8 -*-
"""ICASSP mechanism viz (candidate): the mean-fidelity gap and its repair.
Per-seed gap  d = AUC(SWA-Final thetabar) - AUC(seed's own 7-pt mean), in
points (x0.01), for the two SWA arms of the controlled ablation.
  CTRL+SWA: s1024 -1.07 / s2048 -1.15 / s4096 +0.86  ->  mean|d| = 1.03
  SPSA+SWA: s1024 -0.25 / s2048 -0.42 / s4096 +0.87  ->  mean|d| = 0.51
d=0 means the deployed weight average lands exactly at its own trajectory
level ("on the path it averages"). Source spsa1_ctrl_verified_data.md
sections VI / CTRL+SWA (per-seed 7-pt mean and SWA-Final rows).
Output: fig_mech_gap.pdf (+ png preview).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-seed signed gap d = thetabar - 7pt-mean  (in points)
groups = [
    {
        "name": "CTRL + SWA",
        "sub": "(no SPSA)",
        "seed_vals": [(-1.07, 1024), (-1.15, 2048), (+0.86, 4096)],
        "meanabs": 1.03,
        "note": "SWA's equal-weight average\nlands below the trajectory it averages",
    },
    {
        "name": "SPSA + SWA",
        "sub": "",
        "seed_vals": [(-0.25, 1024), (-0.42, 2048), (+0.87, 4096)],
        "meanabs": 0.51,
        "note": "average lands on the trajectory\n(fidelity restored)",
    },
]
seed_col = {1024: "#1f5fa8", 2048: "#d97706", 4096: "#237a57"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
})

fig, ax = plt.subplots(figsize=(6.3, 2.9))

# on-path reference
ax.axhline(0.0, color="#666", lw=1.0, ls="-", zorder=1)
ax.text(0.02, 0.06, "θ̄ = its own 7-pt trajectory mean   (on-path)",
        fontsize=7, color="#444", va="bottom")

for gi, g in enumerate(groups):
    xc = gi * 1.6
    # per-seed lollipops: vertical segment from path level (0) to the seed's gap
    for j, (d, seed) in enumerate(g["seed_vals"]):
        xj = xc + (j - 1) * 0.16
        col = seed_col[seed]
        ax.plot([xj, xj], [0.0, d], color=col, lw=1.0, alpha=0.85, zorder=2)
        ax.plot(xj, d, marker="o", ms=6.2, mfc=col, mec="white", mew=0.6, zorder=3)
        ax.text(xj, d + (0.10 if d >= 0 else -0.17), f"{d:+.2f}",
                ha="center", va="bottom" if d >= 0 else "top",
                fontsize=6.2, color=col)
    # mean-abs bracket + group header
    ax.text(xc, 1.24, g["name"], ha="center", va="bottom", fontsize=9,
            fontweight="bold")
    if g["sub"]:
        ax.text(xc, 1.08, g["sub"], ha="center", va="top", fontsize=6.5,
                color="#555")
    ax.annotate(f"mean |d| = {g['meanabs']:.2f}",
                xy=(xc, -1.42), ha="center", fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.25", fc="#f2f2f2", ec="#999",
                          lw=0.6))
    ax.text(xc, -1.78, g["note"], ha="center", va="top", fontsize=6.2,
            color="#555")

# negative region shading -> "off-path / below"
ax.axhspan(-1.45, -0.02, color="#1f5fa8", alpha=0.05, zorder=0)
ax.axhspan(0.02, 1.45, color="#237a57", alpha=0.05, zorder=0)
ax.set_xlim(-0.7, 2.3)
ax.set_ylim(-2.15, 1.42)
ax.set_xticks([])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_ylabel(r"gap $d=\mathrm{AUC}(\bar{\theta})-\mathrm{AUC}(\mathrm{7pt\ mean})$  (pts)",
              fontsize=7.5)
ax.tick_params(axis="y", labelsize=7)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.text(-0.62, 0.10, "off-path\n(below)", fontsize=6, color="#1f5fa8",
        ha="left", va="center", alpha=0.9)

# seed legend
for seed in (1024, 2048, 4096):
    ax.plot([], [], marker="o", ms=5, color=seed_col[seed], ls="none",
            label=f"s{seed}")
lg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=3,
               frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)

fig.tight_layout()
fig.savefig("fig_mech_gap.pdf", bbox_inches="tight")
fig.savefig("fig_mech_gap_preview.png", dpi=200, bbox_inches="tight")
print("saved fig_mech_gap.pdf / fig_mech_gap_preview.png")
