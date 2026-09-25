# -*- coding: utf-8 -*-
"""ICASSP fig.3: 3-seed-mean 7-point trajectories of the four controlled
configurations (CTRL / CTRL+SWA / +SPSA / SPSA+SWA). Frame-level AUC,
Celeb-DF, controlled 4-cycle protocol. 3-seed means recomputed pointwise;
cell means reproduce table II (0.7178/0.7139/0.7140/0.7305). Source
spsa1_ctrl_verified_data.md. Output: fig_controlled.pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stages = ["E0e", "E1m", "E1e", "E2m", "E2e", "E3m", "E3e"]
X = list(range(7))

conf = {
    "CTRL": {
        "seeds": [
            [0.6927, 0.7520, 0.6952, 0.7314, 0.6926, 0.7104, 0.7112],
            [0.7373, 0.7249, 0.6868, 0.7384, 0.6986, 0.7135, 0.7129],
            [0.7291, 0.7341, 0.7038, 0.7187, 0.7350, 0.7234, 0.7326],
        ],
        "style": dict(color="#8a8a8a", ls="--", lw=1.0, marker="o", ms=2.6),
    },
    "CTRL+SWA": {
        "seeds": [
            [0.7301, 0.7308, 0.6874, 0.7133, 0.6827, 0.7003, 0.7007],
            [0.7420, 0.6991, 0.7016, 0.7150, 0.7119, 0.7031, 0.6989],
            [0.7333, 0.7222, 0.7232, 0.7079, 0.7371, 0.7225, 0.7296],
        ],
        "style": dict(color="#d97706", ls=":", lw=1.2, marker="s", ms=2.8),
    },
    "+SPSA": {
        "seeds": [
            [0.7151, 0.7493, 0.7062, 0.7236, 0.7151, 0.7201, 0.7144],
            [0.7471, 0.7243, 0.7127, 0.6936, 0.6971, 0.6968, 0.6976],
            [0.6648, 0.6858, 0.7009, 0.7566, 0.7233, 0.7242, 0.7257],
        ],
        "style": dict(color="#237a57", ls="-.", lw=1.1, marker="^", ms=3.0),
    },
    "SPSA+SWA": {
        "seeds": [
            [0.7151, 0.7378, 0.7342, 0.7357, 0.7196, 0.7297, 0.7291],
            [0.7555, 0.7162, 0.7491, 0.7086, 0.7290, 0.7275, 0.7243],
            [0.6941, 0.7453, 0.7193, 0.7478, 0.7456, 0.7396, 0.7377],
        ],
        "style": dict(color="#1f5fa8", ls="-", lw=1.7, marker="o", ms=3.0),
    },
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
})

fig, ax = plt.subplots(figsize=(3.45, 2.15))

for name, c in conf.items():
    arr = c["seeds"]
    mean = [sum(p[i] for p in arr) / 3.0 for i in range(7)]
    st = c["style"]
    ax.plot(X, mean, color=st["color"], ls=st["ls"], lw=st["lw"],
            marker=st["marker"], ms=st["ms"], label=name, zorder=3 if name == "SPSA+SWA" else 2)

ax.annotate("joint mean 0.7305",
            xy=(2.5, 0.732), xytext=(1.35, 0.738),
            fontsize=6.6, color="#1f5fa8",
            arrowprops=dict(arrowstyle="-", color="#1f5fa8", lw=0.6),
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.1))

ax.set_xticks(X)
ax.set_xticklabels(stages, fontsize=6.5)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylim(0.675, 0.748)
ax.set_xlabel("checkpoint (controlled 4-cycle run: E0e = E0-end, Em = mid)", fontsize=7)
ax.set_ylabel("frame-level AUC (3-seed mean)", fontsize=8)
ax.grid(axis="y", color="0.9", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.8, loc="lower center", ncol=2, frameon=False,
          bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(pad=0.4)
fig.savefig("fig_controlled.pdf")
print("saved fig_controlled.pdf")
