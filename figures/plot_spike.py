# -*- coding: utf-8 -*-
"""ICASSP fig.1: best-checkpoint = isolated noise spike whose location the
seed decides. Data: SPSA+SWA, official six-round cosine protocol, fixed
period-index SWA window, seeds 1024/2048/4096 (12 test points each), source
docs/official_cos_verified_data.md. Output: fig_spike.pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seeds = {
    "s1024": [0.7129, 0.7101, 0.7333, 0.7708, 0.6842, 0.7037, 0.7411,
              0.7340, 0.7027, 0.7030, 0.7072, 0.7127],
    "s2048": [0.7471, 0.6698, 0.6878, 0.6739, 0.7183, 0.6961, 0.6667,
              0.7034, 0.7021, 0.7010, 0.7021, 0.6902],
    "s4096": [0.6885, 0.7111, 0.6967, 0.7152, 0.7039, 0.7193, 0.7236,
              0.7097, 0.7387, 0.7316, 0.7267, 0.7186],
}
best = {"s1024": (3, 0.7708), "s2048": (0, 0.7471), "s4096": (8, 0.7387)}
cols = {"s1024": "#1f5fa8", "s2048": "#d97706", "s4096": "#237a57"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
})

fig, ax = plt.subplots(figsize=(3.45, 2.3))
X = list(range(12))

for s, pts in seeds.items():
    ax.plot(X, pts, marker="o", ms=2.8, lw=1.1, color=cols[s],
            label=s, zorder=2)
    bi, bv = best[s]
    ax.plot(bi, bv, marker="o", ms=6.5, mfc="none", mec=cols[s],
            mew=1.5, zorder=3)
    if s == "s1024":
        ax.annotate("best 0.7708 (E2-mid)",
                    xy=(bi, bv), xytext=(4.7, 0.768),
                    fontsize=7, color=cols[s],
                    arrowprops=dict(arrowstyle="-", color=cols[s], lw=0.7),
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.1))
    elif s == "s2048":
        ax.annotate("best 0.7471 (E0-end)",
                    xy=(bi, bv), xytext=(1.2, 0.760),
                    fontsize=7, color=cols[s],
                    arrowprops=dict(arrowstyle="-", color=cols[s], lw=0.7),
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.1))
    else:
        ax.annotate("best 0.7387 (late)",
                    xy=(bi, bv), xytext=(8.2, 0.752),
                    fontsize=7, color=cols[s],
                    arrowprops=dict(arrowstyle="-", color=cols[s], lw=0.7),
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.1))

ax.set_xticks(X)
ax.set_xticklabels([str(i + 1) for i in X], fontsize=6.5)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylim(0.64, 0.79)
ax.set_xlabel("checkpoint index (official 6-round run)", fontsize=8)
ax.set_ylabel("frame-level AUC", fontsize=8)
ax.grid(axis="y", color="0.9", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=7, loc="lower center", ncol=3, frameon=False,
          bbox_to_anchor=(0.5, -0.06))

fig.tight_layout(pad=0.4)
fig.savefig("fig_spike.pdf")
print("saved fig_spike.pdf")
