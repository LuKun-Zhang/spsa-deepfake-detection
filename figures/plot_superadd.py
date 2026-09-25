# -*- coding: utf-8 -*-
"""ICASSP fig.2: interaction plot making the superadditive SPSA x SWA
interaction visible. 7-point-mean AUC, Celeb-DF frame-level, controlled
protocol, 3 seeds (values from spsa1_ctrl_verified_data.md Sec.7.2/VI).
Output: fig_superadd.pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# cell means (3-seed avg of 7-pt mean) and per-seed 7-pt means
no_swa = {  # x=0: no SPSA arm
    "off": (0.7178, [0.7122, 0.7160, 0.7253]),   # CTRL
    "on":  (0.7140, [0.7206, 0.7099, 0.7116]),   # SPSA alone
}
swa = {     # x=1: +SWA arm
    "off": (0.7139, [0.7065, 0.7102, 0.7251]),   # CTRL+SWA
    "on":  (0.7305, [0.7287, 0.7300, 0.7328]),   # SPSA+SWA (joint)
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
})

fig, ax = plt.subplots(figsize=(3.4, 2.0))
x0, x1 = 0.0, 1.0

# ---- draw ----
# x-axis = "SPSA module" on/off; the two curves = SWA off / SWA on.
#   curve SWA off : (x0=no SPSA)=CTRL 0.7178 -> (x1=+SPSA)=SPSA-alone 0.7140
#   curve SWA on  : (x0)=CTRL+SWA 0.7139   -> (x1)=SPSA+SWA 0.7305
# Correct pairing: x-axis = "SPSA module", the two curves = SWA off / SWA on.
#   curve SWA off : (x0=no SPSA)=CTRL 0.7178 -> (x1=+SPSA)=SPSA-alone 0.7140
#   curve SWA on  : (x0)=CTRL+SWA 0.7139   -> (x1)=SPSA+SWA 0.7305
curves = {
    "SWA off (CTRL / +SPSA)": (x0, no_swa["off"], x1, no_swa["on"]),
    "SWA on  (CTRL+SWA / +SPSA+SWA)": (x0, swa["off"], x1, swa["on"]),
}
styles = {
    "SWA off (CTRL / +SPSA)": dict(color="#1f5fa8", ls="--", lw=1.2),
    "SWA on  (CTRL+SWA / +SPSA+SWA)": dict(color="#d97706", ls="-", lw=1.6),
}
for name, (xa, cella, xb, cellb) in curves.items():
    st = styles[name]
    ax.plot([xa, xb], [cella[0], cellb[0]], marker="o", ms=4.2,
            color=st["color"], ls=st["ls"], lw=st["lw"], zorder=3,
            label=name)
    # per-seed scatter
    xs = [xa - 0.05, xa + 0.02, xa + 0.09]
    ys = cella[1]
    ax.scatter(xs, ys, s=9, color=st["color"], alpha=0.55, zorder=2, linewidths=0)
    xs = [xb - 0.05, xb + 0.02, xb + 0.09]
    ys = cellb[1]
    ax.scatter(xs, ys, s=9, color=st["color"], alpha=0.55, zorder=2, linewidths=0)

# value labels at the four cell means
ax.annotate("0.7178", xy=(x0, 0.7178), xytext=(x0, 0.7178 + 0.0009),
            fontsize=6.5, ha="center", color="#1f5fa8")
ax.annotate("0.7140", xy=(x1, 0.7140), xytext=(x1, 0.7140 - 0.0011),
            fontsize=6.5, ha="center", color="#1f5fa8")
ax.annotate("0.7139", xy=(x0, 0.7139), xytext=(x0, 0.7139 - 0.0013),
            fontsize=6.5, ha="center", color="#d97706")
ax.annotate("0.7305", xy=(x1, 0.7305), xytext=(x1, 0.7305 + 0.0009),
            fontsize=6.5, ha="center", color="#d97706",
            fontweight="bold")

# interaction bracket
ax.annotate("interaction $+2.04$\n(superadditive)", xy=(0.5, 0.7305),
            xytext=(0.44, 0.744), fontsize=7, ha="center", color="black",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))

ax.set_xticks([x0, x1])
ax.set_xticklabels(["no SPSA", "$+$SPSA"], fontsize=8)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(0.699, 0.746)
ax.set_ylabel("7-pt mean AUC", fontsize=8)
ax.set_xlabel("SPSA module (single switch)", fontsize=8)
ax.grid(axis="y", color="0.9", lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.6, loc="upper left", frameon=False)

fig.tight_layout(pad=0.4)
fig.savefig("fig_superadd.pdf")
print("saved fig_superadd.pdf")
