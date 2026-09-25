# -*- coding: utf-8 -*-
"""
只画 Fig.3 的 A 行两个小图（CTRL / SPSA 的 BONUS-DEV 剂量-响应）。
不带 B 行。渲染同款样式。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'pdf.fonttype': 42,
})

ORANGE = '#d97706'
BLUE   = '#1f5fa8'
GRAY   = '#444444'

xs   = [0, 1, 2, 3]
xlab = ['$1\\times$', '$40\\times$', '$100\\times$', '$200\\times$']

CTRL_B = [0.08, 0.67, 1.43, 2.18]
CTRL_D = [-0.27, 0.10, 2.53, 3.21]
SPSA_B = [0.15, 0.32, 0.70, 0.85]
SPSA_D = [-0.07, -1.33, -0.69, -0.15]

fig = plt.figure(figsize=(3.4, 1.40))
axA1 = fig.add_axes([0.105, 0.245, 0.385, 0.635])
axA2 = fig.add_axes([0.545, 0.245, 0.385, 0.635])

YLO, YHI = -2.00, 3.85
xf = np.linspace(0, 3, 400)


def panel(ax, B, D, col, title):
    Bf = np.interp(xf, xs, B)
    Df = np.interp(xf, xs, D)

    ax.axhline(0, color=GRAY, lw=0.7, zorder=1)

    neg = Bf < Df
    ax.fill_between(xf, Bf, Df, where=(~neg), color=col, alpha=0.16, lw=0, zorder=1)
    if neg.any():
        ax.fill_between(xf, Bf, Df, where=neg, color=col, alpha=0.50, lw=0, zorder=1)

    ax.plot(xs, B, lw=1.2, color=col, marker='o', ms=2.5,
            mfc=col, mec=col, zorder=3, label='BONUS')
    ax.plot(xs, D, lw=1.2, color=col, marker='s', ms=2.3, mfc='white',
            mec=col, mew=0.7, ls='--', zorder=3, label='DEV')

    s = np.sign(Bf - Df)
    cross = np.where(np.diff(s) != 0)[0]
    if cross.size:
        xc = xf[cross[0]]
        ax.axvline(xc, color=col, ls=':', lw=0.9, zorder=2)
        ax.text(xc - 0.08, YLO + 0.30, r'$\Delta<0$', color=col,
                fontsize=4.8, ha='right', va='bottom', zorder=4)
    else:
        ax.text(0.10, 1.35, r'$\Delta>0$', color=col,
                fontsize=4.8, ha='left', va='bottom', zorder=4)

    ax.set_xlim(-0.42, 3.42)
    ax.set_ylim(YLO, YHI)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlab, fontsize=5.0)
    ax.tick_params(axis='both', labelsize=5.0, length=2.0, width=0.6)
    ax.set_title(title, fontsize=5.6, pad=2.2, color=col)
    ax.legend(fontsize=4.6, loc='upper left', frameon=False,
              handlelength=1.5, borderpad=0.1, labelspacing=0.25,
              handletextpad=0.35)


panel(axA1, CTRL_B, CTRL_D, ORANGE, 'CTRL + SWA  (no SPSA)')
panel(axA2, SPSA_B, SPSA_D, BLUE, 'SPSA + SWA  (module ON)')

axA1.set_ylabel('pts vs. window mean', fontsize=5.2)
axA2.set_yticklabels([])

fig.text(0.012, 0.975, 'A', fontsize=8.5, fontweight='bold', color='black', va='top')

fig.savefig(r'D:\desktop\ICASSP2027_kit\paper\fig_mech_two_panels.pdf')
fig.savefig(r'D:\desktop\ICASSP2027_kit\paper\fig_mech_two_panels_preview.png', dpi=300)
print('done')
