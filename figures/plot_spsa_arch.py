# -*- coding: utf-8 -*-
"""ICASSP fig.4: schematic of the SPSA (split-pool self-attention) module.
Output: fig_spsa_arch.pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7,
})

fig, ax = plt.subplots(figsize=(3.5, 1.75))
ax.set_xlim(-0.3, 12.0)
ax.set_ylim(0.55, 4.6)
ax.axis("off")


def box(cx, cy, w, h, text, fc="#eef2f8", ec="#33557a", lw=0.8, fs=6.4):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)


def arr(x1, y1, x2, y2, **kw):
    kw.setdefault("arrowstyle", "-|>")
    kw.setdefault("mutation_scale", 7)
    kw.setdefault("lw", 0.9)
    kw.setdefault("color", "#222")
    a = FancyArrowPatch((x1, y1), (x2, y2), zorder=1, **kw)
    ax.add_patch(a)


# main path (midline y = 2.6)
box(0.95, 2.6, 2.05, 0.85, "x\n(dconv_up2 out,\n128ch, 32×32)", fc="#f5f7fb")
box(3.35, 2.6, 1.55, 0.95, "1×1 conv\nchannel split", fc="#e8eef7")
# branches
box(5.9, 3.65, 2.6, 0.95, "multi-head\nself-attention\n(global)", fc="#e3f0e4", ec="#2f6b42")
box(5.9, 1.55, 2.6, 0.95, "PoolMix-style\nlocal pooling\n(per half-channel)", fc="#e3f0e4", ec="#2f6b42")
box(9.0, 2.6, 1.7, 0.95, "concat\n+ 1×1 fuse", fc="#e8eef7")
box(11.05, 2.6, 1.75, 0.85, "x′\n(128ch, 32×32)", fc="#f5f7fb")

# arrows: in->split
arr(1.98, 2.6, 2.55, 2.6)
# split -> branches
arr(3.62, 3.05, 4.62, 3.45)
arr(3.62, 2.15, 4.62, 1.85)
# branches -> fuse
arr(7.2, 3.45, 8.2, 2.85)
arr(7.2, 1.75, 8.2, 2.35)
# fuse -> out
arr(9.85, 2.6, 10.15, 2.6)

# residual shortcut (dashed, above)
a = FancyArrowPatch((1.95, 3.05), (11.0, 3.05), arrowstyle="-",
                    lw=0.8, ls=(0, (3, 2)), color="#7a7a7a", zorder=1)
ax.add_patch(a)
ax.plot([11.0], [3.05], marker=">", color="#7a7a7a", ms=4, zorder=1)
ax.text(6.4, 3.12, "residual shortcut", fontsize=6, ha="center", color="#7a7a7a")
ax.text(6.4, 4.32, "", fontsize=6)

# annotations
ax.text(3.35, 0.78, "inserted in the reconstruction decoder only—the\nshared-fingerprint path used for classification is untouched",
        ha="center", fontsize=5.8, color="#444")
ax.text(11.05, 1.15, "66.8K params\n(0.14% of model)",
        ha="center", fontsize=5.8, color="#444")

fig.tight_layout(pad=0.15)
fig.savefig("fig_spsa_arch.pdf")
print("saved fig_spsa_arch.pdf")
