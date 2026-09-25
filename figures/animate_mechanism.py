# -*- coding: utf-8 -*-
"""Animated mechanism figure for the repository README.

Unlike the other scripts in this directory, this one **reads the shipped logs**
rather than embedding literals, so the animation cannot silently drift from the
record it is drawn from.  The parse is the same one `scripts/verify_mechanism.py`
uses; if the two ever disagree, the log is the original record.

Source:  logs/mechanism_ucf/<run>/training.log    six in-window probe AUCs
         logs/mechanism_ucf/<run>/swa_eval.jsonl  AUC of theta_bar, bn = true

What it draws, and why:

    The window is the last learning-rate cycle, steps 17232..22976, probed six
    times.  Each panel plots the six probes as AUC *relative to the window's own
    mean*, so the zero line is "the mean" and every panel shares one y-axis.  A
    shared axis is the point: the claim is about amplitude, so the amplitudes
    have to be drawn to the same scale.

    The animation reveals the six probes one at a time, then drops in two
    markers -- the endpoint theta_end (black) and the SWA average theta_bar
    (orange), joined by a bar whose length is Delta.  The whole mechanism then
    reads as one gesture:

        SPSA off -> at the high rates the endpoint sits ABOVE the mean, so
                    averaging pulls the model down.  Bonus < Dev, Delta < 0.
        SPSA on  -> the amplitude is damped, the endpoint sits BELOW the mean,
                    so averaging pulls it up.  Bonus > Dev, Delta > 0 always.

    Both markers come from the logs: theta_end is the last probe (step 22967)
    and theta_bar is the w = 1.0 row of swa_eval.jsonl.

`tail_lam` (the rate label) is the learning-rate clamp inside the window, as a
fraction of lr_max = 2e-4; lam = 0 leaves the schedule on its own lr = 1e-6, so
the four rungs are 1x / 40x / 100x / 200x of the natural rate.

Outputs:
    figures/mechanism_animation.svg   animated (SMIL), 7.2 s loop, vector
    figures/mechanism_animation.gif   same animation, raster fallback

Run:  python figures/animate_mechanism.py
"""
import io
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs', 'mechanism_ucf')
OUT_SVG = os.path.join(ROOT, 'figures', 'mechanism_animation.svg')
OUT_GIF = os.path.join(ROOT, 'figures', 'mechanism_animation.gif')

STEPS = [18182, 19139, 20096, 21053, 22010, 22967]
RATE = {'0.0': '1x', '0.2': '40x', '0.5': '100x', '1.0': '200x',
        '0.00': '1x', '0.20': '40x', '0.50': '100x', '1.00': '200x'}
LR = {'1x': 'lr 1e-6', '40x': 'lr 8e-5', '100x': 'lr 2e-4', '200x': 'lr 2e-4'}

PT = re.compile(r'dataset: avg\s+step:\s*(\d+)\s+testing-metric, acc: [0-9.]+\s+'
                r'testing-metric, auc: ([0-9.]+)')

# the values verify_mechanism.py asserts against the paper
EXP_SPREAD = {'CTRL': [0.79, 7.22, 5.07, 8.69], 'SPSA': [1.02, 2.66, 3.20, 4.28]}
EXP_DEV = {'CTRL': [-0.27, 0.10, 2.53, 3.21]}

C_CTRL = '#b23a34'
C_SPSA = '#1f5fa8'
C_END = '#1a1a1a'
C_BAR = '#d97706'
C_GRID = '#dfe3e8'
C_TXT = '#20242b'
C_DIM = '#7a828e'

RATES = ('1x', '40x', '100x', '200x')
ROWS = ('CTRL', 'SPSA')
ROWLBL = {'CTRL': 'SPSA off', 'SPSA': 'SPSA on'}


# ---------------------------------------------------------------- read logs
def pts(text):
    return {int(m.group(1)): float(m.group(2)) for m in PT.finditer(text)}


def field(text, key):
    m = re.search(r'(?m)^' + key + r':\s*(.*)$', text)
    return m.group(1).strip() if m else '?'


def theta_bar(run):
    """AUC of theta_bar at w = 1, BN-recalibrated -- the Section 3.3 column."""
    p = os.path.join(LOGS, run, 'swa_eval.jsonl')
    for line in io.open(p, encoding='utf-8'):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if abs(j.get('w', -1) - 1.0) < 1e-9 and bool(j.get('bn')) is True:
            return j['metrics']['auc']
    raise SystemExit('no bn=true w=1 row in ' + p)


def load():
    legs = {}
    for run in sorted(os.listdir(LOGS)):
        lp = os.path.join(LOGS, run, 'training.log')
        if not os.path.isfile(lp):
            continue
        text = io.open(lp, encoding='utf-8', errors='replace').read()
        P = pts(text)
        if not all(s in P for s in STEPS):
            continue                      # not one of the eight probe legs
        arm = 'SPSA' if field(text, 'spsa') == 'True' else 'CTRL'
        rate = RATE.get(field(text, 'tail_lam'))
        if rate is None:
            continue
        six = [P[s] for s in STEPS]
        mean6 = sum(six) / 6.0
        legs.setdefault(arm, {})[rate] = dict(
            run=run,
            rel=[(v - mean6) * 100 for v in six],
            mean6=mean6,
            spread=(max(six) - min(six)) * 100,
            dev=(six[-1] - mean6) * 100,
            bonus=(theta_bar(run) - mean6) * 100,
        )
    return legs


def check(legs):
    """Refuse to draw anything that disagrees with the paper."""
    bad = []
    for arm, exp in (('CTRL', EXP_SPREAD['CTRL']), ('SPSA', EXP_SPREAD['SPSA'])):
        for i, rate in enumerate(RATES):
            got = legs[arm][rate]['spread']
            if abs(got - exp[i]) > 0.006:
                bad.append('%s %s spread: paper %.2f, logs %.2f'
                           % (arm, rate, exp[i], got))
    for i, rate in enumerate(RATES):
        got = legs['CTRL'][rate]['dev']
        if abs(got - EXP_DEV['CTRL'][i]) > 0.006:
            bad.append('CTRL %s Dev: paper %.2f, logs %.2f'
                       % (rate, EXP_DEV['CTRL'][i], got))
    for rate in RATES:
        if legs['SPSA'][rate]['dev'] > 0.01:
            bad.append('SPSA %s Dev must be <= 0, got %.2f'
                       % (rate, legs['SPSA'][rate]['dev']))
        for arm in ROWS:
            if legs[arm][rate]['bonus'] <= 0:
                bad.append('%s %s Bonus must be positive, got %.2f'
                           % (arm, rate, legs[arm][rate]['bonus']))
    if bad:
        for b in bad:
            print('  !! ' + b)
        raise SystemExit('animation refused: the logs disagree with the paper')
    print('all 8 legs match the paper; drawing.')


def ymax_for(legs):
    y = 3.0
    for arm in ROWS:
        for rate in RATES:
            y = max(y, max(abs(v) for v in legs[arm][rate]['rel']))
    return math.ceil(y * 1.10 * 2) / 2.0


# ---------------------------------------------------------------- svg build
W, H = 900, 540
GAP, VGAP = 16, 34
LEFT, RIGHT, TOPM, BOTM = 96, 14, 104, 62
COLW = (W - LEFT - RIGHT - 3 * GAP) / 4.0        # 185.5
ROWH = (H - TOPM - BOTM - VGAP) / 2.0            # 170
PADL, PADR, PADT, PADB = 30, 34, 8, 24
PLOTW = COLW - PADL - PADR
PLOTH = ROWH - PADT - PADB


def colx(i):
    return LEFT + i * (COLW + GAP)


def rowy(j):
    return TOPM + j * (ROWH + VGAP)


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Frame(object):
    def __init__(self, ymax, legs):
        self.o, self.ymax, self.legs = [], ymax, legs

    def add(self, s):
        self.o.append(s)

    def px(self, ci, k):
        return colx(ci) + PADL + k / 5.0 * PLOTW

    def py(self, rj, v):
        return rowy(rj) + PADT + PLOTH / 2.0 - v / (2.0 * self.ymax) * PLOTH

    def chrome(self):
        self.add('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
        self.add('<text x="%d" y="22" class="ti">Why averaging can win: SPSA '
                 'damps the window amplitude</text>' % (LEFT - 62))
        self.add('<text x="%d" y="38" class="su">Six test points inside the '
                 'final learning-rate cycle &#183; y is AUC &#8722; the window'
                 '&#8217;s own mean, in points &#183; one shared y-axis, so the '
                 'amplitudes are drawn to the same scale</text>' % (LEFT - 62))
        for i, rate in enumerate(RATES):
            cx = colx(i) + PADL + PLOTW / 2.0
            self.add('<text x="%.1f" y="58" class="ch">%s</text>' % (cx, rate))
            self.add('<text x="%.1f" y="70" class="cs">%s</text>'
                     % (cx, LR[rate]))
        for j, arm in enumerate(ROWS):
            cy = rowy(j) + ROWH / 2.0
            self.add('<text x="30" y="%.1f" class="rl" fill="%s" '
                     'transform="rotate(-90 30 %.1f)">%s</text>'
                     % (cy, C_CTRL if arm == 'CTRL' else C_SPSA, cy,
                        ROWLBL[arm]))
        for j in range(2):
            for i in range(4):
                x0, y0 = colx(i) + PADL, rowy(j) + PADT
                x1, y1 = colx(i) + COLW - PADR, rowy(j) + ROWH - PADB
                self.add('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                         'fill="#fcfdfe" stroke="%s" stroke-width="0.8"/>'
                         % (x0, y0, x1 - x0, y1 - y0, C_GRID))
                self.add('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                         'stroke="%s" stroke-width="0.8" stroke-dasharray='
                         '"3 3"/>' % (x0, self.py(j, 0), x1, self.py(j, 0),
                                      C_DIM))
                if i == 0:
                    for v in (-self.ymax, 0.0, self.ymax):
                        self.add('<text x="%.1f" y="%.1f" class="tk" '
                                 'text-anchor="end">%+.0f</text>'
                                 % (x0 - 5, self.py(j, v) + 3.2, v))

    def panel(self, j, i, stage):
        d = self.legs[ROWS[j]][RATES[i]]
        col = C_CTRL if ROWS[j] == 'CTRL' else C_SPSA
        n = min(stage + 1, 6)
        xy = [(self.px(i, k), self.py(j, d['rel'][k])) for k in range(n)]
        if n > 1:
            self.add('<polyline fill="none" stroke="%s" stroke-width="1.9" '
                     'stroke-linejoin="round" points="%s"/>'
                     % (col, ' '.join('%.1f,%.1f' % p for p in xy)))
        for k, (x, y) in enumerate(xy):
            self.add('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s"/>'
                     % (x, y, 3.4 if k == n - 1 else 2.6, col))
        if j == 1:
            for k in range(6):
                self.add('<text x="%.1f" y="%.1f" class="tk" '
                         'text-anchor="middle">%d</text>'
                         % (self.px(i, k), rowy(j) + ROWH - PADB + 13, k + 1))
        if stage >= 6:
            ex, ey = xy[-1]
            self.add('<circle cx="%.1f" cy="%.1f" r="5.0" fill="%s" '
                     'stroke="#ffffff" stroke-width="1.6"/>' % (ex, ey, C_END))
        if stage >= 7:
            bx, by = self.px(i, 5) + 14, self.py(j, d['bonus'])
            self.add('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                     'stroke="%s" stroke-width="2.4" stroke-linecap="round"/>'
                     % (ex, ey, ex, by, C_BAR))
            self.add('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                     'stroke="%s" stroke-width="1.1" stroke-dasharray="2 2"/>'
                     % (ex, by, bx, by, C_BAR))
            self.add('<circle cx="%.1f" cy="%.1f" r="5.0" fill="#ffffff" '
                     'stroke="%s" stroke-width="2.0"/>' % (bx, by, C_BAR))
        cx = colx(i) + PADL + PLOTW / 2.0
        top = rowy(j)
        if stage >= 8:
            self.add('<text x="%.1f" y="%.1f" class="hs">spread %.2f '
                     '&#8212; averaging %s</text>'
                     % (cx, top - 21, d['spread'],
                        'helps' if d['bonus'] > d['dev'] else 'hurts'))
        else:
            self.add('<text x="%.1f" y="%.1f" class="hs">spread &#8212;'
                     '</text>' % (cx, top - 21))
        self.add('<text x="%.1f" y="%.1f" class="hm" text-anchor="middle">'
                 '<tspan fill="%s">&#9679; %+0.2f</tspan>'
                 '<tspan fill="%s" dx="12">&#9675; %+0.2f</tspan></text>'
                 % (cx, top - 8, C_END, d['dev'], C_BAR, d['bonus']))

    def legend(self):
        y = H - 46
        self.add('<text x="%.1f" y="%d" class="su2">'
                 '&#9679; &#952;end, the endpoint&#160;&#160;&#183;&#160;&#160;'
                 '&#9675; &#952;&#772;, the SWA average&#160;&#160;&#183;'
                 '&#160;&#160;the bar between them is &#916; = Bonus &#8722; '
                 'Dev&#160;&#160;&#183;&#160;&#160;dashed line = the window'
                 '&#8217;s own mean</text>' % (colx(0) + PADL, y))
        self.add('<text x="%.1f" y="%d" class="su2">probes 1&#8211;6 are test '
                 'points at steps 18182, 19139, 20096, 21053, 22010, 22967 '
                 '(&#8776;957 apart)</text>' % (colx(0) + PADL, y + 15))
        self.add('<text x="%.1f" y="%d" class="su2">&#952;&#772; read from '
                 'swa_eval.jsonl with the BatchNorm buffers re-estimated: the '
                 'Section 3.3 convention</text>' % (colx(0) + PADL, y + 30))

    def render(self, stage):
        self.chrome()
        for j in range(2):
            for i in range(4):
                self.panel(j, i, stage)
        self.legend()
        return '\n'.join(self.o)


CSS = ('<style>'
       'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,'
       'Helvetica,Arial,sans-serif}'
       '.ti{font-weight:700;font-size:15px;fill:%s}'
       '.su{font-weight:400;font-size:10.5px;fill:%s}'
       '.ch{font-weight:700;font-size:14px;fill:%s;text-anchor:middle}'
       '.cs{font-weight:400;font-size:9.5px;fill:%s;text-anchor:middle;'
       'font-family:ui-monospace,Menlo,Consolas,monospace}'
       '.rl{font-weight:700;font-size:12px;text-anchor:middle}'
       '.tk{font-weight:400;font-size:9.5px;fill:%s;'
       'font-family:ui-monospace,Menlo,Consolas,monospace}'
       '.hs{font-weight:400;font-size:9px;fill:%s;text-anchor:middle}'
       '.hm{font-weight:700;font-size:9.5px;'
       'font-family:ui-monospace,Menlo,Consolas,monospace}'
       '.su2{font-weight:400;font-size:10.5px;fill:%s}'
       '</style>' % (C_TXT, C_DIM, C_TXT, C_DIM, C_DIM, C_DIM, C_DIM))


def build(legs, ymax):
    CYC = 7.2
    slots = [0.55] * 6 + [0.90, 0.90, 2.10]      # 3.3 + 1.8 + 2.1 = 7.2
    bounds, t = [], 0.0
    for dur in slots:
        bounds.append((t, t + dur))
        t += dur

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
             'width="%d" height="%d" role="img" aria-label="Animated Section '
             '3.3 mechanism window: SPSA damps the amplitude">'
             % (W, H, W, H), CSS]
    for k, (a, b) in enumerate(bounds):
        if k == 0:
            vals, keys = '1;0', '0;%.6f' % (b / CYC)
        else:
            vals, keys = '0;1;0', '0;%.6f;%.6f' % (a / CYC, b / CYC)
        parts.append('<g>')
        parts.append('<animate attributeName="opacity" calcMode="discrete" '
                     'values="%s" keyTimes="%s" dur="%.1fs" '
                     'repeatCount="indefinite"/>' % (vals, keys, CYC))
        parts.append(Frame(ymax, legs).render(k))
        parts.append('</g>')
    parts.append('<!-- drawn from logs/mechanism_ucf/*/{training.log,'
                 'swa_eval.jsonl} by figures/animate_mechanism.py -->')
    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------- gif build
def build_gif(legs, ymax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 6.05), dpi=100)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.830, bottom=0.170,
                        wspace=0.30, hspace=0.68)

    def draw(stage):
        for j in range(2):
            for i in range(4):
                ax = axes[j][i]
                ax.clear()
                arm, rate = ROWS[j], RATES[i]
                d = legs[arm][rate]
                col = C_CTRL if arm == 'CTRL' else C_SPSA
                n = min(stage + 1, 6)
                xs = list(range(n))
                ax.axhline(0, color=C_DIM, lw=0.9, ls=(0, (3, 3)), zorder=1)
                ax.plot(xs, d['rel'][:n], '-', color=col, lw=1.9, zorder=2)
                ax.plot(xs[:-1], d['rel'][:n - 1], 'o', color=col, ms=3.0,
                        zorder=3)
                ax.plot(xs[-1:], d['rel'][n - 1:n], 'o', color=col, ms=4.6,
                        zorder=3)
                if stage >= 6:
                    ax.plot([5], [d['dev']], 'o', color=C_END, ms=6.5,
                            mec='white', mew=1.4, zorder=5)
                if stage >= 7:
                    ax.plot([5, 5], [d['dev'], d['bonus']], '-', color=C_BAR,
                            lw=2.4, zorder=4)
                    ax.plot([5, 5.42], [d['bonus'], d['bonus']], '--',
                            color=C_BAR, lw=1.1, zorder=4)
                    ax.plot([5.42], [d['bonus']], 'o', color=C_BAR, ms=6.5,
                            mfc='white', mew=1.7, zorder=5)
                ax.set_title(u'● %+.2f   ○ %+.2f\nspread %s'
                             % (d['dev'], d['bonus'],
                                ('%.2f — averaging %s'
                                 % (d['spread'], 'helps'
                                    if d['bonus'] > d['dev'] else 'hurts'))
                                if stage >= 8 else '—'),
                             fontsize=7.8, color=C_TXT, pad=6, linespacing=1.35)
                ax.set_xlim(-0.5, 5.85)
                ax.set_ylim(-ymax, ymax)
                ax.set_xticks(range(6))
                ax.set_xticklabels([str(k + 1) for k in range(6)],
                                   fontsize=6.2)
                ax.tick_params(axis='y', labelsize=6.2)
                if j == 1:
                    ax.set_ylabel(u'AUC − mean (pt)', fontsize=6.8)
                else:
                    ax.set_xticklabels([])
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
                ax.grid(axis='y', color='0.92', lw=0.6, zorder=0)
        for i, rate in enumerate(RATES):
            axes[0][i].text(0.5, 1.30, rate, transform=axes[0][i].transAxes,
                            ha='center', fontsize=12.5, fontweight='bold',
                            color=C_TXT)
        for j in range(2):
            axes[j][0].text(-0.30, 0.5, ROWLBL[ROWS[j]],
                            transform=axes[j][0].transAxes, rotation=90,
                            va='center', ha='center', fontsize=10.5,
                            fontweight='bold',
                            color=C_CTRL if j == 0 else C_SPSA)
        fig.suptitle('Why averaging can win: SPSA damps the window amplitude',
                     fontsize=13.5, fontweight='bold', color=C_TXT, y=0.982)
        fig.text(0.075, 0.105, u'● θend, the endpoint    '
                 u'○ θ̄, the SWA average    '
                 u'bar = Δ = Bonus − Dev    '
                 u'dashed line = the window’s own mean',
                 fontsize=8, color=C_DIM)
        fig.text(0.075, 0.073, u'rate: 1× = lr 1e-6, 40× = 8e-5, 100× = 2e-4, '
                 u'200× = 2e-4 unclamped    probes 1–6 at steps 18182, 19139, '
                 u'20096, 21053, 22010, 22967 (≈957 apart)',
                 fontsize=7.6, color=C_DIM)
        fig.text(0.075, 0.041, u'θ̄ read from swa_eval.jsonl with the '
                 u'BatchNorm buffers re-estimated: the Section 3.3 convention',
                 fontsize=7.6, color=C_DIM)
        return []

    order = list(range(9)) + [8, 8]          # hold the verdict longer
    ani = FuncAnimation(fig, draw, frames=order, blit=False)
    ani.save(OUT_GIF, writer=PillowWriter(fps=1000.0 / 660.0))
    plt.close(fig)


def main():
    legs = load()
    check(legs)
    ymax = ymax_for(legs)
    svg = build(legs, ymax)
    io.open(OUT_SVG, 'w', encoding='utf-8', newline='\n').write(svg)
    print('wrote %s  (%.1f KB, y = +/-%.1f pt)'
          % (OUT_SVG, len(svg.encode('utf-8')) / 1024.0, ymax))
    build_gif(legs, ymax)
    print('wrote %s  (%.1f KB)' % (OUT_GIF,
                                   os.path.getsize(OUT_GIF) / 1024.0))
    for arm in ROWS:
        for rate in RATES:
            d = legs[arm][rate]
            print('  %-5s %-5s spread %5.2f  Dev %+6.2f  Bonus %+6.2f  '
                  'Delta %+6.2f' % (arm, rate, d['spread'], d['dev'],
                                    d['bonus'], d['bonus'] - d['dev']))


if __name__ == '__main__':
    main()
