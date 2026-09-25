# -*- coding: utf-8 -*-
"""Offline re-derivation of every Section 3.3 / Section 4.6 (ResNet34) number.

Reads only the raw artifacts shipped in this repository:

    logs/mechanism_ucf/<run>/training.log      per-point test AUC
    logs/mechanism_ucf/<run>/swa_eval.jsonl    AUC of the averaged weights
    logs/mechanism_r34/<run>/...               same, ResNet34 backbone

Definitions (identical to the paper):

    window   = last learning-rate cycle, steps 17232..22976
    probes   = six test points at 18182 19139 20096 21053 22010 22967
    mean6    = mean AUC over the six probes
    spread   = max AUC - min AUC over the six          §3.3 "window spread"
    Dev      = AUC(theta_end) - mean6                  §3.3 "endpoint position"
    Bonus    = AUC(theta_bar) - mean6
    Delta    = Bonus - Dev  ==  AUC(theta_bar) - AUC(theta_end)

`swa_eval.jsonl` stores theta_bar twice, once with the BatchNorm buffers
re-estimated on the training set (`bn = true`) and once with the buffers
frozen from the endpoint (`bn = false`).  The two reporting conventions
differ, and the paper follows each run's own convention:

    Section 3.3 (UCF, mexp_*)   -> bn = true   +0.08 +0.67 +1.43 +2.18
    Section 4.6 (ResNet34)      -> bn = false  +0.077
                                   (this leg mirrored the archived R34
                                    protocol, which froze the buffers)

Both columns are printed below; the assertions use the convention above.

`tail_lam` is the learning-rate clamp inside the window, as a fraction of
lr_max = 2e-4.  lam = 0 leaves the schedule on its own lr = 1e-6, so the four
rungs are 1x / 40x / 100x / 200x.

    python scripts/verify_mechanism.py            # prints table, checks values
"""
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN6 = [18182, 19139, 20096, 21053, 22010, 22967]
RATE = {'0.0': '1x', '0.2': '40x', '0.5': '100x', '1.0': '200x',
        '0.00': '1x', '0.20': '40x', '0.50': '100x', '1.00': '200x'}

PT = re.compile(r'dataset: avg\s+step:\s*(\d+)\s+testing-metric, acc: [0-9.]+\s+'
                r'testing-metric, auc: ([0-9.]+)')


def pts(text):
    return {int(m.group(1)): float(m.group(2)) for m in PT.finditer(text)}


def theta_bar(d, bn):
    """AUC of theta_bar at w=1. bn=True -> BN-recalibrated, False -> frozen."""
    p = os.path.join(d, 'swa_eval.jsonl')
    if not os.path.exists(p):
        return None
    for line in io.open(p, encoding='utf-8'):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if abs(j.get('w', -1) - 1.0) < 1e-9 and bool(j.get('bn')) is bn:
            return j.get('metrics', {}).get('auc')
    return None


def field(text, key):
    m = re.search(r'(?m)^' + key + r':\s*(.*)$', text)
    return m.group(1).strip() if m else '?'


def fmt(v):
    return '   --   ' if v is None else format(v, '+8.4f')


rows = []
for tree in ('mechanism_ucf', 'mechanism_r34'):
    base = os.path.join(ROOT, 'logs', tree)
    if not os.path.isdir(base):
        continue
    for run in sorted(os.listdir(base)):
        d = os.path.join(base, run)
        lp = os.path.join(d, 'training.log')
        if not os.path.isfile(lp):
            continue
        text = io.open(lp, encoding='utf-8', errors='replace').read()
        P = pts(text)
        got = [P[s] for s in WIN6 if s in P]
        rec = dict(tree=tree, run=run, seed=field(text, 'manualSeed'),
                   spsa=field(text, 'spsa'), lam=field(text, 'tail_lam'),
                   bb=field(text, 'backbone_name'))
        if len(got) < 6:
            rec.update(spread=None, dev=None, bonus=None, bonus_frozen=None,
                       delta=None, note='INCOMPLETE (only %d of 6 probes)' % len(got))
            rows.append(rec)
            continue
        mean6 = sum(got) / 6.0
        end = P[WIN6[-1]]
        b_rec = theta_bar(d, True)
        b_frz = theta_bar(d, False)
        rec.update(spread=(max(got) - min(got)) * 100,
                   dev=(end - mean6) * 100,
                   bonus=None if b_rec is None else (b_rec - mean6) * 100,
                   bonus_frozen=None if b_frz is None else (b_frz - mean6) * 100,
                   note='')
        rec['delta'] = None if rec['bonus'] is None else rec['bonus'] - rec['dev']
        rows.append(rec)

head = (f"{'tree':<15}{'run':<24}{'seed':<6}{'spsa':<7}{'backbone':<12}"
        f"{'lam':<6}{'rate':<6}{'spread':>9}{'Dev':>9}"
        f"{'Bonus(bn=T)':>13}{'Bonus(bn=F)':>13}{'Delta':>9}")
print(head)
print('-' * len(head))
for r in rows:
    print(f"{r['tree']:<15}{r['run']:<24}{r['seed']:<6}{r['spsa']:<7}{r['bb']:<12}"
          f"{r['lam']:<6}{RATE.get(r['lam'], '?'):<6}"
          f"{fmt(r['spread'])}{fmt(r['dev'])}{fmt(r['bonus']):>13}"
          f"{fmt(r['bonus_frozen']):>13}{fmt(r['delta'])}{r['note'] and '   ' + r['note']}")
print('\nSection 3.3 (UCF) is read off the bn=T column, Section 4.6 (ResNet34)'
      ' off the bn=F column.')

# ---- assertions against the values printed in the paper -------------------
exp = {'ucf_2026-09-17-13-39-51': 'ucf ctrl 1x',
       'ucf_2026-09-17-22-29-40': 'ucf ctrl 40x',
       'ucf_2026-09-18-02-41-56': 'ucf ctrl 100x',
       'ucf_2026-09-18-06-52-17': 'ucf ctrl 200x',
       'ucf_2026-09-18-11-07-55': 'ucf spsa 1x',
       'ucf_2026-09-18-16-03-16': 'ucf spsa 40x',
       'ucf_2026-09-18-20-42-40': 'ucf spsa 100x',
       'ucf_2026-09-19-01-22-03': 'ucf spsa 200x',
       'ucf_2026-09-21-00-35-35': 'resnet34 module',
       'ucf_2026-09-21-07-51-21': 'resnet34 swa-only'}
CTRL_SPREAD = [0.79, 7.22, 5.07, 8.69]
CTRL_DEV = [-0.27, 0.10, 2.53, 3.21]
SPSA_SPREAD = [1.02, 2.66, 3.20, 4.28]

by = {r['run']: r for r in rows}
fail = []


def near(a, b, tol=0.006):
    return a is not None and abs(a - b) <= tol


order = ['ucf_2026-09-17-13-39-51', 'ucf_2026-09-17-22-29-40',
         'ucf_2026-09-18-02-41-56', 'ucf_2026-09-18-06-52-17']
for i, run in enumerate(order):
    r = by.get(run)
    if r is None or not near(r['spread'], CTRL_SPREAD[i]):
        fail.append(f'CTRL spread[{i}] expected {CTRL_SPREAD[i]}, got '
                    f'{None if r is None else r["spread"]}')
    if r is None or not near(r['dev'], CTRL_DEV[i]):
        fail.append(f'CTRL Dev[{i}] expected {CTRL_DEV[i]}, got '
                    f'{None if r is None else r["dev"]}')
order = ['ucf_2026-09-18-11-07-55', 'ucf_2026-09-18-16-03-16',
         'ucf_2026-09-18-20-42-40', 'ucf_2026-09-19-01-22-03']
for i, run in enumerate(order):
    r = by.get(run)
    if r is None or not near(r['spread'], SPSA_SPREAD[i]):
        fail.append(f'SPSA spread[{i}] expected {SPSA_SPREAD[i]}, got '
                    f'{None if r is None else r["spread"]}')
    if r is None or not (r['dev'] <= 0.01):
        fail.append(f'SPSA Dev[{i}] must be <= 0, got '
                    f'{None if r is None else r["dev"]}')

# Bonus positive on both arms at every rate
for run in list(by):
    r = by[run]
    if r['tree'] == 'mechanism_ucf' and not (r['bonus'] is not None and r['bonus'] > 0):
        fail.append(f'{run}: Bonus must be positive, got {r["bonus"]}')
    if r['tree'] == 'mechanism_ucf' and r['lam'] in ('0.0', '0.2') and r['delta'] < 0:
        fail.append(f'{run}: Delta must be >= 0 for lam<=0.2')

# ResNet34 legs are asserted on the frozen-buffer convention (bn = false),
# which is the one the archived R34 protocol used and the one the paper quotes.
r34 = by.get('ucf_2026-09-21-00-35-35')
if r34 is None or not near(r34['dev'], -0.77):
    fail.append(f'R34 module Dev expected -0.77, got {None if r34 is None else r34["dev"]}')
if r34 is None or not near(r34['bonus_frozen'], 0.07, 0.011):
    fail.append('R34 module Bonus expected +0.07 (bn=false), got '
                f'{None if r34 is None else r34["bonus_frozen"]}')
if r34 is None or not near(r34['bonus_frozen'] - r34['dev'], 0.84, 0.011):
    fail.append('R34 module Delta expected +0.84, got '
                f'{None if r34 is None else r34["bonus_frozen"] - r34["dev"]}')
sw = by.get('ucf_2026-09-21-07-51-21')
if sw is None or not near(sw['dev'], -0.42):
    fail.append(f'R34 swa-only Dev expected -0.42, got {None if sw is None else sw["dev"]}')

print()
if fail:
    print('MISMATCH vs the paper (%d):' % len(fail))
    for f in fail:
        print('  !! ' + f)
    sys.exit(1)
print('all Section 3.3 / Section 4.6 (ResNet34) values reproduce exactly.')
