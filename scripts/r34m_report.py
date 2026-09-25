# -*- coding: utf-8 -*-
"""Reduce one R34 mechanism leg into the mechanism quantities.

Usage:  python r34m_report.py <run_dir> [<run_out>]

Reads mediators.jsonl / probes.jsonl / window_probe.json / swa_eval.jsonl and
scrapes the per-test-point AUC(theta_t) out of the run log, then prints:

  * the theta_t test trajectory (dense: 6 per epoch, not the protocol's 2)
  * the AUC(theta_bar(w)) dose-response, frozen buffers vs BN-recalibrated
  * BONUS = AUC(theta_bar) - mean_t AUC(theta_t)
    DEV   = AUC(theta_end)  - mean_t AUC(theta_t)
    Delta = BONUS - DEV  ==  AUC(theta_bar) - AUC(theta_end)
  * a LEAVE-ONE-OUT table over the in-window test points, because the archive's
    R34 verdict rests on a single point (step 17231) and a mean whose sign is
    decided by one sample is the best-s1024-signflip disease, not a result
  * the weight-space mediators (rms_w / drift_w / R) and the probe series
    (f_t / grad_norm / agree / kl)
  * window_probe.json: J(w) frozen vs J_bn, B, DriftGap_quad, win_len_check

No GPU, no torch: this only reads text.
"""
import json
import os
import re
import sys


def jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


# The real line, confirmed on the 2026-09-21 run at step 956:
#
#   2026-09-21 00:45:14,988 - INFO - dataset: Celeb-DF-v2    step: 956    \
#       testing-metric, acc: 0.488...  testing-metric, auc: 0.69318320... \
#       testing-metric, eer: ...  testing-metric, ap: ...  video_auc: ...
#   2026-09-21 00:45:16,856 - INFO - dataset: avg    step: 956    \
#       testing-metric, acc: 0.488...  testing-metric, auc: 0.69318320...
#
# Three traps, all fatal and all silent.  Every one of them was caught by
# running the regex against this real line before trusting it:
#   * the anchor is `step: N`, NOT `Iter: N` -- `Iter:` is the training-side
#     counter, so a regex keyed on it matches nothing at all (it returned zero
#     points on a log that already had eleven);
#   * EVERY test point is printed TWICE, once under `dataset: Celeb-DF-v2` and
#     once under `dataset: avg`, with identical auc.  Requiring the dataset name
#     is what keeps one test point from being counted as two -- and a doubled
#     trajectory would corrupt mean_t, and through it BONUS and DEV;
#   * `acc:` is printed BEFORE `auc:` on the same line, so `testing-metric, auc:`
#     does not sit next to `step: N`.  The `.*?` spans that gap; it stays inside
#     one line because `.` does not match a newline, so it cannot reach across
#     into a neighbouring test point.
TEST_RE = re.compile(
    r'dataset:\s*Celeb-DF-v2\s+step:\s*(\d+)'
    r'.*?testing-metric,\s*auc:\s*([0-9.eE+-]+)')


def test_trajectory(out_path):
    """[(step, auc), ...] for the non-SWA test passes, in order."""
    if not out_path or not os.path.isfile(out_path):
        return []
    txt = open(out_path, encoding='utf-8', errors='replace').read()
    return [(int(m.group(1)), float(m.group(2))) for m in TEST_RE.finditer(txt)]


ITER_RE = re.compile(r'Iter:\s*(\d+)')


def last_iter(out_path):
    """Largest training step the log has reached, or None.

    Needed to tell a healthy mid-run leg whose window has not opened yet from a
    leg whose instruments failed to attach.  Both show zero instrument rows, and
    they mean opposite things -- the first is the expected state for the first
    ~2.8 h of every SWA leg (the window opens at step 17232), the second is the
    failure this whole exercise exists to detect.  Warn only on the second.
    """
    if not out_path or not os.path.isfile(out_path):
        return None
    txt = open(out_path, encoding='utf-8', errors='replace').read()
    hits = [int(m.group(1)) for m in ITER_RE.finditer(txt)]
    return max(hits) if hits else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(x, n=4):
    return 'None' if x is None else ('%.*f' % (n, x))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rd = sys.argv[1].rstrip('/')
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    name = os.path.basename(rd)
    print('=== %s' % name)
    print('    dir: %s' % rd)

    med = jsonl(os.path.join(rd, 'mediators.jsonl'))
    prb = jsonl(os.path.join(rd, 'probes.jsonl'))
    swa = jsonl(os.path.join(rd, 'swa_eval.jsonl'))
    wp = {}
    wpp = os.path.join(rd, 'window_probe.json')
    if os.path.isfile(wpp):
        try:
            wp = json.load(open(wpp))
        except Exception as e:
            print('    window_probe.json unreadable: %r' % (e,))

    tr = test_trajectory(out_path)
    print('    rows: mediators=%d probes=%d swa_eval=%d test_points=%d'
          % (len(med), len(prb), len(swa), len(tr)))
    if not med and not prb:
        li = last_iter(out_path)
        # window opens when epoch > swa_start(2), i.e. global step 3 * 5744
        WOPEN = 17232
        if li is not None and li < WOPEN:
            print('    window has not opened yet (at step %d of %d); '
                  'instruments silent by design' % (li, WOPEN))
        else:
            print('    !! NO WINDOW INSTRUMENTS at step %s (window opens %d) -- '
                  'this arm has SWA off, or the instruments failed to attach. '
                  'Nothing to explain the failure with.' % (li, WOPEN))

    # ---------------- window geometry ----------------
    wstart = wp.get('window_start_step')
    wlen = wp.get('window_len')
    if wstart is None and med:
        wstart = med[0]['step']
    print('\n--- window ---')
    print('    start=%s len=%s' % (wstart, wlen))
    if wp:
        wlc = wp.get('win_len_check', {})
        print('    win_len_check: collected=%s win_len=%s ok=%s'
              % (wlc.get('collected'), wlc.get('win_len'), wlc.get('ok')))
        print('    rng_restored=%s' % wp.get('rng_restored'))
        if wp.get('error'):
            print('    !! window_end_probe error: %s' % wp['error'])
        print('    f_start=%s f_end=%s' % (f(wp.get('f_start')), f(wp.get('f_end'))))

    # ---------------- theta_t trajectory + the mechanism identity ----------
    win_pts = []
    if wstart is not None and wlen:
        win_pts = [(s, a) for s, a in tr if wstart <= s <= wstart + wlen]
    print('\n--- theta_t test trajectory (window points marked *) ---')
    for s, a in tr:
        mark = '*' if (wstart is not None and wlen and wstart <= s <= wstart + wlen) else ' '
        print('    %s step %-6d auc %.4f' % (mark, s, a))

    if len(win_pts) >= 2:
        mt = mean([a for _, a in win_pts])
        end = win_pts[-1][1]
        print('\n--- the mechanism identity over %d in-window points ---' % len(win_pts))
        print('    mean_t AUC(theta_t) = %s   (min %.4f max %.4f)'
              % (f(mt), min(a for _, a in win_pts), max(a for _, a in win_pts)))
        print('    AUC(theta_end)      = %s  at step %d' % (f(end), win_pts[-1][0]))
        print('    DEV = AUC(theta_end) - mean_t = %s' % f(end - mt))
        # dose-response against the frozen-buffer convention
        for bn in (False, True):
            row = {}
            for r in swa:
                if bool(r.get('bn')) == bn:
                    auc = (r.get('metrics') or {}).get('auc')
                    if auc is not None:
                        row[r['w']] = auc
            if not row:
                continue
            tag = 'BN-recalibrated' if bn else 'frozen buffers'
            print('    --- theta_bar(w), %s ---' % tag)
            for w in sorted(row):
                bonus = row[w] - mt
                print('        w=%-7g AUC %.4f  BONUS %+0.4f  Delta=%s  G(vs max)=%+0.4f'
                      % (w, row[w], bonus, f(bonus - (end - mt)),
                         row[w] - max(a for _, a in win_pts)))
        w1 = dict((r['w'], (r.get('metrics') or {}).get('auc'))
                  for r in swa if not r.get('bn'))
        if 1.0 in w1 and w1[1.0] is not None:
            print('    headline (w=1.0, frozen): Delta = AUC(theta_bar) - AUC(theta_end) = %+0.4f'
                  % (w1[1.0] - end))

        # ---------------- leave-one-out ----------------
        print('\n--- leave-one-out on the in-window mean (is one point deciding?) ---')
        full = mean([a for _, a in win_pts]) - end
        print('    all %d points: mean_t-excl-nothing -> Delta(mean - end) = %+0.4f'
              % (len(win_pts), full))
        worst = min(win_pts, key=lambda p: p[1])
        best = max(win_pts, key=lambda p: p[1])
        for tag, pt in (('drop WORST', worst), ('drop BEST', best)):
            rest = [a for s, a in win_pts if s != pt[0]]
            m2 = mean(rest)
            print('    %s step %-6d (auc %.4f): mean_t=%.4f -> Delta = %+0.4f'
                  % (tag, pt[0], pt[1], m2, m2 - end))

    # ---------------- weight-space mediators ----------------
    if med:
        ws = sorted({k for r in med for k in (r.get('per_w') or {})})
        print('\n--- mediators over the window (per width) ---')
        for w in (['1'] if '1' in ws else ws[:1]) + [x for x in ws if x != '1'][:1]:
            seq = [(r['step'], (r.get('per_w') or {}).get(w, {})) for r in med]
            rs = [(s, d.get('rms_w')) for s, d in seq if d.get('rms_w') is not None]
            ds = [(s, d.get('drift_w')) for s, d in seq if d.get('drift_w') is not None]
            Rs = [(s, d.get('R')) for s, d in seq if d.get('R') is not None]
            print('    w=%s  n=%d' % (w, len(seq)))
            if rs:
                print('        rms_w   first %s  last %s  max %s'
                      % (f(rs[0][1], 6), f(rs[-1][1], 6), f(max(x[1] for x in rs), 6)))
            if ds:
                print('        drift_w first %s  last %s  max %s'
                      % (f(ds[0][1], 6), f(ds[-1][1], 6), f(max(x[1] for x in ds), 6)))
            if Rs:
                print('        R       first %s  last %s  min %s max %s  (n=%d)'
                      % (f(Rs[0][1]), f(Rs[-1][1]), f(min(x[1] for x in Rs)),
                         f(max(x[1] for x in Rs)), len(Rs)))
            else:
                print('        R       ALL NULL -- rms_w collapsed (defect E class)')

    # ---------------- probes ----------------
    if prb:
        ft = [(r['step'], r.get('f_t')) for r in prb if r.get('f_t') is not None]
        gn = [(r['step'], r.get('grad_norm')) for r in prb
              if r.get('grad_norm') is not None]
        moved = sum(1 for r in prb if r.get('rng_moved'))
        print('\n--- probes on the fixed batch B0 ---')
        print('    n=%d  rng_moved=%d %s' % (len(prb), moved,
                                             '(OK)' if moved == 0 else '(!! RNG PERTURBED)'))
        if ft:
            print('    f_t     first %s last %s min %s max %s'
                  % (f(ft[0][1]), f(ft[-1][1]),
                     f(min(x[1] for x in ft)), f(max(x[1] for x in ft))))
        if gn:
            print('    grad_norm first %s last %s min %s max %s (n=%d)'
                  % (f(gn[0][1]), f(gn[-1][1]),
                     f(min(x[1] for x in gn)), f(max(x[1] for x in gn)), len(gn)))
        w1 = [(r['step'], (r.get('per_w') or {}).get('1', {})) for r in prb]
        ag = [(s, d.get('agree')) for s, d in w1 if d.get('agree') is not None]
        kl = [(s, d.get('kl')) for s, d in w1 if d.get('kl') is not None]
        if ag:
            print('    agree(w=1) min %s last %s' % (f(min(x[1] for x in ag)), f(ag[-1][1])))
        if kl:
            print('    kl(w=1)    max %s  last %s' % (f(max(x[1] for x in kl)), f(kl[-1][1])))

    # ---------------- window_probe J / B ----------------
    if wp.get('J'):
        print('\n--- loss-surface J(w) = fbar(w) - f(theta_bar(w)) at window end ---')
        for w in sorted(wp['J'], key=float):
            j = wp['J'][w]
            jb = (wp.get('J_bn') or {}).get(w)
            print('    w=%-7s J=%s   J_bn=%s' % (w, f(j.get('J')), f(jb.get('J')) if jb else 'None'))
    for k in ('B', 'DriftGap_quad', 'DriftGap'):
        if wp.get(k) is not None:
            print('    %s = %s' % (k, f(wp[k], 6)))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    sys.exit(main())
