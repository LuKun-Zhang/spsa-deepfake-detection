# -*- coding: utf-8 -*-
"""Generate the ResNet34 mechanism yamls from the two SWA-carrying controlled arms.

Why ResNet34.  R34 is the one backbone where SPSA+SWA flips sign (-2.34 pt,
7/7 points negative, worst -4.15 at step 17231, and the official x6 run crashes
to 0.6247 at that same step).  Three things make that number unreadable as it
stands, and this run is built to remove all three:

  1. the archive legs were run with `cudnn: true`, and the same-machine rerun sd
     of this very protocol is 1.5-2.9 pt -- the same size as the effect, so the
     flip may not exist at all;
  2. the archive's matched-control arm has SWA=False, so the contrast mixes the
     SPSA module and the averaging into one number and cannot attribute either;
  3. the archive had NO window instruments and its 7-point cadence makes
     G = AUC(theta_bar) - max_t AUC(theta_t) a max over two in-window samples,
     i.e. one point can decide the sign (the best-s1024-signflip disease).

So every leg below is built on the 871 mechanism recipe (gen_mexp_configs.py):
same instruments, same cadences, same protocol repairs.  What changes is the
backbone, and that the CONTROL here is the SWA-carrying arm -- the mechanism
section's own control -- rather than the SWA-free one, so that BOTH sides of the
contrast produce window dynamics instead of only one.

Run on the box, from the DeepfakeBench root:
    /root/miniconda3/bin/python /root/gen_r34m_configs.py
"""
import os

import yaml

REPO = '/root/autodl-tmp/DeepfakeBench'
DET = os.path.join(REPO, 'training/config/detector')
BASE_SPSA1SWA = os.path.join(DET, 'r34_stat_SPSA1SWA_resnet34_s2048.yaml')
BASE_SWAONLY = os.path.join(DET, 'r34_stat_swaonly_resnet34_s2048.yaml')
BASE_CTRL = os.path.join(DET, 'r34_stat_ctrl_resnet34_s2048.yaml')
BASE_SPSAONLY = os.path.join(DET, 'r34_stat_spsaonly_resnet34_s2048.yaml')

# ---- the 871 mechanism recipe, backbone-independent ----------------------
COMMON = dict(
    # /autodl-fs is the persistent store (2.3 TB free); the system disk has
    # 7.9 GB left and one leg writes ~19 testpoint checkpoints + 5 swa_w*.pth.
    log_dir='/autodl-fs/data/r34m/logs',
    cudnn=False,               # cudnn.benchmark is the non-determinism root cause
    rng_neutral_build=True,    # arms differ by the module, not by the data order
    manualSeed=2048,           # same seed as the archive leg being questioned
    SWA=True,
    swa_start=2,
    nEpochs=3,
    save_ckpt=True,            # train.py's CLI default is already True; explicit
    resolution=256,
    # design 5.3: five parallel averages, same endpoint, different WIDTH
    swa_ws=[1.0, 0.5, 0.25, 0.125, 0.0625],
    # design 5.2: fixed-batch loss-landscape probe, window-relative cadence
    probe_every=300,
    # Raised from the 871 value of 1500: ||grad L(theta_t;B0)|| is the one
    # dynamics channel that costs nothing (one extra backward on a fixed batch)
    # and 1500 would give only 4 readings across the whole window.  At 300 it
    # rides on every probe row, and 300 is a multiple of probe_every so the
    # window-relative guard lines up by construction.
    probe_grad_every=300,
    # 6 per epoch instead of the protocol's 2.  This is the instrument that
    # makes the R34 verdict readable: the archive's 7-point mean carries its
    # whole effect in one point, and G is a difference against a MAXIMUM, which
    # a 2-sample max cannot support.
    test_times_per_epoch=6,
    # AveragedModel keeps the buffers of the last averaged point, so all five
    # theta_bar(w) share theta_end's statistics; 100 training batches per width
    # at window end recompute them and give a second, interpretable column.
    bn_recalib_batches=100,
    # Natural rate.  The lambda route is dead (see trainer.py's header); this run
    # measures the protocol's own regime, not a manipulated one.
    tail_lam=0.0,
)

# ---- legs, in chain order.  The first two carry SWA and therefore the whole
# ---- instrument set; they are the primary pair and the only two that must run.
LEGS = [
    # the failing arm, exactly the one Table 4 reports
    ('r34m_spsa1swa_s2048', BASE_SPSA1SWA, dict(spsa=True, spsa_mode='spsa')),
    # the same backbone and the same window with only the module removed: this
    # is the arm that separates "SPSA breaks R34" from "SWA on R34 breaks R34",
    # and it is one of the four split legs chain_wave1_r34.sh never ran.
    ('r34m_swaonly_s2048', BASE_SWAONLY, dict(spsa=False)),
    # SWA-free arms.  No window and therefore no instruments, but with the dense
    # test cadence they still give the per-point trajectory that decides whether
    # a dip at 17231 is special to SPSA or is a property of R34 at that step.
    ('r34m_ctrl_s2048', BASE_CTRL, dict(spsa=False, SWA=False)),
    ('r34m_spsaonly_s2048', BASE_SPSAONLY, dict(spsa=True, spsa_mode='spsa', SWA=False)),
]

# ---- the single-key control pair (added 2026-09-21 mid-campaign, on the
# ---- user's explicit instruction to run it ahead of the remaining legs).
# ----
# ---- The question it answers.  The archive's R34 pair came out -2.34 pt, and
# ---- the identical protocol run here comes out +1.72 pt -- at the SAME
# ---- manualSeed=2048, so the seed is not the discriminating variable and more
# ---- seeds would not explain the disagreement.  Of every key in COMMON, only
# ---- rng_neutral_build changes WHICH DATA the model sees and in what order.
# ---- The archive ran with it off, which means its "matched pair" was not
# ---- matched at the data level at all: constructing the SPSA module draws the
# ---- global RNG before the DataLoader gets its base seed, so the two arms
# ---- trained on different shuffle orders.  This pair flips that one key back
# ---- to what the archive had and changes nothing else.
# ----
# ---- Why this one and not cudnn.  cudnn.benchmark is the documented root cause
# ---- of this project's 1.5-2.9 pt rerun sd -- but it is NOISE, so one
# ---- replicate landing negative proves nothing and landing positive exonerates
# ---- nothing; it would take several runs.  rng_neutral_build is a systematic
# ---- change and this pair runs deterministic (cudnn=False), so a single run is
# ---- decidable.  Two outcomes, both informative, no ambiguity:
# ----   * negative (~-2.3) -> the data order IS the cause, the archive's pair
# ----     was confounded, and +1.72 is the properly-paired number;
# ----   * positive (~+1.7) -> the data order is exonerated and the next cut is
# ----     cudnn, which then needs replicates rather than one run.
CONTROL = [
    ('r34m_spsa1swa_norgb_s2048', BASE_SPSA1SWA,
     dict(spsa=True, spsa_mode='spsa', rng_neutral_build=False)),
    ('r34m_ctrl_norgb_s2048', BASE_CTRL,
     dict(spsa=False, SWA=False, rng_neutral_build=False)),
]

# ---- preflight: a ~5 minute leg whose only job is to prove that the
# ---- instruments WRITE on this backbone before 16 GPU-hours are committed.
# ---- swa_start=-1 opens the window at global step 0, so mediators.jsonl and
# ---- probes.jsonl must both have rows within the first minute.  This is a
# ---- write-test, not a correctness test: a window opening at step 0 has
# ---- residue 0 (mod 300) and is blind to the defect-D class, which is why the
# ---- real chain is what actually exercises the guard (window at 17232 = 132).
PREFLIGHT = ('r34m_preflight', BASE_SPSA1SWA, dict(
    spsa=True, spsa_mode='spsa', nEpochs=0, dry_run=True, swa_start=-1,
    probe_every=100, probe_grad_every=100, bn_recalib_batches=10,
    log_dir='/autodl-fs/data/r34m/preflight'))


def write(name, base, over):
    if not os.path.isfile(base):
        print('!! MISSING BASE %s -- skipped' % base)
        return False
    cfg = yaml.safe_load(open(base))
    cfg.update(COMMON)
    cfg.update(over)
    path = os.path.join(DET, name + '.yaml')
    with open(path, 'w') as fh:
        fh.write('# GENERATED by gen_r34m_configs.py from %s\n' % base)
        fh.write('# overrides: %s\n' % over)
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    print('wrote %s' % path)
    print('   spsa=%-5s SWA=%-5s tail_lam=%-4s seed=%s cudnn=%s rng_neutral=%s '
          'ws=%s probe=%s/%s test/ep=%s bn=%s'
          % (cfg.get('spsa'), cfg['SWA'], cfg['tail_lam'], cfg['manualSeed'],
             cfg['cudnn'], cfg['rng_neutral_build'], len(cfg['swa_ws']),
             cfg['probe_every'], cfg['probe_grad_every'],
             cfg['test_times_per_epoch'], cfg['bn_recalib_batches']))
    # the parse check that matters: swa_ws must survive as a real list, because
    # train.py's `or [1.0]` silently collapses None/[] to a single width.
    rt = yaml.safe_load(open(path))
    if rt['SWA']:
        assert isinstance(rt['swa_ws'], list) and len(rt['swa_ws']) == 5, rt['swa_ws']
    return True


def main():
    for d in ('/autodl-fs/data/r34m/logs', '/autodl-fs/data/r34m/preflight'):
        os.makedirs(d, exist_ok=True)
    n = 0
    for name, base, over in LEGS:
        n += write(name, base, over)
    write(*PREFLIGHT)
    print('\n%d/%d legs written + 1 preflight' % (n, len(LEGS)))
    print('chain order: %s' % ' '.join(l[0] for l in LEGS))

    c = 0
    for name, base, over in CONTROL:
        c += write(name, base, over)
    print('\n%d/%d control legs written (single key flipped: rng_neutral_build)'
          % (c, len(CONTROL)))
    print('control order: %s' % ' '.join(l[0] for l in CONTROL))


if __name__ == '__main__':
    main()
