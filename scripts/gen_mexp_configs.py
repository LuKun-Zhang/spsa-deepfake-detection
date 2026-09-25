# -*- coding: utf-8 -*-
"""Generate the mechanism-experiment yamls from the controlled SPSA+SWA base.

All legs share one base so that the ONLY differences are the declared factors:
  factor 1  spsa      : off / on
  factor 2  tail_lam  : 0.0 / 0.20 / 1.0

lambda is the window's lr clamp, expressed as a fraction of lr_max (2e-4):
  lam=0.0   the window stays on the schedule's own lr (1e-6) -> frozen, the
            original protocol's inert manipulation
  lam=0.20  a moderate window
  lam=1.0   the window runs at lr_max -- the top of the protocol's own lr
            envelope.  lam>1 would exceed any lr the schedule ever uses, so
            {0, 0.20, 1.0} spans the whole admissible domain rather than being
            an arbitrary coarse sample of it.
and, for every leg, the two protocol repairs:
  cudnn: false             -- cudnn.benchmark is the non-determinism root cause
  rng_neutral_build: true  -- makes arms differ only by the module, not by the
                              data order (measured confound, 2026-09-16)

Run on the box, from the DeepfakeBench root:
    /root/miniconda3/bin/python /root/gen_mexp_configs.py
"""
import os
import yaml

DET = 'training/config/detector'
BASE = os.path.join(DET, 'ucfx_stat_spsa1swa_xception_s2048.yaml')

COMMON = dict(
    # /autodl-fs is the persistent file store (976 GB free, survives instance
    # release); the system disk is only 28 GB and the 4 legs alone would put
    # ~25 GB of swa_w*.pth on it.  The earlier /root/swa_exp choice was made
    # when /autodl-fs was reported full -- it is not.
    log_dir='/autodl-fs/data/swa_exp/logs',
    cudnn=False,
    rng_neutral_build=True,
    manualSeed=2048,
    SWA=True,
    swa_start=2,
    nEpochs=3,
    save_ckpt=False,
    resolution=256,
    spsa_mode='spsa',
    # design 5.3: five parallel averages, same endpoint, different window WIDTH
    swa_ws=[1.0, 0.5, 0.25, 0.125, 0.0625],
    # design 5.2: fixed-batch loss-landscape probe every 300 steps
    probe_every=300,
    probe_grad_every=1500,
    # G = AUC(theta_bar) - max_t AUC(theta_t) is a difference against a MAXIMUM;
    # with the default 2 in-window evaluation points the max is a 2-sample
    # estimate and G can flip sign on one point (see best-s1024-signflip).
    test_times_per_epoch=6,
    # design 5.3 / the 2026-09-16 smoke: AveragedModel keeps the BatchNorm
    # buffers of a single point, and all five theta_bar(w) share them.  Harmless
    # while the window sits on a tiny lr (the protocol's own regime); at a
    # window lr of 2e-4 it makes f(theta_bar(1)) diverge geometrically to 3.4e10
    # while every point on the trajectory stays under 2.3.  This many training
    # batches are run through each averaged model at window end to recompute its
    # running stats, giving a second, interpretable AUC(theta_bar(w)) column.
    # 0 disables the recalibrated column (frozen-buffer only).
    bn_recalib_batches=100,
)

# ---- the 6-leg grid: lambda x SPSA, seed 2048 ----------------------------
# Ordered the way the chain should run them: the two CTRL legs come first so
# the lambda->R gate is answered before any SPSA GPU time is spent, then the
# two SPSA legs complete the 2x2, then lam=1.0 completes the domain.
PROBE = [
    ('mexp_p_ctrl_l0',     dict(spsa=False, tail_lam=0.0)),
    ('mexp_p_ctrl_l20',    dict(spsa=False, tail_lam=0.20)),
    ('mexp_p_spsa_l0',     dict(spsa=True,  tail_lam=0.0)),
    ('mexp_p_spsa_l20',    dict(spsa=True,  tail_lam=0.20)),
    ('mexp_p_ctrl_l100',   dict(spsa=False, tail_lam=1.0)),
    ('mexp_p_spsa_l100',   dict(spsa=True,  tail_lam=1.0)),
]

# ---- smoke: forces the collection window open at epoch 0 so the new code
# ---- paths (lambda clamp, mediator dump) are exercised within ~200 steps ----
# dry_run now reaches through the train_config.yaml merge only because train.py's
# whitelist was extended; it also pins nEpochs to 0 and turns feature saving off.
SMOKE = ('mexp_smoke', dict(spsa=True, tail_lam=0.20, nEpochs=0, dry_run=True,
                            swa_start=-1, log_dir='/autodl-fs/data/swa_exp/smoke'))

# ---- lambda = 0 smoke (V4): the clamp must NOT fire and lr must stay on the
# ---- scheduler's value; this is the neutral arm of the manipulation ---------
SMOKE_L0 = ('mexp_smoke_l0', dict(spsa=True, tail_lam=0.0, nEpochs=0, dry_run=True,
                                  swa_start=-1,
                                  log_dir='/autodl-fs/data/swa_exp/smoke_l0'))

# ---- WINDOW-RESIDUE smoke (2026-09-17).  The only smoke that can catch a
# ---- defect of the form "the guard counts in one unit, the caller in another".
# ---- nEpochs:1 + swa_start:0 opens the window at global step 5744 = 44 (mod
# ---- 300) -- the same NON-ZERO residue class as the real legs' 17232 = 132.
# ---- The two smokes above open the window at global step 0, residue 0, which
# ---- makes every `step % 300 == 0` guard trivially satisfiable: that is
# ---- exactly how defect D (window-relative probe cadence) passed two clean
# ---- smokes and still produced 0 probes on the real leg.
# ---- dry_run must be OFF -- it pins nEpochs to 0 and would re-create the same
# ---- blind spot.  Window = epoch 1, length 5744, i.e. the real window length.
# ---- Two epochs => ~2.1 h on the 4090D, deliberately half a real leg.
SMOKE_WIN = ('mexp_smoke_win', dict(spsa=False, tail_lam=0.20, nEpochs=1,
                                    swa_start=0, dry_run=False,
                                    log_dir='/autodl-fs/data/swa_exp/smoke_win'))


def write(name, over):
    cfg = yaml.safe_load(open(BASE))
    cfg.update(COMMON)
    cfg.update(over)
    path = os.path.join(DET, name + '.yaml')
    with open(path, 'w') as fh:
        fh.write('# GENERATED by gen_mexp_configs.py from %s\n' % BASE)
        fh.write('# overrides: %s\n' % over)
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    print('wrote %s' % path)
    # echo the keys that decide this leg's identity
    print('   spsa=%s spsa_mode=%s tail_lam=%s SWA=%s cudnn=%s rng_neutral=%s seed=%s'
          % (cfg['spsa'], cfg.get('spsa_mode'), cfg['tail_lam'], cfg['SWA'],
             cfg['cudnn'], cfg['rng_neutral_build'], cfg['manualSeed']))


def main():
    for d in ('/autodl-fs/data/swa_exp/logs', '/autodl-fs/data/swa_exp/smoke',
              '/autodl-fs/data/swa_exp/smoke_l0', '/autodl-fs/data/swa_exp/smoke_win'):
        os.makedirs(d, exist_ok=True)
    for name, over in PROBE:
        write(name, over)
    write(*SMOKE)
    write(*SMOKE_L0)
    write(*SMOKE_WIN)
    print('\nbase = %s' % BASE)


if __name__ == '__main__':
    main()
