#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Generate 3 EMA-arm yamls (s1024/2048/4096) from the stat_sam template.
# EMA arm = controlled pure detector (spsa:false, SWA:false) + Adam (same hparams) + EMA weight averaging.
import yaml, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stat_sam_s1024.yaml')
OUTDIR = os.path.dirname(os.path.abspath(__file__))
seeds = [1024, 2048, 4096]

with open(SRC, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

for seed in seeds:
    c = dict(cfg)
    c['log_dir'] = '/root/autodl-tmp/DeepfakeBench/logs/ucf_ema_s%d' % seed
    c['manualSeed'] = seed
    c['optimizer'] = {'type': 'adam',
                      'adam': dict(cfg['optimizer']['adam'])}
    c['SWA'] = False
    c['spsa'] = False
    c['EMA'] = True
    c['ema_decay'] = 0.999
    out = os.path.join(OUTDIR, 'stat_ema_s%d.yaml' % seed)
    with open(out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(c, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
    print('WROTE', out, 'optimizer.type=', c['optimizer']['type'], 'EMA=', c['EMA'], 'decay=', c['ema_decay'])
