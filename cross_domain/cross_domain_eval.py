#!/usr/bin/env python
# Cross-domain evaluation for 7 test-point snapshots + SWA model.
# Reuses DeepfakeBench dataset loading / metrics exactly as trainer.test_one_dataset.
# Usage:
#   python cross_domain_eval.py [--config YAML] [--ckpt_dir DIR] [--swa PATH]
#       [--datasets DS...] [--steps s1,s2] [--workers N] [--out PATH]
import sys, os, yaml, argparse, json, glob
import numpy as np
import torch

BASE = '/root/autodl-tmp/DeepfakeBench'
sys.path.insert(0, BASE); sys.path.insert(0, BASE + '/training'); os.chdir(BASE)

from detectors import DETECTOR
from dataset import DeepfakeAbstractBaseDataset
from metrics.utils import get_test_metrics


def load_config(yaml_path):
    """Replicate train.py config merge exactly."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']
    for _k in ['SWA', 'swa_start', 'lr_scheduler', 'lr_T_max', 'lr_eta_min']:
        if _k in config:
            config2[_k] = config[_k]
    config.update(config2)
    if config.get('lmdb', False):
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['local_rank'] = 0; config['ddp'] = False
    return config


def make_loader(config, dataset_name, workers):
    c2 = config.copy(); c2['test_dataset'] = dataset_name
    ts = DeepfakeAbstractBaseDataset(config=c2, mode='test')
    loader = torch.utils.data.DataLoader(
        dataset=ts, batch_size=config['test_batchSize'], shuffle=False,
        num_workers=workers, collate_fn=ts.collate_fn,
        drop_last=False)  # no drop_last: eval must keep every frame, else img_names length mismatch
    return loader, ts.data_dict['image']


@torch.no_grad()
def eval_ckpt(config, model, ckpt_path, datasets, workers, name=None):
    sd = torch.load(ckpt_path, map_location='cpu')
    # strip extra AveragedModel keys (swa.pth has n_averaged etc.)
    # swa.pth keys carry a 'module.' prefix from torch.optim.swa_utils.AveragedModel
    model_sd = model.state_dict()
    clean = {}
    for k, v in sd.items():
        if k in model_sd:
            clean[k] = v
        elif k.startswith('module.') and k[len('module.'):] in model_sd:
            clean[k[len('module.'):]] = v
    sd = clean
    model.load_state_dict(sd)
    model.eval()
    results = {}
    for ds in datasets:
        try:
            loader, img_names = make_loader(config, ds, workers)
            preds, labels = [], []
            for data_dict in loader:
                if 'label_spe' in data_dict:
                    data_dict.pop('label_spe')
                data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
                for k in data_dict.keys():
                    if data_dict[k] is not None:
                        data_dict[k] = data_dict[k].cuda()
                pred = model(data_dict, inference=True)
                labels += list(data_dict['label'].cpu().numpy())
                preds += list(pred['prob'].cpu().numpy())
            # safety alignment: guard against any drop / missing tail
            img_names = list(img_names)
            if len(img_names) > len(preds):
                print(f'  [warn] {name}/{ds}: img_names {len(img_names)} > preds {len(preds)}, truncating', flush=True)
                img_names = img_names[:len(preds)]
            elif len(img_names) < len(preds):
                print(f'  [warn] {name}/{ds}: img_names {len(img_names)} < preds {len(preds)}, truncating preds', flush=True)
                preds = preds[:len(img_names)]
            m = get_test_metrics(y_pred=np.array(preds), y_true=np.array(labels), img_names=img_names)
            results[ds] = {k: float(v) for k, v in m.items() if k in ('auc', 'acc', 'eer', 'ap', 'video_auc')}
            print(f'    {ds}: auc={results[ds]["auc"]:.6f} video_auc={results[ds]["video_auc"]:.6f}', flush=True)
        except Exception as e:
            print(f'    [FAIL] {ds}: {type(e).__name__}: {e}', flush=True)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='/root/autodl-tmp/stat_cfgs/stat_SPSA1SWA_s2048.yaml')
    ap.add_argument('--ckpt_dir', default='/root/autodl-tmp/DeepfakeBench/logs/training/ucf_2026-08-31-15-35-06/testpoints')
    ap.add_argument('--swa', nargs='+', default=[
        '/root/autodl-tmp/DeepfakeBench/logs/training/ucf_2026-08-31-15-35-06/swa.pth',
        '/root/autodl-tmp/DeepfakeBench/logs/training/ucf_2026-08-24-09-59-38/swa.pth'], help='SWA ckpt(s); labels = dir suffix')
    ap.add_argument('--datasets', nargs='+',
                    default=['Celeb-DF-v2', 'DFDC', 'FaceShifter', 'DeepFakeDetection', 'Celeb-DF-v1', 'UADFV', 'DFDCP'])
    ap.add_argument('--steps', default=None, help='comma-separated steps, e.g. 5743,8615; default=all in ckpt_dir')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='/root/autodl-tmp/cross_domain_results.json')
    args = ap.parse_args()

    config = load_config(args.config)
    model = DETECTOR[config['model_name']](config).cuda()
    model.device = torch.device('cuda')
    model.epoch = 0

    # discover checkpoints
    ckpts = []
    if args.steps:
        for s in args.steps.split(','):
            p = os.path.join(args.ckpt_dir, f'step_{s.strip()}.pth')
            if os.path.exists(p):
                ckpts.append((f'step_{s.strip()}', p))
            else:
                print(f'[warn] missing {p}')
    else:
        for p in sorted(glob.glob(os.path.join(args.ckpt_dir, 'step_*.pth'))):
            ckpts.append((os.path.basename(p)[:-4], p))
    for swa_p in args.swa:
        if os.path.exists(swa_p):
            lab = 'swa_' + os.path.basename(os.path.dirname(swa_p))[-11:]
            ckpts.append((lab, swa_p))
        else:
            print(f'[warn] missing swa {swa_p}')
    print(f'Evaluating {len(ckpts)} checkpoints on {args.datasets}')

    all_results = {}
    for name, path in ckpts:
        print(f'== {name} ==', flush=True)
        all_results[name] = eval_ckpt(config, model, path, args.datasets, args.workers, name=name)
        # incremental save: keep partial progress across crashes
        json.dump(all_results, open(args.out, 'w'), indent=2)
        print(f'  (saved {name} -> {args.out})', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(all_results, open(args.out, 'w'), indent=2)
    print(f'\nSaved -> {args.out}')

    dss = args.datasets
    print('\n=== SUMMARY (auc) ===')
    print('ckpt'.ljust(12), *[d.ljust(18) for d in dss])
    for name, r in all_results.items():
        row = [name.ljust(12)]
        for d in dss:
            row.append(f'{r.get(d, {}).get("auc", -1):.6f}'.ljust(18))
        print(*row)
