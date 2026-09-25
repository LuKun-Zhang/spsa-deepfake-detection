
import os, sys, yaml, json, torch, numpy as np, torch.nn as nn
os.chdir("/root/autodl-tmp/DeepfakeBench")
sys.path.insert(0, "/root/autodl-tmp/DeepfakeBench/training")
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics
from torch.utils.data import DataLoader

LEGS = [('r34_a', 'ucf_2026-09-21-00-35-35'), ('r34_c', 'ucf_2026-09-21-07-51-21')]
DEV = "cuda"
ROOT = "/autodl-fs/data/r34m/logs"
STEPS = (17225, 18182, 19139, 20096, 21053, 22010, 22967)
WS = ("0.0625", "0.125", "0.25", "0.5", "1")
Y_SP = "/root/eval_probe_r34_spsa.yaml"
Y_CT = "/root/eval_probe_r34_ctrl.yaml"

def cfg_of(leg):
    """按 checkpoint 里的 spsa 键选 yaml——不靠时间戳猜身份。"""
    p = os.path.join(leg, "testpoints", "step_22967.pth")
    ck = torch.load(p, map_location="cpu")
    n = sum(1 for k in ck if ".spsa" in k)
    yp = Y_SP if n else Y_CT
    c = yaml.safe_load(open(yp))
    c.update(yaml.safe_load(open("./training/config/test_config.yaml")))
    print("    [cfg] %s  spsa_keys=%d -> %s" % (os.path.basename(leg), n, os.path.basename(yp)))
    sys.stdout.flush()
    return c

def build(cfg, path):
    m = DETECTOR[cfg["model_name"]](cfg).to(DEV)
    m.load_state_dict(torch.load(path, map_location=DEV), strict=True)
    return m

def flat(src, dst):
    ck = torch.load(src, map_location="cpu")
    out = {k[7:] if k.startswith("module.") else k: v
           for k, v in ck.items() if torch.is_tensor(v) and k != "n_averaged"}
    torch.save(out, dst); return dst

def mk_loader(cfg, name, mode, bs=32):
    c = dict(cfg); c["test_dataset"] = name; c["train_dataset"] = [name]
    ds = DeepfakeAbstractBaseDataset(config=c, mode=mode)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4,
                    collate_fn=ds.collate_fn, drop_last=False)
    return ds, dl

def move(d):
    lab = torch.where(d["label"] != 0, 1, 0)
    d["image"] = d["image"].to(DEV); d["label"] = lab.to(DEV)
    if d.get("mask") is not None: d["mask"] = d["mask"].to(DEV)
    if d.get("landmark") is not None: d["landmark"] = d["landmark"].to(DEV)
    return d

@torch.no_grad()
def evaluate(m, dl, names):
    m.eval(); ps, ls = [], []
    for d in dl:
        d = move(d)
        p = m(d, inference=True)
        ps += list(p["prob"].cpu().numpy()); ls += list(d["label"].cpu().numpy())
    return get_test_metrics(y_pred=np.array(ps), y_true=np.array(ls), img_names=names)["auc"]

def recalib(m, dl, n=100):
    m.train()
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            mod.momentum = None; mod.reset_running_stats()
    with torch.no_grad():
        for i, d in enumerate(dl):
            if i >= n: break
            move(d); m(d)
    return m

ds_te, dl_te = mk_loader(cfg_of(os.path.join(ROOT, LEGS[0][1])), "Celeb-DF-v2", "test")
names_te = ds_te.data_dict["image"]
# 训练集（重校准用）固定用 xcep 版 yaml 即可：它只提供 data 路径，与骨干无关
_c = yaml.safe_load(open(Y_CT)); _c.update(yaml.safe_load(open("./training/config/test_config.yaml")))
ds_tr, dl_tr = mk_loader(_c, "FF-F2F", "train")
print("test batches = %d  train batches = %d" % (len(dl_te), len(dl_tr))); sys.stdout.flush()

for tag, dname in LEGS:
    outp = "/root/recalib8_%s.json" % tag
    if os.path.exists(outp):
        print("SKIP (exists) %s" % tag); sys.stdout.flush(); continue
    leg = os.path.join(ROOT, dname)
    print("=== LEG %s  (%s) ===" % (tag, dname)); sys.stdout.flush()
    cfg = cfg_of(leg)
    rows = []
    for s in STEPS:
        src = os.path.join(leg, "testpoints", "step_%d.pth" % s)
        if not os.path.exists(src): print("  MISSING", src); continue
        a1 = evaluate(recalib(build(cfg, src), dl_tr, 100), dl_te, names_te)
        rows.append(dict(kind="theta_t", name="step_%d" % s, recalib_train=a1))
        print("  t step_%-6d recalib=%.10f" % (s, a1)); sys.stdout.flush()
        del a1
    for w in WS:
        src = os.path.join(leg, "swa_w%s.pth" % w)
        if not os.path.exists(src): print("  MISSING", src); continue
        fp = flat(src, "/root/_tmp_flat_r34.pth")
        a0 = evaluate(build(cfg, fp), dl_te, names_te)
        a1 = evaluate(recalib(build(cfg, fp), dl_tr, 100), dl_te, names_te)
        rows.append(dict(kind="theta_bar", name="swa_w%s" % w, frozen=a0, recalib_train=a1))
        print("  b w=%-7s frozen=%.10f  recalib=%.10f" % (w, a0, a1)); sys.stdout.flush()
    json.dump(rows, open(outp, "w"), indent=1)
    print("  WROTE", outp); sys.stdout.flush()
print("== ALL DONE ==")
