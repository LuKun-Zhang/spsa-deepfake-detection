# SPSA + SWA for Cross-Domain Deepfake Detection — Reproduction Package

Code, configurations, and training logs for the paper:

> **A Superadditive Weight-Averaging Strategy with Split-Pool Self-Attention for Deepfake Detection**
> Lukun Zhang, Teng Wen, Rui Wang, Jingxin Su, and Huiyan Liu
> School of Software Engineering, Beijing Jiaotong University, Beijing, China
> *Submitted to ICASSP 2027*

---

## 👋 A note for ICASSP 2027 reviewers

Welcome — and thank you for reviewing this submission. This repository is its artifact.

**You do not need to run anything.** Reproducing the pipeline end to end takes FaceForensics++,
the four cross-domain datasets, and multi-GPU-days, and **no claim in the paper depends on it**.
The repository exists so that what *was* executed is inspectable in full.

Three pointers that may save you time:

- **[Where every claim lives](#where-every-claim-in-the-paper-lives)** — a table mapping each
  numbered claim in the paper to the logs, configs, or JSON files that carry it. Each headline
  number traces to a training log or a JSON file in this repository.
- **[Known caveats](#known-caveats)** — five things that would otherwise look like
  inconsistencies: a double-application in the detector code that is **deliberately preserved**,
  a checkpoint-format mismatch between training and evaluation, and one run whose intermediate
  checkpoints were lost and whose numbers survive only as frozen JSON.
- **[Evaluation protocol](#evaluation-protocol)** — what "7-point mean" means here, and why the
  paper reports the trajectory mean rather than the best checkpoint

Questions about the code, the protocol, or any number are very welcome.

![Figure 2](assets/fig2_pipeline.png)

*Fig. 2 of the paper — the full training and inference pipeline (`H_c`, the real/fake head, and
`H_s`, the method-classification head, both shown). The **SPSA** module is the red box inside the
reconstruction decoder: a 1×1 convolution expands the 128 channels, `Split(c | c)` halves them,
one branch runs multi-head self-attention (4 heads, global context) and the other PoolMix (3×3,
PoolFormer-style local pooling); the outputs are concatenated, fused by a 1×1 convolution, and
re-added via a residual shortcut. It **sits entirely inside the reconstruction decoder and never
touches the fingerprint-representation path used for classification** — 66.8K parameters, 0.14%
of the model. The blue box (top left) is the SWA collection window, drawn as the two protocols
actually used: `Controlled: E2+E3 average` and `Official: adaptive (lr < 40%, avg 3 ep)`.*

---

## What this repository is

A full reproduction package for a **negative-result-driven** study on training-time interventions
for cross-domain deepfake detection. The paper makes three points:

1. **Stochastic Weight Averaging (SWA) alone provides no consistent benefit** in short-horizon
   training.
2. A lightweight **Split-Pool Self-Attention (SPSA)** module — **66.8K parameters, 0.14% of the
   model** — is added to the reconstruction decoder. In controlled three-seed experiments
   **neither SPSA nor SWA helps independently** (SPSA alone **−0.38**, SWA alone **−0.39**),
   while their **combination** improves trajectory-mean AUC by **+1.27 points** (SPSA's net
   contribution on top of SWA is **+1.66**). What repeats is the **sign, under three different
   settings, each 3/3 seeds**: **+1.27** controlled (§4.2), **+1.02** under the official
   six-round protocol (§4.4), and **+1.23** on the deployed checkpoint (§4.6). The gain also
   holds across **four unseen cross-domain benchmarks** (OOD mean **+1.00**, positive on 4/4
   protocols — §4.5), and is **backbone-dependent** (§4.7).
3. **Best-checkpoint reporting is unstable** — it is sensitive to the random seed and to which
   checkpoints are sampled, and it can **reverse the sign** of a paired comparison. The paper
   therefore reports the **mean AUC along the training trajectory**.

The paper's headline claim rests on the **agreement of the sign across settings**, not on the
+2.04 interaction term — which the paper itself describes as *the arithmetic residual of the 2×2
table rather than a tested effect*.

This repository contains every configuration, training log, and evaluation artifact behind those
claims, laid out as a **single tree** (there are no experiment branches to navigate).

> 📖 **A note on language.** Every `README.md` in this repository is in English. The detailed
> documents under `docs/` are written in **Chinese**: they are the original lab notebooks, kept
> verbatim as the primary record. The numbers and provenance in them are authoritative; only the
> prose is Chinese.

---

## Repository layout

```
.
├── code/               Patched DeepfakeBench files — four self-contained protocol variants
├── configs/            All YAML run configurations (controlled, official, backbone, control arms)
├── config_dump/        Reference config dumps read from the header of each training log
├── logs/               Training logs — the only non-regenerable source of the 7-point trajectories
├── cross_domain/       Table 3 data: per-checkpoint metrics, evaluation script, final matrix
├── variants/           Additional controlled experiments — GaussSWA, SPSAOFF, reproduction, diagnostics
├── docs/               Lab notebooks: verified-data documents, experiment timeline, benchmark
├── scripts/            Chain scripts used to launch the multi-run batches
├── assets/             fig2_pipeline.png — the paper's Fig. 2, embedded above
└── README.md
```

| Path | Contents |
|---|---|
| `code/` | Four **self-contained** protocol variants — see below |
| `configs/` | YAML configs for every run: controlled 2×2, official protocol, backbone runs, and the SAM / Lookahead / EMA arms |
| `config_dump/` | Configuration dumps parsed from the **header of each training log** — the authoritative config source, since it records what the run actually used |
| `logs/` | `ctrl_training_logs/` holds the 12 controlled runs (one directory per run); the flat `train_*.log` files are the official six-round runs. Per-checkpoint metrics live here and nowhere else |
| `cross_domain/` | Table 3 (OOD): `cross_domain_results_7points.json`, `crossdomain_final_matrix_20260831.md`, `cross_domain_eval.py` |
| `variants/` | GaussSWA, SPSAOFF, reproduction runs, and the diagnostics that established seed determinism |
| `docs/` | `spsa1_ctrl_verified_data.md` (the single source of truth for all metrics), `authority_7point_benchmark.md`, the per-experiment verified-data documents, and `00_实验历史时间线.md` |
| `scripts/` | Chain scripts for the multi-run batches, plus the SAM / Lookahead / EMA patches |
| `assets/` | `fig2_pipeline.png` |

### `code/` — four protocol variants

The training loop and trainer differ between protocols; everything else is shared verbatim. Rather
than ship one tree with three mutually exclusive variants of `train.py`, each protocol gets its own
**complete, runnable** directory:

| Directory | Protocol | Also contains |
|---|---|---|
| `code/controlled/` | Controlled 4-cycle (`nEpochs=3`, four cosine cycles, consistency loss off) | `ucf_detector.py.bak_spsa2` — see caveat 1 |
| `code/official-fixed/` | Official 6-round cosine, fixed two-point SWA window | |
| `code/adaptive/` | Official 6-round cosine, learning-rate-threshold adaptive window | `networks/` — the EfficientNet-B4 and ResNet-34 backbones |
| `code/alt-methods/` | Controlled protocol + SAM / Lookahead / EMA arms | `optimizor/` — the SAM and LinearLR implementations |

The 12 files outside `train.py` / `trainer.py` are byte-identical across all four directories.

---

## Where every claim in the paper lives

| Paper | Claim | Evidence in this repository |
|---|---|---|
| §3.3, Table 1 | Controlled 2×2: SPSA alone **−0.38**, SWA alone **−0.39**, joint **+1.27** (3/3 seeds), SPSA net **+1.66**; interaction **+2.04** (arithmetic residual, not a tested effect) | `logs/ctrl_training_logs/` (12 runs), `config_dump/`, `docs/spsa1_ctrl_verified_data.md` |
| §3.3, Fig. 3 | Mean-fidelity gap: without SPSA the deployed averaged weights sit **1.15 pts** below their own 7-pt mean (**mean \|d\| = 1.03**); with SPSA, **\|d\| = 0.51** | `docs/spsa1_ctrl_verified_data.md`, `configs/` |
| §3.3 | Paired noise band **±0.95 pts** (**σ_mean = 0.947**, **σ_best = 1.039**) | `docs/authority_7point_benchmark.md` |
| §4.2, Table 2 | Control arms: SAM **+0.28**, Lookahead **+0.32** (both inside the band, both flip sign), EMA **−3.37** (sign-consistent, out of band) | `docs/sam_verified_data.md`, `docs/lookahead_verified_data.md`, `docs/ema_verified_data.md`, `code/alt-methods/optimizor/` |
| §4.3 | Fixed two-point SWA window: **+0.71**, only **2/3 seeds positive, negative on s2048** | `logs/train_official_cos_*.log`, `docs/official_cos_verified_data.md`, `docs/official_cos_s2048_correction.md` |
| §4.4 | Lr-threshold adaptive window (collects once lr < 40% of lr_max, averaging the cycle endpoints of the final three rounds): **+1.02**, 3/3 seeds, taking the 3-seed mean from **0.7012** (CTRL) to **0.7115** (SPSA+SWA) | `logs/train_official_*_adapt.log`, `docs/adaptive_window_verified_data.md` |
| §4.5, Table 3 | Cross-domain OOD mean **+1.00** (3/3 seeds, positive on 4/4): DFDC **+1.37**, FaceShifter **+1.53**, DeepFakeDetection **+0.14**, DFDCP **+0.94** | `cross_domain/` |
| §4.6 | Deployment sanity check: **+1.23** (3/3 seeds) | `configs/`, `logs/` |
| §4.7 | Backbone sensitivity: EfficientNet-B4 **+1.41** (all three controlled seeds), ResNet34 **−2.34** (all three) | `docs/efnb4_3seed_verified_data.md`, `docs/b4_backbone_3seed.md`, `docs/cross_backbone_verified_data.md`, `code/adaptive/networks/` |
| Fig. 1, §4.2 | Best-checkpoint instability: SPSA+SWA peaks at **0.7708** (E2-mid) / **0.7471** (E0-end) / **0.7387** (E4-end); the best moves by **2.97 / 4.37 / 0.71** pts (mean \|Δ\| **2.68**, beyond the ±0.95 band) while the trajectory mean moves by **1.08 / 1.52 / 0.44**; under best the sign flips (avg **+0.52**, s1024 **−1.42**) | `docs/authority_7point_benchmark.md`, `logs/train_official_cos_*.log`, `variants/` |
| §3.2 | Four cycles vs six: gain **+1.27** at **22975** steps against **+1.02** at **34463** — exactly **2/3** | `logs/`, `config_dump/` |
| §4.2 | `GaussMix+SWA` 7-point means **0.7098 / 0.6973**; misplaced/overscaled 1.65M variant **+0.36** (two-seed mean) | `variants/GaussSWA/`, `docs/spsa1_ctrl_verified_data.md` |

---

## Evaluation protocol

All numbers are **frame-level AUC on Celeb-DF-v2**, reported as the **7-point mean** over the
training trajectory:

```
E0-end · E1-mid · E1-end · E2-mid · E2-end · E3-mid · E3-end
```

The controlled protocol uses `nEpochs=3`, which produces **four** cosine cycles (E0–E3,
`T_max=3`, consistency loss off). The official protocol uses six cosine rounds. Seeds are labeled
by their `manualSeed` value: **s1024 / s2048 / s4096** (these are seed labels, *not* image
resolutions).

---

## Quick start

**Environment.** AutoDL RTX 5090 32 GB, PyTorch, running a fork of
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). Copy the protocol directory you want
from `code/` over your DeepfakeBench checkout. The backbone is **Xception**, initialized from
`xception-b5690688.pth`.

**Data.** As recorded in the log-header configuration dumps (the authoritative source):

| | |
|---|---|
| Train | **FaceForensics++** — all four manipulation subsets `FF-F2F, FF-DF, FF-FS, FF-NT`, **c23** compression |
| Test | **Celeb-DF-v2** |
| Cross-domain | DFDC, DFDCP, FaceShifter, DeepFakeDetection — symlinked into `datasets/rgb` |
| Input | pairs, 256×256, 32 frames per clip, batch size 16 |

Datasets must be obtained separately from their original providers — see the License section.

**Run.**

```bash
python train.py --detector_path configs/<config>.yaml
```

The scripts in `scripts/` chain the full multi-run batches. Each training writes a `training.log`
whose header contains a complete configuration dump — use that dump, not the YAML alone, when
reproducing a specific run.

---

## Known caveats

These are deliberate and load-bearing; please read before reproducing.

**1. SPSA2 is applied twice (kept intentionally).**
In `code/*/ucf_detector.py`, the forward pass contains `f_share = self.spsa2(f_share)` appearing
**twice in succession** (and the same duplication in the `__init__` block). This is the exact code
state used for all 12 controlled runs, and it is **preserved bug-for-bug**. Changing it will make
your SPSA2 numbers disagree with the paper. **SPSA1** (`spsa: true`, single application inside the
decoder) is unaffected. `code/controlled/ucf_detector.py.bak_spsa2` is kept as the byte-exact
artifact of the state this was diagnosed from.

**2. The SWA checkpoint has a deployment-form mismatch.**
The training output `swa.pth` carries a `module.` prefix and an `n_averaged` key (**617 keys**), while
`test.py` expects unprefixed weights (**616 keys**). Convert to `swa_fixed.pth` before cross-domain
evaluation — see `scripts/swa_retest_chain.sh`. The "SWA-Final" column in the tables is a
single-evaluation reference; the **primary metric is the 7-point mean**.

**3. Seed determinism required a patch.**
Deterministic behaviour across `np.random`, the dataloader workers, albumentations, and cuDNN is
handled in `code/*/train.py`. Without it, runs are not seed-reproducible. See
`docs/00_实验历史时间线.md`, stage 4.

**4. One run's intermediate checkpoints were lost.**
The reproduction run `ucf_2026-08-31-15-35-06` (controlled SPSA+SWA s2048, the 7-checkpoint weight
source for the Table 3 SPSA column) disappeared from cloud storage after 2026-08-31 and could not be
recovered. Its UCF-side trajectory is replaced by the first run of the same configuration
(`ucf_2026-08-24-09-59-38`), and its cross-domain values are frozen in
`cross_domain/cross_domain_results_7points.json` and `crossdomain_final_matrix_20260831.md`, so the
Table 3 numbers remain fully traceable. Details in `logs/ctrl_training_logs/README.md`.

**5. The documents under `docs/` predate the final paper.**
They were written during the experiments, against an earlier and longer manuscript, and still use
its table numbering (you will see references to "Table 4", "Table 5", etc. that no longer exist in
the submitted paper). Where a number in `docs/` differs from the paper, **the paper is
authoritative** — the docs preserve the analysis history, including intermediate attributions that
were later superseded by the three-seed finalization.

---

## Citation

If you use this code or these protocols, please cite:

```bibtex
@inproceedings{zhang2027spsa,
  title     = {A Superadditive Weight-Averaging Strategy with Split-Pool
               Self-Attention for Deepfake Detection},
  author    = {Zhang, Lukun and Wen, Teng and Wang, Rui and Su, Jingxin and Liu, Huiyan},
  booktitle = {Submitted to IEEE International Conference on Acoustics,
               Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

The entry will be updated with the proceedings details upon publication.

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

Note that **FaceForensics++, Celeb-DF-v2, DFDC, DFDCP, and DeepFakeDetection are third-party
datasets** with their own licenses and terms of use. This repository distributes **no dataset
content**; you must obtain each dataset from its original provider.
