# Configuration provenance — which YAMLs are originals, which are reconstructions, and which describe runs that never happened

`configs/` mixes three kinds of file. This document says which is which, so that
no configuration is mistaken for evidence.

## Class A — archived originals (the majority)

Pulled byte-for-byte from the detector config directory of the machine that ran
them. Nothing was edited. This covers `stat_*`, `r34_stat_*` (s2048), `r34x6*`,
`cnx_*`, `swv2_*`, `mexp_*`, `r34m_*`, `sota_*`, `ucfx_stat_*`, and the
official-protocol configs.

## Class B — reconstructed (3 files) — **the run happened, the YAML was not archived**

| file | derived from | seed | run | log shipped? |
|---|---|---|---|---|
| `configs/stat_SPSA_s1024.yaml` | `stat_SPSA_s2048.yaml` | 2048 → 1024 | `ucf_2026-08-22-14-28-48` | yes |
| `configs/stat_CTRLSWA_s4096.yaml` | `stat_CTRLSWA_s2048.yaml` | 2048 → 4096 | `ucf_2026-08-29-02-08-19` | yes |
| `configs/stat_SPSA1SWA_s4096.yaml` | `stat_SPSA1SWA_s2048.yaml` | 2048 → 4096 | `ucf_2026-08-28-23-38-12` | yes |

These three seeds are part of the 12-run 2×2 table, but their per-run YAMLs were
not kept. Each was rebuilt from its **same-arm sibling** by changing
`manualSeed` **and nothing else**, and each file carries a header saying so.

**They are not guesses.** Each was checked field by field against the
configuration dump in that run's own `training.log` header — the authoritative
record of what the run actually used:

```
config                        seed(cfg) seed(dump)  keys compared  mismatches
stat_SPSA_s1024                   1024       1024             46           0
stat_CTRLSWA_s4096                4096       4096             46           0
stat_SPSA1SWA_s4096               4096       4096             46           0
```

The only keys present in a dump but absent from the config are the runtime-only
keys the repository normalises away by convention (`rgb_dir`, `lmdb`, `ddp`,
`dry_run`, `local_rank`, `mode`, `save_avg`, `dataset_json_folder`, `lmdb_dir`).
Those are paths and launch flags, not experimental variables.

## Class C — never run (4 files) — **not evidence for anything**

| file | derived from | seed |
|---|---|---|
| `configs/r34_stat_ctrl_resnet34_s1024.yaml` | `r34_stat_ctrl_resnet34_s2048.yaml` | 1024 |
| `configs/r34_stat_ctrl_resnet34_s4096.yaml` | `r34_stat_ctrl_resnet34_s2048.yaml` | 4096 |
| `configs/r34_stat_SPSA1SWA_resnet34_s1024.yaml` | `r34_stat_SPSA1SWA_resnet34_s2048.yaml` | 1024 |
| `configs/r34_stat_SPSA1SWA_resnet34_s4096.yaml` | `r34_stat_SPSA1SWA_resnet34_s2048.yaml` | 4096 |

No ResNet34 run at any seed other than `manualSeed: 2048` exists in any archive.
These four files exist only so that the ResNet34 arm of the configuration set is
symmetrical with the UCF arm and is runnable as-is. Each carries a header that
begins

```
# !! THIS SEED WAS NEVER RUN. THIS FILE IS NOT EVIDENCE FOR ANY PAPER NUMBER.
```

**No training log exists for any of them and none is claimed.** They must not be
read as support for §4.6's three-seed wording; see
[`r34_backbone_rerun.md`](r34_backbone_rerun.md) and README caveat 8.

## What was never touched

No log was edited, renumbered, truncated or synthesised. No paper number was
back-filled from a configuration. The only modifications made outside pure
addition are the two registry imports appended to
`code/adaptive/networks/__init__.py`, without which the ConvNeXt-Tiny and
SwinV2-Tiny backbone files could not be imported.

## The generator

`scripts/gen_mexp_configs.py` and `scripts/gen_r34m_configs.py` are the original
generators for the mechanism and ResNet34-mechanism batches; their docstrings
record the protocol fixes (`cudnn: false`, `rng_neutral_build: true`,
`swa_ws=[1.0, 0.5, 0.25, 0.125, 0.0625]`, `probe_every=300`,
`test_times_per_epoch=6`, `bn_recalib_batches=100`) and the `tail_lam` ladder.
