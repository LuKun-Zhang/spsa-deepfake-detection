# `code/` — four self-contained protocol variants

These are patched files from a fork of [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).
Copy the directory for the protocol you want over your DeepfakeBench checkout.

Only **`train.py`** and **`trainer.py`** differ between protocols; the other 12 files are
byte-identical across all four directories. They are duplicated rather than factored out so that
each directory is complete and runnable on its own — this is the same code state that produced the
paper's numbers, and it is preserved **bug-for-bug** (see caveat 1 in the top-level `README.md`).

| Directory | Protocol | Paper | Run it with |
|---|---|---|---|
| `controlled/` | Controlled 4-cycle 2×2 archive (`nEpochs=3`, four cosine cycles, consistency loss off) | §3.3, §4.2 | `configs/stat_*.yaml` |
| `official-fixed/` | Official 6-round cosine, fixed two-point SWA window (`swa_start=3`) | §4.3 | `configs/*official*cos*.yaml` |
| `adaptive/` | Official 6-round cosine, learning-rate-threshold adaptive SWA window — **the paper's main protocol** | §4.4 | `configs/*_adapt.yaml` |
| `alt-methods/` | Controlled protocol plus the SAM / Lookahead / EMA control arms | §4.2, Table 2 | `configs/stat_sam_*.yaml`, `configs/stat_la_*.yaml`, `configs/stat_ema_*.yaml` |

## File-by-file

| Shared by all four | Role |
|---|---|
| `dataset_abstract_dataset.py`, `dataset_pair_dataset.py` | Dataset patches |
| `loss___init__.py`, `loss_cross_method_consistency.py` | Loss patches (the cross-method consistency loss is switched **off** under the controlled protocol) |
| `spsa_modules.py` | The SPSA module — `spsa` (SPSA1, inside the decoder) and `spsa2` |
| `ucf_detector.py` | The UCF detector: reconstruction decoder, the `H_c` / `H_s` heads, and the SPSA insertion points |

| Present in | Files | Note |
|---|---|---|
| `controlled/` only | `ucf_detector.py.bak_spsa2` | Byte-exact artifact of the doubled `spsa2` application — kept as evidence, **not** used by the run |
| `adaptive/`, `alt-methods/` | `networks/` | EfficientNet-B4 and ResNet-34 backbones (§4.7) |
| `alt-methods/` only | `optimizor/` | `SAM.py`, `LinearLR.py` |

All four `train.py` variants import `from trainer.trainer import Trainer`, i.e. the upstream
DeepfakeBench module (its repo root is on `sys.path`, which is why the patched copy sits here as a
sibling file rather than inside `trainer/`). **`trainer.py` is the patched replacement for that
module**, and it is present only from the official protocol onward — `controlled/` carries none,
because the controlled runs used the stock upstream trainer unpatched, with the SWA window driven
entirely from the config.
