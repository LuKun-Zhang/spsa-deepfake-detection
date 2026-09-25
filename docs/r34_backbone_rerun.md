# §4.6 — The ResNet34 legs: rerun, reporting convention, and seed coverage

This document covers the ResNet34 rows of §4.6 and the ResNet34 half of the
mechanism identity (`Bonus`, `Dev`, `Delta`). The ResNet34 mechanism numbers are
re-derived offline by

```bash
python scripts/verify_mechanism.py     # the two `mechanism_r34` rows
```

## What the paper reports

> On three other UCF-style backbones the gain remains positive (+0.71 on
> ConvNeXt-Tiny, +1.41 on EfficientNet-B4, and +1.71 on ResNet34, all three
> controlled seeds). … In the module arm Bonus = +0.07 pt and Dev = −0.77 pt, so
> Δ closes at +0.84 pt, and the module deepens the endpoint's gap below the window
> mean relative to SWA alone (Dev = −0.42 pt).

Reproduced from `logs/mechanism_r34/`:

| run | backbone | arm | spread | Dev | Bonus (`bn=F`) | Δ |
|---|---|---|---|---|---|---|
| `ucf_2026-09-21-00-35-35` | resnet34 | SPSA module | +1.8922 | **−0.7675** | **+0.0770** | **+0.8445** |
| `ucf_2026-09-21-07-51-21` | resnet34 | SWA only | +0.6731 | **−0.4193** | −0.3570 | +0.3247 |

`−0.77`, `+0.07`, `+0.84` and `−0.42` all reproduce. ✓

## Reporting convention: `bn = false`

The ResNet34 legs are read off the **frozen-buffer** column, and this is a
deliberate difference from §3.3. On this run the two columns are
`+0.0770` (`bn = false`) and `+0.1008` (`bn = true`) — the paper quotes the
former. The reason is protocol fidelity rather than selection: the R34 mechanism
legs were run to explain the **archived** R34 result, and the archived R34
protocol evaluated the averaged weights with the endpoint's BatchNorm buffers
frozen. Mirroring it keeps `Dev` and `Bonus` on the same footing as the run they
were built to explain. On UCF the opposite convention is used, for the structural
reason given in [`mechanism_endpoint_position.md`](mechanism_endpoint_position.md).

`scripts/verify_mechanism.py` prints both columns for every run and asserts the
ResNet34 rows against the frozen values, so the choice is auditable rather than
implicit.

## Provenance of the +1.71

The archived ResNet34 comparison was **negative**: an earlier batch of R34 runs
put the SPSA+SWA arm *below* its matched CTRL on all seven points. The +1.71 stems
from the **2026-09-21 controlled rerun** whose legs are shipped here, and it
replaced the archived number in the manuscript.

**The rerun changed more than one variable at a time** — it enabled the protocol
fixes (`cudnn: false`, `rng_neutral_build`, a denser test grid, and the probe
instrumentation) together. The sign change therefore **cannot be attributed to
any single one of them**, and the paper does not attempt to. What the rerun
establishes is the value of the ResNet34 leg under the *protocol used throughout
the rest of the paper*; it does not, on its own, explain the archived reading.

Runs not shipped (see README caveat 7):

- `ucf_2026-09-21-04-34-51` — a truncated duplicate of the module-arm run.
- `ucf_2026-09-21-11-22-53` — killed after two test points.

## ⚠️ Seed coverage

**The ResNet34 legs exist at a single controlled seed, `s2048`.** No `s1024` or
`s4096` ResNet34 run exists in any archive — not on the machine that produced
these logs, and not in the earlier salvage sets. Every R34 training run in
existence, including the 2026-09-03/04 archive pairs, carries `manualSeed: 2048`.

`configs/r34_stat_{ctrl,SPSA1SWA}_resnet34_s{1024,4096}.yaml` are present so the
configuration set is complete, but they carry a header stating
`THIS SEED WAS NEVER RUN` — see
[`config_generation_manifest.md`](config_generation_manifest.md).

This is stated in the README as caveat 8 and applies equally to ConvNeXt-Tiny:
of the three backbones named in §4.6, **only EfficientNet-B4 is a genuine
three-seed result** (`docs/efnb4_3seed_verified_data.md`).

## Provenance

- Code: `code/mechanism/` (instrumented build) + `code/mechanism/networks/resnet34.py`
- Configs: `configs/r34m_{ctrl,spsa1swa,spsaonly,swaonly}_s2048.yaml`,
  `configs/r34_stat_*_resnet34_s2048.yaml`
- Launcher: `scripts/launch_r34m.sh`; report generator: `scripts/r34m_report.py`
- Config generator: `scripts/gen_r34m_configs.py`
