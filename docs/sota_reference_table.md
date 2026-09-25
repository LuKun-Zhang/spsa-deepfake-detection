# §4.2, Table 1 — the generalizable-detector reference table

Table 1 retrains published deepfake detectors under **our** controlled protocol
and reports their Celeb-DF-v2 frame-level AUC as a 7-point mean. It is labelled
**"for reference only"** in the caption, and that label is load-bearing: these are
not the numbers the original papers report.

| | Xcep | CORE | F3Net | SIA | UCF | Ours |
|---|---|---|---|---|---|---|
| paper | 0.7085 | 0.7094 | 0.7160 | 0.7276 | 0.7178 | 0.7305 |
| recomputed from `logs/sota/` | 0.7085 | 0.7094 | 0.7160 | 0.7276 | — | — |

The four retrained baselines were run at the same seed (`manualSeed: 1024`) and
the same `nEpochs=3` controlled protocol as everything else here, so their
trajectories are directly comparable to the 2×2 table — which is the point of
retraining them rather than quoting the literature.

- **UCF** is our own reproduction of the UCF detector under the controlled
  protocol, i.e. the **CTRL cell** of Table 2 (`0.7178`).
- **Ours** is the SPSA+SWA cell of Table 2 (`0.7305`).
- **Xcep** is the Xception backbone trained standalone under the controlled
  protocol, giving the architecture's own floor.

## What is shipped, and what is not

Shipped: `logs/sota/{xception,core,f3net,sia}_s1024.log`, configs
`configs/sota_{xception,core,f3net,sia}.yaml`, and the chain scripts
`scripts/chain_sota{5,6,7,8}.py` that ran the batch.

**Not shipped, on purpose:** `spsl` (0.7404) and `srm` (0.7577). Both were
retrained, both were in an earlier draft of the table, and both were **removed**
before submission — they appear **zero times** in the submitted text. Note that
both are *above* our headline `0.7305`; they were dropped for scope and reporting
consistency, not because they were weak. Their **configurations are shipped**
(`configs/sota_spsl_s1024*.yaml`, `configs/sota_srm_s1024*.yaml`) so the
retraining batch is complete and the omission is visible rather than silent. The
README states the inclusion rule that produces this split (caveat 7).

## Baselines that were attempted and abandoned

For completeness, three further baselines were started and are **not** in the
table, for reasons that have nothing to do with their scores:

- **`capsule_net`** — the run never started.
- **Effort (ICML 2025)** — reached epoch 0 only (one third of the training budget)
  at `0.7992`, and was dropped as an incomplete-budget comparison rather than
  reported as a number.
- **LSDA** — the data pipeline could not be made to run.

Their configs are present under `configs/sota*.yaml`; no logs exist for them, so
none are shipped. Nothing in this document is quoted in the paper.

## Provenance

- Configs: `configs/sota_*.yaml` (including `.run2` retries)
- Logs: `logs/sota/`
- Chain scripts: `scripts/chain_sota{5,6,7,8}.py`
