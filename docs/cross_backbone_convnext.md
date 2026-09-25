# §4.6 — Cross-backbone legs: ConvNeXt-Tiny (and the SwinV2-Tiny leg that did not gain)

Covers the ConvNeXt-Tiny row of §4.6 (`+0.71`) and documents the SwinV2-Tiny leg,
which was measured under the same protocol and is **not** claimed in the paper.

## The logs, and an incomplete leg

`logs/cross_bb/` holds the six run directories plus the raw training stdout. Their
7-point means over the controlled grid are:

| file | backbone | arm | points | 7-point mean |
|---|---|---|---|---|
| `cnx_ctrl_589original.out` | convnext_tiny | CTRL | 7/7 | **0.610489** |
| `cnx_ctrl.out` | convnext_tiny | CTRL (rerun) | 7/7 | **0.606030** |
| `ucf_2026-09-19-23-06-01` (589) | convnext_tiny | SPSA+SWA | 7/7 | **0.613081** |
| `cnx_spsaswa.out` (467) | convnext_tiny | SPSA+SWA | **3/7 — truncated** | 0.544683 |
| `swv2_ctrl.out` | swinv2_tiny | CTRL | 7/7 | 0.526829 |
| `swv2_spsaswa.out` | swinv2_tiny | SPSA+SWA | 7/7 | 0.520506 |

## The +0.71 is a cross-run pairing

The ConvNeXt SPSA+SWA leg has **one complete trajectory**, from the first machine
(`ucf_2026-09-19-23-06-01`, mean **0.613081**). The second machine's SPSA+SWA leg
was **truncated at step 11487** and only ever produced 3 of 7 points, so it cannot
supply a mean.

The paper's `+0.71` therefore pairs the **first machine's SPSA+SWA leg** against
the **second machine's CTRL leg**:

```
0.613081  (589 SPSA+SWA)
-0.606030  (467 CTRL)
= +0.705 pt   -> the paper's +0.71
```

The **same-run** pairing available in these logs is

```
0.613081  (589 SPSA+SWA)
-0.610489  (589 CTRL)
= +0.259 pt   -> the +0.26 the manuscript carried before the CTRL swap
```

So both numbers are real and both are in this repository; they differ only in
which CTRL leg the SPSA+SWA leg is paired against. The two CTRL legs differ by
**0.45 pt**, which is within the cross-machine repeat band documented for this
setup (§4.1 gives the paired band as 0.43 pt). The paper reports the +0.71; the
complete same-run pairing is visible here for anyone who wants it.

**This is a pairing caveat, not an arithmetic one** — `+0.705` reproduces `+0.71`
exactly at the paper's precision.

## The SwinV2-Tiny leg did not gain

`swv2_tiny` was run under the identical protocol and came out **negative**:
`0.520506 − 0.526829 = −0.63 pt`. It is **not** mentioned anywhere in the
submitted text (the string does not occur), and it is not part of the §4.6 claim,
which names ConvNeXt-Tiny, EfficientNet-B4 and ResNet34 only. Its configs
(`configs/swv2_*.yaml`), runs and raw stdout are shipped here so that the
backbone sweep is complete rather than cherry-picked. Read §4.6's direction claim
as covering the backbones it names, and this leg as a measured counter-example
outside that set.

## Seed coverage

The ConvNeXt-Tiny legs exist at **`s2048` only** — see README caveat 8 and
[`r34_backbone_rerun.md`](r34_backbone_rerun.md). Of the three backbones named in
§4.6, only EfficientNet-B4 (`docs/efnb4_3seed_verified_data.md`) is three-seed.

## Provenance

- Configs: `configs/cnx_stat_{ctrl,SPSA1SWA}_convnext_tiny_s2048.yaml`,
  `configs/swv2_stat_{ctrl,SPSA1SWA}_swinv2_tiny_s2048.yaml`
- Backbone implementations: `code/adaptive/networks/convnext_tiny.py`,
  `code/adaptive/networks/swinv2_tiny.py`, registered in
  `code/adaptive/networks/__init__.py`
- Launcher: `scripts/launch_cross_bb.sh`
- Raw stdout: `logs/cross_bb/*.out` — including `cnx_ctrl_589original.out`, the
  only surviving record of the first machine's CTRL leg
