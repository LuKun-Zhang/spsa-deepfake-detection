# §3.3 — The SPSA module and endpoint position: what was run, and how to re-derive it

This document covers the rate sweep behind the §3.3 mechanism claim and
`figures/_draw_mech_two_panels.py` (Fig. 3A). Every number below is reproduced
offline, from the artifacts in this repository, by

```bash
python scripts/verify_mechanism.py
```

which prints the table and exits non-zero if any asserted value disagrees with
the paper.

## The window

SWA in the controlled protocol averages weights at the **endpoints of the final
two cycles**. The mechanism experiment instead opens the **whole final cycle**
and probes it six times:

```
window   steps 17232 .. 22976
probes   18182  19139  20096  21053  22010  22967
```

and reports, against the window's **own mean**:

| quantity | definition | reads |
|---|---|---|
| `spread` | `max AUC - min AUC` over the six probes | §3.3 "window spread" |
| `Dev` | `AUC(theta_end) - mean6` | §3.3 "endpoint position" |
| `Bonus` | `AUC(theta_bar) - mean6` | §3.3 "what averaging buys" |
| `Delta` | `Bonus - Dev` = `AUC(theta_bar) - AUC(theta_end)` | the identity |

## `tail_lam`: the four rates

`tail_lam` clamps the learning rate **inside the window**, as a fraction of
`lr_max = 2e-4`. `lam = 0` leaves the schedule on its own `lr = 1e-6`, so the
four rungs of `configs/mexp_p_*.yaml` are

| config suffix | `tail_lam` | lr in the window | label |
|---|---|---|---|
| `l0` | 0.0 | 1e-6 | **1×** |
| `l20` | 0.2 | 8e-5 | **40×** |
| `l50` | 0.5 | 2e-4 | **100×** |
| `l100` | 1.0 | 2e-4 (no clamp) | **200×** |

Two arms: `mexp_p_ctrl_*` (SPSA off) and `mexp_p_spsa_*` (SPSA on). The sweep is
**zero-training** to read — it is pure analysis of the eight logs shipped in
`logs/mechanism_ucf/`.

## The result

Full-precision output of `scripts/verify_mechanism.py`, `bn = true` column:

| arm | rate | spread | Dev | Bonus | Delta |
|---|---|---|---|---|---|
| CTRL | 1× | +0.7915 | −0.2679 | +0.0766 | +0.3445 |
| CTRL | 40× | +7.2194 | +0.0954 | +0.6743 | +0.5788 |
| CTRL | 100× | +5.0712 | +2.5326 | +1.4305 | −1.1022 |
| CTRL | 200× | +8.6896 | +3.2065 | +2.1819 | −1.0246 |
| SPSA | 1× | +1.0177 | −0.0698 | +0.1449 | +0.2147 |
| SPSA | 40× | +2.6590 | −1.3296 | +0.3244 | +1.6540 |
| SPSA | 100× | +3.1954 | −0.6860 | +0.7026 | +1.3886 |
| SPSA | 200× | +4.2796 | −0.1521 | +0.8524 | +1.0044 |

Read off directly against the paper:

- **Window spread** — the control's grows `0.79 → 7.22 → 5.07 → 8.69`; SPSA holds
  it at `1.02 → 2.66 → 3.20 → 4.28`. ✓
- **Bonus is positive on both arms at every rate.** ✓
- **Dev ≤ 0 on the module arm at every rate**, and `Delta < 0` occurs only on the
  control, between 40× and 100×. ✓

## Which `theta_bar` column, and why

`swa_eval.jsonl` stores `AUC(theta_bar)` twice: with the BatchNorm buffers
re-estimated on the training set (`bn = true`) and with the endpoint's buffers
frozen (`bn = false`). The two are not interchangeable, and the choice is not
cosmetic — on the control arm at 200×, `bn = true` gives **+2.18** and
`bn = false` gives **−3.03**.

**§3.3 reads the `bn = true` column.** The reason is structural, not selective:
the window probes are test points on the target distribution, and `theta_bar` is
an average of weights trained across a learning-rate range that spans 200×. Its
frozen BatchNorm statistics belong to the endpoint, not to the average, so the
frozen-buffer score is measuring a model whose normalization is mis-set — which
is exactly what the −3.03 shows. Re-estimating the buffers is what makes
`theta_bar` a model at all. `scripts/verify_mechanism.py` prints both columns so
the reader can check this.

**§4.6 (ResNet34) reads the `bn = false` column**, because that leg mirrored the
archived R34 protocol, which froze the buffers — see
[`r34_backbone_rerun.md`](r34_backbone_rerun.md).

## Runs that are *not* shipped

- `ucf_2026-09-16-21-34-05` — a run before the probe instrumentation was fixed;
  its `probes.jsonl` is empty, so it is not evidence for anything.
- `ucf_2026-09-17-17-51-30` — the `l0_nz` variant (`configs/mexp_p_ctrl_l0_nz.yaml`),
  which removes the zero-noise baseline from the `l0` rung. It is a control on the
  control and is not quoted in the paper.

Both configurations are shipped; only their logs are omitted, under the stated
rule in the README (caveat 7).

## Provenance

- Code: `code/mechanism/` — the **instrumented** build. Its `networks/resnet34.py`
  and `networks/efficientnetb4.py` are **not** byte-identical to the plain copies
  elsewhere: they carry the probe hooks.
- Config generator: `scripts/gen_mexp_configs.py`
- Launcher: `scripts/launch_mexp.sh`
- Probe re-scoring: `scripts/eval_probe_sweep8.py` and
  `scripts/eval_probe_sweep_r34.py`, driven by `configs/eval_probe_*.yaml`. These
  are what turn a stored checkpoint into the six in-window probe AUCs that
  `training.log` reports; they run offline against a checkpoint and are the only
  way to regenerate `probes.jsonl`.
- Verifier: `scripts/verify_mechanism.py`
