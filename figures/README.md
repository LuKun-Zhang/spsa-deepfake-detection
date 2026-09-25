# `figures/` — figure scripts

The scripts that draw the paper's figures, kept here so each figure's numbers
have a visible source.

Most are **plot-only**: `matplotlib` plus the literals embedded in them. They do
not read the logs, so they cannot silently disagree with them — if a number in a
script and a number in a log ever diverge, the log is the original record.

`animate_mechanism.py` is the exception and is deliberately the other way round:
it **reads the shipped logs** and re-derives every number, refusing to draw if
the logs disagree with the paper. It is the only file here whose output is a
function of the record rather than a transcription of it.

| script | figure | notes |
|---|---|---|
| `animate_mechanism.py` | **the README animation** | reads `logs/mechanism_ucf/*/`; see below |
| `plot_spike.py` | **Fig. 1** (a) SPSA+SWA, (b) matched CTRL | official six-round trajectories, seeds 1024/2048/4096; embeds the 12 trajectories and the best-checkpoint indices; both panels share one y-axis |
| `plot_spsa_arch.py` | **Fig. 2** | the SPSA module diagram (`assets/fig2_pipeline.png` is the full pipeline) |
| `_draw_mech_two_panels.py` | **Fig. 3A** | Bonus and Dev against the four rates; `CTRL_B = [0.08, 0.67, 1.43, 2.18]`, x-labels `1× / 40× / 100× / 200×`. These reproduce `scripts/verify_mechanism.py`'s `bn = true` Bonus column exactly |
| `plot_mech_gap.py`, `plot_mech_traj.py`, `plot_mech_board.py`, `plot_superadd.py`, `plot_controlled.py` | supporting / earlier drafts | trajectory and interaction-term plots from the analysis pass. `plot_controlled.py` draws the Fig. 3B 3-seed mean 7-point trajectories |

The `_`-prefixed filename marks a script that was run for the final figure rather
than kept as a general utility; it is shipped under its original name so the
figure it produced traces back to the exact file.

## `animate_mechanism.py`

Produces the two files the top-level README embeds:

| output | what | why |
|---|---|---|
| `mechanism_animation.svg` | 2 rows (SPSA off / on) × 4 columns (1× / 40× / 100× / 200×), nine frames over a 7.2 s loop | vector, ~97 KB, and it diffs. This is the one the README shows |
| `mechanism_animation.gif` | the same animation, raster | a fallback for viewers that will not play SMIL — the GitHub mobile app shows only the first frame of an SVG |

Both are built from `logs/mechanism_ucf/`, not from literals:

- the six in-window probe AUCs come from each run's `training.log`, parsed with
  the same regex `scripts/verify_mechanism.py` uses;
- `theta_bar` comes from the `w = 1.0`, `bn = true` row of `swa_eval.jsonl`,
  which is the Section 3.3 convention;
- `check()` asserts all eight legs against the spreads and Dev values printed in
  the paper and **exits non-zero rather than draw a figure that disagrees**.

Re-run it after any change to `logs/mechanism_ucf/`:

```bash
python figures/animate_mechanism.py
```

The revealed quantity is amplitude, so every panel shares one y-axis — a shared
axis is the honest choice here and also the whole point, since the claim is that
SPSA holds the amplitude down. The y-axis is `AUC − the window's own mean`, in
points, so the dashed zero line *is* the mean, `Dev` is where the endpoint lands
on it, `Bonus` is where the average lands, and the orange bar between them is
`Delta = Bonus − Dev` read directly off the scale.
