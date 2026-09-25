# `figures/` — figure scripts

The scripts that draw the paper's figures, kept here so each figure's numbers
have a visible source. They are **plot-only**: `matplotlib` and the literal 7-point
trajectories embedded in them. They do not read the logs, so they cannot silently
disagree with them — if a number in a script and a number in a log ever diverge,
the log is the original record.

| script | figure | notes |
|---|---|---|
| `plot_spike.py` | **Fig. 1** (a) SPSA+SWA, (b) matched CTRL | official six-round trajectories, seeds 1024/2048/4096; embeds the 12 trajectories and the best-checkpoint indices; both panels share one y-axis |
| `plot_spsa_arch.py` | **Fig. 2** | the SPSA module diagram (`assets/fig2_pipeline.png` is the full pipeline) |
| `_draw_mech_two_panels.py` | **Fig. 3A** | Bonus and Dev against the four rates; `CTRL_B = [0.08, 0.67, 1.43, 2.18]`, x-labels `1× / 40× / 100× / 200×`. These reproduce `scripts/verify_mechanism.py`'s `bn = true` Bonus column exactly |
| `plot_mech_gap.py`, `plot_mech_traj.py`, `plot_mech_board.py`, `plot_superadd.py`, `plot_controlled.py` | supporting / earlier drafts | trajectory and interaction-term plots from the analysis pass. `plot_controlled.py` draws the Fig. 3B 3-seed mean 7-point trajectories |

The `_`-prefixed filename marks a script that was run for the final figure rather
than kept as a general utility; it is shipped under its original name so the
figure it produced traces back to the exact file.
