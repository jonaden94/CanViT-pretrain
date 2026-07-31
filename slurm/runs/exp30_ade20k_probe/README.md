# exp30 — ADE20K frozen-probe training (harness, current code)

Four frozen-probe runs, one per exp22 pretrained source. Recipe copied from **exp24** = the
original `canvit_specialize` ade20k probe, reproduced by the harness `ade20k` defaults
(frozen backbone via the default `probe` preset, 40k steps, random-view training,
`n_timesteps 10`, scene 512, `canvas_grid 32`).

Sources are the same four as exp29 (see its README; `ade20k-fovi-1901k` is the new one).
The foveated arms pass `--cfg.foveated-scale.fixed-scale 2.0`.

**`resize_mode=squish` for every arm, including foveated.** That is the protocol every earlier
CanViT / specialize number was measured under, so it is what makes these comparable to exp24
and to the published values. It distorts aspect ratio, so it is *not* the right choice for a
human-viewing comparison — `center_crop` preserves the geometry foveated sampling assumes, at
the cost of cropping the long side. Whichever is used has to be reported with the number.

Note: ADE20K train-mIoU (`train_miou_mean`, `best_val_miou_t{t}`) is **deliberately not
logged** by the harness (owner decision, 2026-07-31 — see
`unification_docs/17-harness-consolidation.md`). Train loss and per-timestep val mIoU are
unaffected. Old pre-2026-07-31 ade20k wandb runs have `train_miou_mean` panels these will not.
