"""Entry point for CanViT pretraining."""

# CRITICAL (DDP-safety): redirect matplotlib config + cache to a per-rank,
# per-job /tmp directory BEFORE any matplotlib import happens anywhere in the
# dep tree. The default ~/.config/matplotlib + ~/.cache/matplotlib paths are
# on shared NFS, and concurrent reads/writes from multiple DDP ranks (and from
# concurrent jobs on the same node) race on the font cache file, leaving it in
# a state that makes subsequent matplotlib calls hang indefinitely. Also force
# the non-interactive Agg backend to skip any GUI/display probing.
import os as _os
_slurm_rank = _os.environ.get("SLURM_PROCID", "0")
_slurm_job = _os.environ.get("SLURM_JOB_ID", "nojob")
_mpl_dir = f"/tmp/mpl_config_rank{_slurm_rank}_job{_slurm_job}"
_os.makedirs(_mpl_dir, exist_ok=True)
_os.environ["MPLCONFIGDIR"] = _mpl_dir
import matplotlib as _matplotlib  # noqa: E402
_matplotlib.use("Agg")

import logging
from dataclasses import replace

import optuna
import torch
import tyro

from .config import Config
from .loop import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)

    log.info("=" * 60)
    log.info("CanViT Pretraining")
    log.info("=" * 60)
    log.info(f"Config: {cfg}")
    log.info(f"Device: {cfg.device}")
    log.info(f"Steps per job: {cfg.steps_per_job:,}")
    log.info(f"Canvas patch grid size: {cfg.canvas_patch_grid_size}")
    log.info(f"Warmup: {cfg.warmup_steps} steps")
    log.info("=" * 60)

    def objective(trial: optuna.Trial) -> float:
        peak_lr = trial.suggest_float("peak_lr", 1e-6, 1e-2, log=True)
        train_cfg = replace(cfg, peak_lr=peak_lr)
        return train(train_cfg, trial)

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"peak_lr": cfg.peak_lr})
    study.optimize(objective, n_trials=cfg.n_trials)

    log.info("=" * 60)
    log.info(f"Best trial: {study.best_trial.params}")
    log.info(f"Best val_loss: {study.best_value:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
