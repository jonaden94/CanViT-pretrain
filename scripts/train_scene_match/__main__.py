"""Entry point for curriculum scene matching training."""

import logging
from dataclasses import replace

import optuna
import torch
import tyro

from .config import Config
from .train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)

    log.info("=" * 60)
    log.info("Scene Matching Training with Curriculum")
    log.info("=" * 60)
    log.info(f"Config: {cfg}")
    log.info(f"Device: {cfg.device}")
    log.info(f"Total steps: {cfg.n_steps:,}")

    # Log schedule
    log.info(f"PROBE: {cfg.probe_steps:,} steps (largest→smallest, {cfg.probe_ramp_ratio:.0%} ramp)")
    log.info(f"MAIN: {cfg.main_steps:,} steps (smallest→largest, {cfg.main_ramp_steps} ramp steps)")
    for entry in cfg.get_schedule():
        log.info(
            f"  [{entry.phase:5}] G={entry.grid_size}: "
            f"steps {entry.start_step:,}-{entry.end_step:,} "
            f"(ramp={entry.lr_ramp_steps}, decay={entry.lr_decay_steps})"
        )

    log.info("=" * 60)

    def objective(trial: optuna.Trial) -> float:
        ref_lr = trial.suggest_float("ref_lr", 1e-6, 1e-2, log=True)
        train_cfg = replace(cfg, ref_lr=ref_lr)
        return train(train_cfg, trial)

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"ref_lr": cfg.ref_lr})
    study.optimize(objective, n_trials=cfg.n_trials)

    log.info("=" * 60)
    log.info(f"Best trial: {study.best_trial.params}")
    log.info(f"Best val_loss: {study.best_value:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
