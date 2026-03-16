"""ADE20K probe training.

Evaluation is now in the canvit-eval repo (~/code/CanViT-eval).

Usage:
    python -m canvit_eval.ade20k train ...              # CanViT canvas probes
    python -m canvit_eval.ade20k train-dinov3-probe ... # DINOv3 baseline probe
"""

from typing import Annotated

import tyro

from canvit_eval.ade20k.train_dinov3_probe import DINOv3ProbeTrainConfig
from canvit_eval.ade20k.train_dinov3_probe import train as run_train_dinov3
from canvit_eval.ade20k.train_probe import train as run_train
from canvit_eval.ade20k.train_probe.config import Config as TrainConfig


def main() -> None:
    cmd = tyro.cli(
        Annotated[TrainConfig, tyro.conf.subcommand("train")]
        | Annotated[DINOv3ProbeTrainConfig, tyro.conf.subcommand("train-dinov3-probe")]
    )
    match cmd:
        case TrainConfig():
            run_train(cmd)
        case DINOv3ProbeTrainConfig():
            run_train_dinov3(cmd)


if __name__ == "__main__":
    main()
