"""ADE20K probe training and evaluation.

Usage:
    python -m canvit_eval.ade20k train ...                         # CanViT canvas probes
    python -m canvit_eval.ade20k evaluate --probe-repo ...         # CanViT probe eval
    python -m canvit_eval.ade20k train-dinov3-probe ...            # DINOv3 baseline probe
    python -m canvit_eval.ade20k eval-dinov3-probe --probe-repo .. # DINOv3 probe eval
"""

from typing import Annotated

import tyro

from canvit_eval.ade20k.eval_dinov3_probe import DINOv3ProbeEvalConfig
from canvit_eval.ade20k.eval_dinov3_probe import evaluate as run_eval_dinov3
from canvit_eval.ade20k.evaluate import EvalConfig
from canvit_eval.ade20k.evaluate import evaluate as run_evaluate
from canvit_eval.ade20k.train_dinov3_probe import DINOv3ProbeTrainConfig
from canvit_eval.ade20k.train_dinov3_probe import train as run_train_dinov3
from canvit_eval.ade20k.train_probe import train as run_train
from canvit_eval.ade20k.train_probe.config import Config as TrainConfig


def main() -> None:
    cmd = tyro.cli(
        Annotated[TrainConfig, tyro.conf.subcommand("train")]
        | Annotated[EvalConfig, tyro.conf.subcommand("evaluate")]
        | Annotated[DINOv3ProbeTrainConfig, tyro.conf.subcommand("train-dinov3-probe")]
        | Annotated[DINOv3ProbeEvalConfig, tyro.conf.subcommand("eval-dinov3-probe")]
    )
    match cmd:
        case TrainConfig():
            run_train(cmd)
        case EvalConfig():
            run_evaluate(cmd)
        case DINOv3ProbeTrainConfig():
            run_train_dinov3(cmd)
        case DINOv3ProbeEvalConfig():
            run_eval_dinov3(cmd)


if __name__ == "__main__":
    main()
