"""ADE20K probe training and evaluation.

Usage:
    python -m canvit_eval.ade20k train --ade20k-root /path/to/ade20k ...
    python -m canvit_eval.ade20k evaluate --probe-ckpt /path/to/best.pt --ade20k-root /path/to/ade20k ...
"""

from typing import Annotated

import tyro

from canvit_eval.ade20k.evaluate import EvalConfig
from canvit_eval.ade20k.evaluate import evaluate as run_evaluate
from canvit_eval.ade20k.train_probe import train as run_train
from canvit_eval.ade20k.train_probe.config import Config as TrainConfig


def main() -> None:
    cmd = tyro.cli(
        Annotated[TrainConfig, tyro.conf.subcommand("train")]
        | Annotated[EvalConfig, tyro.conf.subcommand("evaluate")]
    )
    match cmd:
        case TrainConfig():
            run_train(cmd)
        case EvalConfig():
            run_evaluate(cmd)


if __name__ == "__main__":
    main()
