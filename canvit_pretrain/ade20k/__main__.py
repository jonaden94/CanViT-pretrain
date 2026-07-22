"""Entry point: python -m canvit_pretrain.ade20k [tyro flags]."""

import tyro

from .config import Ade20kConfig
from .train import train


def main() -> None:
    train(tyro.cli(Ade20kConfig))


if __name__ == "__main__":
    main()
