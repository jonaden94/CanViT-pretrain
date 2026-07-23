"""Entry point: python -m canvit_pretrain.in1k [tyro flags]."""

import tyro

from .config import In1kConfig
from .train import train


def main() -> None:
    train(tyro.cli(In1kConfig))


if __name__ == "__main__":
    main()
