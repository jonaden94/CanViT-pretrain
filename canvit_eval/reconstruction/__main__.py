"""CLI entry point for reconstruction evaluation."""

import logging

import tyro

from canvit_eval.reconstruction import ReconstructionEvalConfig, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

evaluate(tyro.cli(ReconstructionEvalConfig))
