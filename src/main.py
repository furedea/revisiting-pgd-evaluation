"""
PGD loss/correctness visualization (MNIST/CIFAR10, 1..5 panels) with optional
DeepFool init.

- Python 3.6.9 compatible.
- No external libs beyond: numpy, matplotlib, tensorflow.
"""

import matplotlib
import tensorflow as tf

matplotlib.use("Agg")
tf.compat.v1.disable_eager_execution()

from .cli import build_arg_parser, validate_args
from .logging_config import setup_logging
from .pipeline import run_pipeline


def main() -> None:
    """Entry point for PGD visualization."""
    setup_logging()
    args = build_arg_parser().parse_args()
    validate_args(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
