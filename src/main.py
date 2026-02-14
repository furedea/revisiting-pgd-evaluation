"""
PGD loss/correctness visualization (MNIST/CIFAR10, 1..5 panels) with optional
DeepFool init.

- Python 3.6.9 compatible.
- No external libs beyond: numpy, matplotlib, tensorflow.
"""

import sys
from pathlib import Path

# Add parent directory to path for direct execution
_src_dir = Path(__file__).resolve().parent
if str(_src_dir.parent) not in sys.path:
    sys.path.insert(0, str(_src_dir.parent))

import matplotlib
import tensorflow as tf

matplotlib.use("Agg")
tf.compat.v1.disable_eager_execution()

from src.cli import build_arg_parser, validate_args
from src.logging_config import setup_logging
from src.pipeline import run_pipeline


def main() -> None:
    """Entry point for PGD visualization."""
    setup_logging()
    args = build_arg_parser().parse_args()
    validate_args(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
