"""Logging configuration for PGD visualization."""

import logging

LOGGER_NAME = "pgd_loss_curves"
LOGGER = logging.getLogger(LOGGER_NAME)


def setup_logging() -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
