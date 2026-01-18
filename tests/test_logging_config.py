"""Tests for logging_config module."""

import logging

from src.logging_config import LOGGER, LOGGER_NAME, setup_logging


class TestLoggingConfig:
    def test_logger_name(self):
        assert LOGGER_NAME == "pgd_loss_curves"

    def test_logger_instance(self):
        assert isinstance(LOGGER, logging.Logger)
        assert LOGGER.name == LOGGER_NAME

    def test_setup_logging_does_not_raise(self):
        setup_logging()

    def test_setup_logging_sets_level(self):
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
