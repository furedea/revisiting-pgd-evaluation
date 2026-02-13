"""Tests for dataset_config module."""

import pytest

from src.dataset_config import DatasetConfig, resolve_dataset_config


class TestDatasetConfig:
    """Test DatasetConfig DTO."""

    def test_slots_defined(self):
        """DatasetConfig uses __slots__ for memory efficiency."""
        cfg = DatasetConfig(epsilon=0.3, alpha=0.01, model_src_dir="test")
        assert hasattr(cfg, "__slots__")

    def test_fields_set(self):
        """All fields are correctly assigned."""
        cfg = DatasetConfig(
            epsilon=0.3, alpha=0.01, model_src_dir="model_src/mnist_challenge"
        )
        assert cfg.epsilon == 0.3
        assert cfg.alpha == 0.01
        assert cfg.model_src_dir == "model_src/mnist_challenge"

    def test_no_extra_attributes(self):
        """Cannot assign arbitrary attributes due to __slots__."""
        cfg = DatasetConfig(epsilon=0.3, alpha=0.01, model_src_dir="test")
        with pytest.raises(AttributeError):
            cfg.unknown = 42


class TestResolveMnist:
    """Test resolve_dataset_config for MNIST."""

    def test_epsilon(self):
        cfg = resolve_dataset_config("mnist")
        assert cfg.epsilon == 0.3

    def test_alpha(self):
        cfg = resolve_dataset_config("mnist")
        assert cfg.alpha == 0.01

    def test_model_src_dir(self):
        cfg = resolve_dataset_config("mnist")
        assert cfg.model_src_dir == "model_src/mnist_challenge"

    def test_positive_values(self):
        """epsilon and alpha must be positive."""
        cfg = resolve_dataset_config("mnist")
        assert cfg.epsilon > 0
        assert cfg.alpha > 0


class TestResolveCifar10:
    """Test resolve_dataset_config for CIFAR-10."""

    def test_epsilon(self):
        cfg = resolve_dataset_config("cifar10")
        assert cfg.epsilon == pytest.approx(8.0 / 255.0)

    def test_alpha(self):
        cfg = resolve_dataset_config("cifar10")
        assert cfg.alpha == pytest.approx(2.0 / 255.0)

    def test_model_src_dir(self):
        cfg = resolve_dataset_config("cifar10")
        assert cfg.model_src_dir == "model_src/cifar10_challenge"

    def test_positive_values(self):
        """epsilon and alpha must be positive."""
        cfg = resolve_dataset_config("cifar10")
        assert cfg.epsilon > 0
        assert cfg.alpha > 0


class TestResolveUnknownDataset:
    """Test resolve_dataset_config raises ValueError for unknown datasets."""

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown"):
            resolve_dataset_config("unknown")

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_dataset_config("")

    def test_similar_name_raises_value_error(self):
        """Typos should not be silently accepted."""
        with pytest.raises(ValueError):
            resolve_dataset_config("MNIST")

    def test_error_message_contains_dataset_name(self):
        """Error message should include the invalid dataset name."""
        with pytest.raises(ValueError, match="imagenet"):
            resolve_dataset_config("imagenet")
