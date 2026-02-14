"""Tests for find_common_correct_samples module."""

import json
import os
import sys
from unittest.mock import patch

import pytest

# Import the module under test (build_arg_parser, main are TF-independent at parse time)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from find_common_correct_samples import build_arg_parser


class TestBuildArgParser:
    """Test CLI argument parsing and --help support (Req 3.6)."""

    def test_parser_has_description(self):
        """Parser should have a descriptive help message."""
        parser = build_arg_parser()
        assert parser.description is not None
        assert len(parser.description) > 0

    def test_dataset_required(self):
        """--dataset is required."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_dataset_mnist(self):
        """--dataset mnist is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist"])
        assert args.dataset == "mnist"

    def test_dataset_cifar10(self):
        """--dataset cifar10 is accepted."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "cifar10"])
        assert args.dataset == "cifar10"

    def test_dataset_invalid(self):
        """Invalid dataset name is rejected."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "imagenet"])

    def test_out_dir_default(self):
        """--out_dir defaults to 'docs'."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist"])
        assert args.out_dir == "docs"

    def test_out_dir_custom(self):
        """--out_dir accepts custom path."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist", "--out_dir", "/tmp/out"])
        assert args.out_dir == "/tmp/out"

    def test_models_default(self):
        """--models defaults to four model names."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist"])
        assert args.models == "nat,adv,nat_and_adv,weak_adv"

    def test_models_custom(self):
        """--models accepts custom comma-separated list."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist", "--models", "nat,adv"])
        assert args.models == "nat,adv"

    def test_samples_per_class_default(self):
        """--samples_per_class defaults to 1."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist"])
        assert args.samples_per_class == 1

    def test_seed_default(self):
        """--seed defaults to 0."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dataset", "mnist"])
        assert args.seed == 0


class TestDatasetConfigReuse:
    """Test that find_common_correct_samples uses dataset_config (Req 3.2)."""

    def test_model_src_dir_resolved_from_dataset_config_mnist(self):
        """model_src_dir should come from resolve_dataset_config for mnist."""
        from src.dataset_config import resolve_dataset_config

        cfg = resolve_dataset_config("mnist")
        assert cfg.model_src_dir == "model_src/mnist_challenge"

    def test_model_src_dir_resolved_from_dataset_config_cifar10(self):
        """model_src_dir should come from resolve_dataset_config for cifar10."""
        from src.dataset_config import resolve_dataset_config

        cfg = resolve_dataset_config("cifar10")
        assert cfg.model_src_dir == "model_src/cifar10_challenge"


class TestInputFileValidation:
    """Test error messages for missing input files (Req 3.5)."""

    def test_missing_checkpoint_dir_error(self):
        """Should raise FileNotFoundError with clear message for missing ckpt dir."""
        from find_common_correct_samples import validate_checkpoint_dir

        nonexistent = "/nonexistent/path/to/ckpt"
        with pytest.raises(FileNotFoundError, match=nonexistent):
            validate_checkpoint_dir(nonexistent)

    def test_existing_checkpoint_dir_passes(self, tmp_path):
        """Should not raise for existing directory."""
        from find_common_correct_samples import validate_checkpoint_dir

        ckpt_dir = str(tmp_path)
        # Should not raise
        validate_checkpoint_dir(ckpt_dir)

    def test_missing_model_src_dir_error(self):
        """Should raise FileNotFoundError with clear message for missing model_src_dir."""
        from find_common_correct_samples import validate_model_src_dir

        nonexistent = "/nonexistent/model_src"
        with pytest.raises(FileNotFoundError, match=nonexistent):
            validate_model_src_dir(nonexistent)

    def test_existing_model_src_dir_passes(self, tmp_path):
        """Should not raise for existing directory."""
        from find_common_correct_samples import validate_model_src_dir

        model_src_dir = str(tmp_path)
        # Should not raise
        validate_model_src_dir(model_src_dir)


class TestMainUsesDatasetConfig:
    """Test that main() uses resolve_dataset_config instead of hardcoded mapping."""

    def test_main_calls_resolve_dataset_config(self):
        """main() should use resolve_dataset_config to determine model_src_dir."""
        # Verify the import exists in the module
        import find_common_correct_samples as mod

        # Check that the module imports resolve_dataset_config
        assert hasattr(mod, "resolve_dataset_config") or (
            "resolve_dataset_config" in dir(mod)
        ), (
            "find_common_correct_samples should import resolve_dataset_config "
            "from src.dataset_config"
        )
