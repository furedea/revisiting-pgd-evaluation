"""Tests for cli module."""

import argparse

import pytest

from src.cli import (
    build_arg_parser,
    format_base_name,
    format_title,
    get_model_tag,
    validate_args,
)


class TestBuildArgParser:
    def test_returns_argument_parser(self):
        parser = build_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_required_arguments(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_minimal_args(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model_src_dir", "/path/to/model",
            "--ckpt_dir", "/path/to/ckpt",
            "--out_dir", "/path/to/out",
            "--exp_name", "test_exp",
            "--epsilon", "0.3",
            "--alpha", "0.01",
        ])
        assert args.dataset == "mnist"
        assert args.epsilon == 0.3
        assert args.alpha == 0.01

    def test_default_values(self):
        parser = build_arg_parser()
        args = parser.parse_args([
            "--dataset", "cifar10",
            "--model_src_dir", "/path",
            "--ckpt_dir", "/path",
            "--out_dir", "/path",
            "--exp_name", "test_exp",
            "--epsilon", "0.031",
            "--alpha", "0.007",
        ])
        assert args.start_idx == 0
        assert args.n_examples == 1
        assert args.total_iter == 100
        assert args.num_restarts == 20
        assert args.init == "random"
        assert args.df_project == "clip"

    def test_init_choices_include_multi_deepfool(self):
        """--init accepts multi_deepfool as a valid choice."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model_src_dir", "/path",
            "--ckpt_dir", "/path/to/ckpt/nat",
            "--out_dir", "/path/to/out",
            "--exp_name", "test_exp",
            "--epsilon", "0.3",
            "--alpha", "0.01",
            "--init", "multi_deepfool",
        ])
        assert args.init == "multi_deepfool"

    def test_all_init_choices_accepted(self):
        """All four init methods are accepted by the parser."""
        parser = build_arg_parser()
        for init_method in ["random", "deepfool", "multi_deepfool", "clean"]:
            args = parser.parse_args([
                "--dataset", "mnist",
                "--model_src_dir", "/path",
                "--ckpt_dir", "/path/to/ckpt/nat",
                "--out_dir", "/path/to/out",
                "--exp_name", "test_exp",
                "--epsilon", "0.3",
                "--alpha", "0.01",
                "--init", init_method,
            ])
            assert args.init == init_method

    def test_invalid_init_choice_rejected(self):
        """Invalid init method is rejected by the parser."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--dataset", "mnist",
                "--model_src_dir", "/path",
                "--ckpt_dir", "/path",
                "--out_dir", "/path",
                "--epsilon", "0.3",
                "--alpha", "0.01",
                "--init", "invalid_method",
            ])


class TestValidateArgs:
    def test_valid_args(self):
        args = argparse.Namespace(
            n_examples=3,
            init="random",
            df_max_iter=50,
        )
        validate_args(args)

    def test_n_examples_too_low(self):
        args = argparse.Namespace(
            n_examples=0,
            init="random",
            df_max_iter=50,
        )
        with pytest.raises(ValueError, match="n_examples"):
            validate_args(args)

    def test_deepfool_invalid_max_iter(self):
        args = argparse.Namespace(
            n_examples=1,
            init="deepfool",
            df_max_iter=0,
        )
        with pytest.raises(ValueError, match="df_max_iter must be > 0"):
            validate_args(args)

    def test_multi_deepfool_valid_args(self):
        """multi_deepfool with valid df_max_iter passes validation."""
        args = argparse.Namespace(
            n_examples=1,
            init="multi_deepfool",
            df_max_iter=50,
        )
        validate_args(args)

    def test_multi_deepfool_invalid_max_iter_zero(self):
        """multi_deepfool with df_max_iter=0 raises ValueError."""
        args = argparse.Namespace(
            n_examples=1,
            init="multi_deepfool",
            df_max_iter=0,
        )
        with pytest.raises(ValueError, match="df_max_iter must be > 0"):
            validate_args(args)

    def test_multi_deepfool_invalid_max_iter_negative(self):
        """multi_deepfool with negative df_max_iter raises ValueError."""
        args = argparse.Namespace(
            n_examples=1,
            init="multi_deepfool",
            df_max_iter=-5,
        )
        with pytest.raises(ValueError, match="df_max_iter must be > 0"):
            validate_args(args)


class TestGetModelTag:
    def test_extract_from_ckpt_dir(self):
        assert get_model_tag("models/nat") == "nat"

    def test_extract_from_nested_path(self):
        assert get_model_tag("/path/to/models/adv_trained") == "adv_trained"


class TestFormatBaseName:
    def test_random_init(self):
        args = argparse.Namespace(
            dataset="mnist",
            ckpt_dir="/path/to/models/nat",
            init="random",
            df_max_iter=50,
            df_overshoot=0.02,
        )
        result = format_base_name(args, (0,))
        assert result == "mnist_nat_random"

    def test_deepfool_init_includes_df_part(self):
        args = argparse.Namespace(
            dataset="cifar10",
            ckpt_dir="/path/to/models/adv_trained",
            init="deepfool",
            df_max_iter=50,
            df_overshoot=0.02,
        )
        result = format_base_name(args, (1, 2))
        assert "cifar10" in result
        assert "adv_trained" in result
        assert "deepfool" in result
        assert "dfiter50" in result
        assert "dfo0.02" in result

    def test_multi_deepfool_init_includes_df_part(self):
        """multi_deepfool format_base_name includes dfiter and dfo parameters."""
        args = argparse.Namespace(
            dataset="mnist",
            ckpt_dir="/path/to/models/nat",
            init="multi_deepfool",
            df_max_iter=50,
            df_overshoot=0.02,
        )
        result = format_base_name(args, (0,))
        assert "mnist" in result
        assert "nat" in result
        assert "multi_deepfool" in result
        assert "dfiter50" in result
        assert "dfo0.02" in result

    def test_clean_init_no_df_part(self):
        args = argparse.Namespace(
            dataset="mnist",
            ckpt_dir="/path/to/models/nat",
            init="clean",
            df_max_iter=50,
            df_overshoot=0.02,
        )
        result = format_base_name(args, (0,))
        assert result == "mnist_nat_clean"
        assert "dfiter" not in result


class TestFormatTitle:
    def test_random_init(self):
        args = argparse.Namespace(
            dataset="mnist",
            ckpt_dir="/path/to/models/nat",
            init="random",
        )
        result = format_title(args)
        assert "MNIST" in result
        assert "random-init" in result

    def test_deepfool_init(self):
        args = argparse.Namespace(
            dataset="cifar10",
            ckpt_dir="/path/to/models/adv_trained",
            init="deepfool",
        )
        result = format_title(args)
        assert "CIFAR10" in result
        assert "deepfool-init" in result

    def test_multi_deepfool_init(self):
        """multi_deepfool format_title includes multi_deepfool-init."""
        args = argparse.Namespace(
            dataset="mnist",
            ckpt_dir="/path/to/models/nat",
            init="multi_deepfool",
        )
        result = format_title(args)
        assert "MNIST" in result
        assert "multi_deepfool-init" in result
        assert "PGD" in result

    def test_clean_init(self):
        args = argparse.Namespace(
            dataset="cifar10",
            ckpt_dir="/path/to/models/adv_trained",
            init="clean",
        )
        result = format_title(args)
        assert "CIFAR10" in result
        assert "clean-init" in result
