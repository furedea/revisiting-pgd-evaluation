"""Tests for cli module."""

import argparse

import pytest

from src.cli import (
    build_arg_parser,
    format_base_name,
    format_indices_part,
    format_title,
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
            "--epsilon", "0.031",
            "--alpha", "0.007",
        ])
        assert args.start_idx == 0
        assert args.n_examples == 1
        assert args.steps == 100
        assert args.num_restarts == 20
        assert args.init == "random"
        assert args.df_project == "clip"


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
        with pytest.raises(ValueError, match="n_examples must be 1..5"):
            validate_args(args)

    def test_n_examples_too_high(self):
        args = argparse.Namespace(
            n_examples=6,
            init="random",
            df_max_iter=50,
        )
        with pytest.raises(ValueError, match="n_examples must be 1..5"):
            validate_args(args)

    def test_deepfool_invalid_max_iter(self):
        args = argparse.Namespace(
            n_examples=1,
            init="deepfool",
            df_max_iter=0,
        )
        with pytest.raises(ValueError, match="df_max_iter must be > 0"):
            validate_args(args)


class TestFormatIndicesPart:
    def test_single_index(self):
        result = format_indices_part((42,))
        assert result == "idx42"

    def test_multiple_indices(self):
        result = format_indices_part((1, 2, 3))
        assert result == "indices1-2-3"


class TestFormatBaseName:
    def test_random_init(self):
        args = argparse.Namespace(
            dataset="mnist",
            tag="naturally_trained",
            init="random",
            steps=100,
            epsilon=0.3,
            alpha=0.01,
            num_restarts=20,
            seed=0,
            no_clip=False,
        )
        result = format_base_name(args, (0,))
        assert "mnist" in result
        assert "random_init" in result
        assert "idx0" in result
        assert "k100" in result
        assert "eps0.3" in result

    def test_deepfool_init(self):
        args = argparse.Namespace(
            dataset="cifar10",
            tag="adv_trained",
            init="deepfool",
            steps=200,
            epsilon=0.031,
            alpha=0.007,
            num_restarts=10,
            seed=42,
            no_clip=False,
            df_max_iter=50,
            df_overshoot=0.02,
            df_jitter=0.01,
            df_project="scale",
        )
        result = format_base_name(args, (1, 2))
        assert "cifar10" in result
        assert "deepfool_init" in result
        assert "dfiter50" in result
        assert "dfproject_scale" in result


class TestFormatTitle:
    def test_random_init(self):
        args = argparse.Namespace(
            dataset="mnist",
            tag="naturally_trained",
            init="random",
        )
        result = format_title(args)
        assert "MNIST" in result
        assert "random-init" in result

    def test_deepfool_init(self):
        args = argparse.Namespace(
            dataset="cifar10",
            tag="adv_trained",
            init="deepfool",
            df_jitter=0.01,
            df_project="maxloss",
        )
        result = format_title(args)
        assert "CIFAR10" in result
        assert "deepfool-init" in result
        assert "df_jitter=0.01" in result
        assert "df_project=maxloss" in result
