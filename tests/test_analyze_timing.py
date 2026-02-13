"""Tests for analyze_timing module."""

import json
import os
from typing import Dict, List

import numpy as np
import pytest

from analyze_timing import (
    build_arg_parser,
    compute_statistics,
    infer_exp_name,
    load_timing_results,
    validate_input_dir,
)


class TestInferExpName:
    """Test experiment name inference from input directory path."""

    def test_standard_path(self):
        """Standard timing directory path yields basename."""
        result = infer_exp_name("outputs/timing/timing_ex100")
        assert result == "timing_ex100"

    def test_trailing_slash(self):
        """Trailing slash should be stripped before inference."""
        result = infer_exp_name("outputs/timing/timing_ex100/")
        assert result == "timing_ex100"

    def test_nested_path(self):
        """Deeply nested path still yields final component."""
        result = infer_exp_name("/work/outputs/timing/timing_ex10")
        assert result == "timing_ex10"

    def test_single_component(self):
        """Single-component path yields that component."""
        result = infer_exp_name("timing_ex5")
        assert result == "timing_ex5"

    def test_dot_relative_path(self):
        """Relative path with dot yields correct basename."""
        result = infer_exp_name("./outputs/timing/timing_ex100")
        assert result == "timing_ex100"


class TestValidateInputDir:
    """Test input directory validation with clear error messages."""

    def test_nonexistent_directory_raises_error(self, tmp_path):
        """Non-existent directory raises FileNotFoundError with path in message."""
        bad_dir = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            validate_input_dir(bad_dir)

    def test_empty_directory_raises_error(self, tmp_path):
        """Empty directory raises FileNotFoundError with informative message."""
        empty_dir = str(tmp_path / "empty_timing")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError, match="timing.*JSON"):
            validate_input_dir(empty_dir)

    def test_directory_with_non_timing_files(self, tmp_path):
        """Directory with non-timing files raises FileNotFoundError."""
        d = str(tmp_path / "some_dir")
        os.makedirs(d)
        # Create a non-timing file
        with open(os.path.join(d, "other.txt"), "w") as f:
            f.write("not timing data")
        with pytest.raises(FileNotFoundError, match="timing.*JSON"):
            validate_input_dir(d)

    def test_valid_directory_passes(self, tmp_path):
        """Directory with timing JSON files passes validation."""
        d = str(tmp_path / "timing_data")
        os.makedirs(d)
        timing_data = {
            "dataset": "mnist",
            "model": "nat",
            "init": "random",
            "num_restarts": 1,
            "indices": [0],
            "total_iter": 100,
            "eps": 0.3,
            "alpha": 0.01,
            "results": [{"init": 0.001, "pgd": 0.5, "total": 0.501}],
        }
        with open(os.path.join(d, "timing_mnist_nat_random_n1.json"), "w") as f:
            json.dump(timing_data, f)
        # Should not raise
        validate_input_dir(d)


class TestLoadTimingResults:
    """Test loading timing results from JSON files."""

    def _create_timing_file(
        self,
        directory: str,
        filename: str,
        dataset: str = "mnist",
        model: str = "nat",
        init: str = "random",
        num_restarts: int = 1,
        results: List[Dict[str, float]] = None,
    ) -> None:
        """Helper to create timing JSON file."""
        if results is None:
            results = [{"init": 0.001, "pgd": 0.5, "total": 0.501}]
        data = {
            "dataset": dataset,
            "model": model,
            "init": init,
            "num_restarts": num_restarts,
            "indices": list(range(len(results))),
            "total_iter": 100,
            "eps": 0.3,
            "alpha": 0.01,
            "results": results,
        }
        with open(os.path.join(directory, filename), "w") as f:
            json.dump(data, f)

    def test_loads_single_file(self, tmp_path):
        """Load a single timing JSON file correctly."""
        d = str(tmp_path)
        self._create_timing_file(d, "timing_mnist_nat_random_n1.json")
        results = load_timing_results(d)
        assert "mnist" in results
        assert "random_n1" in results["mnist"]
        assert len(results["mnist"]["random_n1"]) == 1

    def test_loads_multiple_datasets(self, tmp_path):
        """Load files from multiple datasets."""
        d = str(tmp_path)
        self._create_timing_file(
            d, "timing_mnist_nat_random_n1.json", dataset="mnist"
        )
        self._create_timing_file(
            d, "timing_cifar10_nat_random_n1.json", dataset="cifar10"
        )
        results = load_timing_results(d)
        assert "mnist" in results
        assert "cifar10" in results

    def test_aggregates_results_from_multiple_files(self, tmp_path):
        """Multiple files for same method are aggregated."""
        d = str(tmp_path)
        self._create_timing_file(
            d,
            "timing_mnist_nat_random_n1.json",
            results=[{"init": 0.001, "pgd": 0.5, "total": 0.501}],
        )
        self._create_timing_file(
            d,
            "timing_mnist_adv_random_n1.json",
            model="adv",
            results=[{"init": 0.002, "pgd": 0.6, "total": 0.602}],
        )
        results = load_timing_results(d)
        assert len(results["mnist"]["random_n1"]) == 2

    def test_empty_directory_returns_empty(self, tmp_path):
        """Empty directory returns empty dict."""
        d = str(tmp_path)
        results = load_timing_results(d)
        assert results == {}

    def test_ignores_non_timing_files(self, tmp_path):
        """Non-timing files are ignored."""
        d = str(tmp_path)
        with open(os.path.join(d, "other.json"), "w") as f:
            json.dump({"data": "not timing"}, f)
        self._create_timing_file(d, "timing_mnist_nat_random_n1.json")
        results = load_timing_results(d)
        assert "mnist" in results
        assert len(results) == 1


class TestComputeStatistics:
    """Test timing statistics computation."""

    def test_single_sample_means(self):
        """Single sample returns its values as mean."""
        timing_list = [{"init": 0.1, "pgd": 0.5, "total": 0.6}]
        stats = compute_statistics(timing_list)
        assert stats["init_mean"] == pytest.approx(0.1)
        assert stats["pgd_mean"] == pytest.approx(0.5)
        assert stats["total_mean"] == pytest.approx(0.6)

    def test_single_sample_std_is_zero(self):
        """Single sample has zero std."""
        timing_list = [{"init": 0.1, "pgd": 0.5, "total": 0.6}]
        stats = compute_statistics(timing_list)
        assert stats["init_std"] == pytest.approx(0.0)
        assert stats["pgd_std"] == pytest.approx(0.0)
        assert stats["total_std"] == pytest.approx(0.0)

    def test_multiple_samples_mean(self):
        """Multiple samples compute correct mean."""
        timing_list = [
            {"init": 0.1, "pgd": 0.4, "total": 0.5},
            {"init": 0.3, "pgd": 0.6, "total": 0.9},
        ]
        stats = compute_statistics(timing_list)
        assert stats["init_mean"] == pytest.approx(0.2)
        assert stats["pgd_mean"] == pytest.approx(0.5)
        assert stats["total_mean"] == pytest.approx(0.7)

    def test_n_samples_count(self):
        """n_samples reflects number of timing entries."""
        timing_list = [
            {"init": 0.1, "pgd": 0.4, "total": 0.5},
            {"init": 0.3, "pgd": 0.6, "total": 0.9},
            {"init": 0.2, "pgd": 0.5, "total": 0.7},
        ]
        stats = compute_statistics(timing_list)
        assert stats["n_samples"] == 3

    def test_returns_float_values(self):
        """All returned values are Python floats, not numpy types."""
        timing_list = [{"init": 0.1, "pgd": 0.5, "total": 0.6}]
        stats = compute_statistics(timing_list)
        for key in ["init_mean", "init_std", "pgd_mean", "pgd_std", "total_mean", "total_std"]:
            assert isinstance(stats[key], float), f"{key} should be float"


class TestBuildArgParser:
    """Test argparse configuration for --help and argument validation."""

    def test_help_description_present(self):
        """Parser has a non-empty description for --help output."""
        parser = build_arg_parser()
        assert parser.description is not None
        assert len(parser.description) > 0

    def test_input_dir_required(self):
        """--input_dir is a required argument."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_input_dir_parsed(self):
        """--input_dir is parsed correctly."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "/path/to/timing"])
        assert args.input_dir == "/path/to/timing"

    def test_out_dir_default(self):
        """--out_dir has a default value."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "/path/to/timing"])
        assert args.out_dir is not None

    def test_out_dir_custom(self):
        """--out_dir can be customized."""
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--input_dir", "/path/to/timing", "--out_dir", "/custom"]
        )
        assert args.out_dir == "/custom"

    def test_dataset_optional(self):
        """--dataset is optional for filtering."""
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--input_dir", "/path/to/timing", "--dataset", "mnist"]
        )
        assert args.dataset == "mnist"

    def test_dataset_default_is_none(self):
        """--dataset defaults to None (process all datasets)."""
        parser = build_arg_parser()
        args = parser.parse_args(["--input_dir", "/path/to/timing"])
        assert args.dataset is None

    def test_input_dir_help_text(self):
        """--input_dir has help text."""
        parser = build_arg_parser()
        # Check that the input_dir action has help text
        for action in parser._actions:
            if hasattr(action, "dest") and action.dest == "input_dir":
                assert action.help is not None
                assert len(action.help) > 0
                break
