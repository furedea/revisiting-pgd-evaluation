"""Tests for src/timing_cli.py (timing CLI entry point).

Requirements: 2.4, 2.5, 2.9, 7.4
"""

import argparse
import importlib
import inspect
import json
import os
import sys
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module importability (Req 2.9: python -m src.timing_cli)
# ---------------------------------------------------------------------------
class TestModuleImport:
    """Verify that src.timing_cli can be imported and exposes required API."""

    def test_module_imports(self):
        """src.timing_cli module should be importable."""
        mod = importlib.import_module("src.timing_cli")
        assert mod is not None

    def test_has_build_timing_arg_parser(self):
        """Module must expose build_timing_arg_parser function."""
        from src.timing_cli import build_timing_arg_parser
        assert callable(build_timing_arg_parser)

    def test_has_main(self):
        """Module must expose main function."""
        from src.timing_cli import main
        assert callable(main)


# ---------------------------------------------------------------------------
# CLI argument parsing (Req 2.4)
# ---------------------------------------------------------------------------
class TestBuildTimingArgParser:
    """Verify build_timing_arg_parser returns correct argument parser."""

    def test_returns_argument_parser(self):
        """Must return an ArgumentParser instance."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_required_arguments(self):
        """Parser must require mandatory arguments."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_dataset_argument(self):
        """--dataset must accept mnist and cifar10."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/path/to/out",
            "--exp_name", "test",
        ])
        assert args.dataset == "mnist"

    def test_model_argument(self):
        """--model must accept model names."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "adv",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/path/to/out",
            "--exp_name", "test",
        ])
        assert args.model == "adv"

    def test_init_argument_choices(self):
        """--init must accept random, deepfool, multi_deepfool."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        for method in ["random", "deepfool", "multi_deepfool"]:
            args = parser.parse_args([
                "--dataset", "mnist",
                "--model", "nat",
                "--init", method,
                "--num_restarts", "1",
                "--common_indices_file", "/path/to/indices.json",
                "--out_dir", "/path/to/out",
                "--exp_name", "test",
            ])
            assert args.init == method

    def test_num_restarts_argument(self):
        """--num_restarts must be parsed as integer."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "5",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/path/to/out",
            "--exp_name", "test",
        ])
        assert args.num_restarts == 5

    def test_common_indices_file_argument(self):
        """--common_indices_file must be parsed as string."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/path/to/out",
            "--exp_name", "test",
        ])
        assert args.common_indices_file == "/path/to/indices.json"

    def test_optional_defaults(self):
        """Optional arguments must have correct defaults."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/path/to/out",
            "--exp_name", "test",
        ])
        assert args.total_iter == 100
        assert args.df_max_iter == 50
        assert args.df_overshoot == 0.02
        assert args.seed == 0

    def test_out_dir_argument(self):
        """--out_dir must be a required argument."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/custom/output",
            "--exp_name", "test",
        ])
        assert args.out_dir == "/custom/output"

    def test_exp_name_argument(self):
        """--exp_name must be a required argument."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        args = parser.parse_args([
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", "/path/to/indices.json",
            "--out_dir", "/path/to/out",
            "--exp_name", "my_experiment",
        ])
        assert args.exp_name == "my_experiment"

    def test_invalid_init_choice_rejected(self):
        """Invalid init method must be rejected."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--dataset", "mnist",
                "--model", "nat",
                "--init", "invalid",
                "--num_restarts", "1",
                "--common_indices_file", "/path/to/indices.json",
                "--out_dir", "/path/to/out",
                "--exp_name", "test",
            ])

    def test_help_option_available(self):
        """--help should be available (Req 2.4: help shows usage)."""
        from src.timing_cli import build_timing_arg_parser
        parser = build_timing_arg_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Dataset config auto-resolution (Req 2.4, 3.1)
# ---------------------------------------------------------------------------
class TestDatasetConfigAutoResolution:
    """Verify timing CLI integrates dataset config auto-resolution."""

    def test_uses_dataset_config_module(self):
        """timing_cli must import and use resolve_dataset_config."""
        import src.timing_cli as mod
        source = inspect.getsource(mod)
        assert "resolve_dataset_config" in source

    def test_source_uses_dataset_config_for_eps_alpha(self):
        """Main must use DatasetConfig for eps/alpha, not hardcoded values."""
        import src.timing_cli as mod
        source = inspect.getsource(mod.main)
        assert "resolve_dataset_config" in source


# ---------------------------------------------------------------------------
# JSON output format (Req 2.5, 7.4)
# ---------------------------------------------------------------------------
class TestJSONOutputFormat:
    """Verify JSON output format is compatible with legacy measure_timing.py."""

    def _run_main_with_mock(self, tmp_path, dataset="mnist", model="nat",
                            init_method="random", num_restarts=1):
        """Helper: run main() with mocked run_timing_experiment."""
        from src.timing_cli import main

        indices_file = str(tmp_path / "indices.json")
        with open(indices_file, "w") as f:
            json.dump({"selected_indices": [0, 1, 2]}, f)

        out_dir = str(tmp_path / "output")
        exp_name = "test_exp"

        mock_results = [
            {"init": 0.001, "pgd": 0.5, "total": 0.501},
            {"init": 0.002, "pgd": 0.48, "total": 0.482},
            {"init": 0.001, "pgd": 0.51, "total": 0.511},
        ]

        test_args = [
            "--dataset", dataset,
            "--model", model,
            "--init", init_method,
            "--num_restarts", str(num_restarts),
            "--common_indices_file", indices_file,
            "--out_dir", out_dir,
            "--exp_name", exp_name,
        ]

        with mock.patch("src.timing_cli.run_timing_experiment", return_value=mock_results), \
             mock.patch("sys.argv", ["timing_cli"] + test_args):
            main()

        timing_dir = os.path.join(out_dir, "timing", exp_name)
        json_file = os.path.join(
            timing_dir,
            f"timing_{dataset}_{model}_{init_method}_n{num_restarts}.json"
        )
        return json_file

    def test_json_file_created(self, tmp_path):
        """JSON output file must be created at expected path."""
        json_file = self._run_main_with_mock(tmp_path)
        assert os.path.exists(json_file)

    def test_json_has_required_keys(self, tmp_path):
        """JSON output must contain all required top-level keys (Req 7.4)."""
        json_file = self._run_main_with_mock(tmp_path)
        with open(json_file) as f:
            data = json.load(f)
        required_keys = {
            "dataset", "model", "init", "num_restarts",
            "indices", "total_iter", "df_max_iter", "df_overshoot",
            "eps", "alpha", "results",
        }
        assert required_keys.issubset(set(data.keys()))

    def test_json_dataset_field(self, tmp_path):
        """JSON 'dataset' field must match input."""
        json_file = self._run_main_with_mock(tmp_path, dataset="mnist")
        with open(json_file) as f:
            data = json.load(f)
        assert data["dataset"] == "mnist"

    def test_json_model_field(self, tmp_path):
        """JSON 'model' field must match input."""
        json_file = self._run_main_with_mock(tmp_path, model="nat")
        with open(json_file) as f:
            data = json.load(f)
        assert data["model"] == "nat"

    def test_json_init_field(self, tmp_path):
        """JSON 'init' field must match input."""
        json_file = self._run_main_with_mock(tmp_path, init_method="random")
        with open(json_file) as f:
            data = json.load(f)
        assert data["init"] == "random"

    def test_json_num_restarts_field(self, tmp_path):
        """JSON 'num_restarts' field must match input."""
        json_file = self._run_main_with_mock(tmp_path, num_restarts=3)
        with open(json_file) as f:
            data = json.load(f)
        assert data["num_restarts"] == 3

    def test_json_indices_field(self, tmp_path):
        """JSON 'indices' field must be the loaded indices list."""
        json_file = self._run_main_with_mock(tmp_path)
        with open(json_file) as f:
            data = json.load(f)
        assert data["indices"] == [0, 1, 2]

    def test_json_results_is_list(self, tmp_path):
        """JSON 'results' field must be a list of timing dicts."""
        json_file = self._run_main_with_mock(tmp_path)
        with open(json_file) as f:
            data = json.load(f)
        assert isinstance(data["results"], list)
        for r in data["results"]:
            assert "init" in r
            assert "pgd" in r
            assert "total" in r

    def test_json_eps_resolved_from_dataset_config(self, tmp_path):
        """JSON 'eps' must be auto-resolved from dataset_config (Req 3.1)."""
        json_file = self._run_main_with_mock(tmp_path, dataset="mnist")
        with open(json_file) as f:
            data = json.load(f)
        assert abs(data["eps"] - 0.3) < 1e-6

    def test_json_alpha_resolved_from_dataset_config(self, tmp_path):
        """JSON 'alpha' must be auto-resolved from dataset_config (Req 3.1)."""
        json_file = self._run_main_with_mock(tmp_path, dataset="mnist")
        with open(json_file) as f:
            data = json.load(f)
        assert abs(data["alpha"] - 0.01) < 1e-6

    def test_json_output_naming_convention(self, tmp_path):
        """Output file name must follow legacy convention: timing_{dataset}_{model}_{init}_n{num_restarts}.json."""
        json_file = self._run_main_with_mock(
            tmp_path, dataset="cifar10", model="adv",
            init_method="deepfool", num_restarts=5
        )
        basename = os.path.basename(json_file)
        assert basename == "timing_cifar10_adv_deepfool_n5.json"

    def test_json_output_directory_structure(self, tmp_path):
        """Output must be saved in {out_dir}/timing/{exp_name}/ directory."""
        json_file = self._run_main_with_mock(tmp_path)
        assert "/timing/test_exp/" in json_file


# ---------------------------------------------------------------------------
# __main__.py entry point (Req 2.9: python -m src.timing_cli)
# ---------------------------------------------------------------------------
class TestMainEntryPoint:
    """Verify python -m src.timing_cli is executable."""

    def test_timing_cli_has_main_guard(self):
        """src/timing_cli.py must have if __name__ == '__main__' guard."""
        import src.timing_cli as mod
        source = inspect.getsource(mod)
        assert 'if __name__ ==' in source or "if __name__==" in source

    def test_main_calls_run_timing_experiment(self):
        """main() must call run_timing_experiment from src.timing."""
        import src.timing_cli as mod
        source = inspect.getsource(mod.main)
        assert "run_timing_experiment" in source

    def test_main_loads_common_indices(self):
        """main() must load common indices from file."""
        import src.timing_cli as mod
        source = inspect.getsource(mod.main)
        assert "common_indices_file" in source

    def test_main_saves_json(self):
        """main() must save results as JSON."""
        import src.timing_cli as mod
        source = inspect.getsource(mod.main)
        assert "json.dump" in source or "json_dump" in source


# ---------------------------------------------------------------------------
# run_timing_experiment integration (Req 2.9: model_src_dir/ckpt_dir auto-resolved)
# ---------------------------------------------------------------------------
class TestModelPathAutoResolution:
    """Verify that model_src_dir and ckpt_dir are auto-resolved from dataset and model name."""

    def test_main_constructs_model_src_dir_from_dataset(self):
        """main() must construct model_src_dir from dataset config."""
        import src.timing_cli as mod
        source = inspect.getsource(mod.main)
        # Must use DatasetConfig.model_src_dir
        assert "model_src_dir" in source

    def test_main_constructs_ckpt_dir_from_model(self):
        """main() must construct ckpt_dir from model_src_dir and model name."""
        import src.timing_cli as mod
        source = inspect.getsource(mod.main)
        assert "ckpt_dir" in source


# ---------------------------------------------------------------------------
# Summary output (informational, mirrors legacy behavior)
# ---------------------------------------------------------------------------
class TestSummaryOutput:
    """Verify main prints summary after measurement."""

    def test_main_prints_info(self, tmp_path, capsys):
        """main() should print informational messages."""
        from src.timing_cli import main

        indices_file = str(tmp_path / "indices.json")
        with open(indices_file, "w") as f:
            json.dump({"selected_indices": [0, 1]}, f)

        out_dir = str(tmp_path / "output")
        test_args = [
            "--dataset", "mnist",
            "--model", "nat",
            "--init", "random",
            "--num_restarts", "1",
            "--common_indices_file", indices_file,
            "--out_dir", out_dir,
            "--exp_name", "test",
        ]

        mock_results = [
            {"init": 0.001, "pgd": 0.5, "total": 0.501},
            {"init": 0.002, "pgd": 0.48, "total": 0.482},
        ]

        with mock.patch("src.timing_cli.run_timing_experiment", return_value=mock_results), \
             mock.patch("sys.argv", ["timing_cli"] + test_args):
            main()

        captured = capsys.readouterr()
        assert "Saved" in captured.out or "saved" in captured.out.lower() or \
               "SUMMARY" in captured.out or len(captured.out) > 0
