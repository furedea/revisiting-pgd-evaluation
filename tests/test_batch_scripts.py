"""Tests for batch shell scripts (run_timing_ex100.sh).

Requirements: 4.5
Task: 6.2 - Timing batch script consolidation
"""

import os

import pytest


# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRunTimingEx100ShEntryPoint:
    """Verify run_timing_ex100.sh uses python -m src.timing_cli instead of measure_timing.py."""

    @pytest.fixture()
    def script_content(self):
        """Read run_timing_ex100.sh content."""
        path = os.path.join(PROJECT_ROOT, "run_timing_ex100.sh")
        with open(path) as f:
            return f.read()

    def test_script_exists(self):
        """run_timing_ex100.sh must exist."""
        path = os.path.join(PROJECT_ROOT, "run_timing_ex100.sh")
        assert os.path.isfile(path)

    def test_no_measure_timing_py_reference(self, script_content):
        """Script must not reference measure_timing.py (Req 4.5)."""
        assert "measure_timing.py" not in script_content

    def test_uses_src_timing_cli(self, script_content):
        """Script must use 'python -m src.timing_cli' or '-m src.timing_cli' (Req 4.5)."""
        assert "-m src.timing_cli" in script_content

    def test_cli_arg_dataset_present(self, script_content):
        """Script must pass --dataset argument."""
        assert "--dataset" in script_content

    def test_cli_arg_model_present(self, script_content):
        """Script must pass --model argument."""
        assert "--model" in script_content

    def test_cli_arg_init_present(self, script_content):
        """Script must pass --init argument."""
        assert "--init" in script_content

    def test_cli_arg_num_restarts_present(self, script_content):
        """Script must pass --num_restarts argument."""
        assert "--num_restarts" in script_content

    def test_cli_arg_common_indices_file_present(self, script_content):
        """Script must pass --common_indices_file argument."""
        assert "--common_indices_file" in script_content

    def test_cli_arg_out_dir_present(self, script_content):
        """Script must pass --out_dir argument."""
        assert "--out_dir" in script_content

    def test_cli_arg_exp_name_present(self, script_content):
        """Script must pass --exp_name argument."""
        assert "--exp_name" in script_content

    def test_mnist_timing_covered(self, script_content):
        """Script must include MNIST timing measurements."""
        assert "mnist" in script_content.lower()

    def test_cifar10_timing_covered(self, script_content):
        """Script must include CIFAR10 timing measurements."""
        assert "cifar10" in script_content.lower()

    def test_random_init_covered(self, script_content):
        """Script must measure timing for random initialization."""
        assert "random" in script_content

    def test_deepfool_init_covered(self, script_content):
        """Script must measure timing for deepfool initialization."""
        # Check for deepfool init lines (not multi_deepfool)
        lines = script_content.split("\n")
        has_deepfool = any(
            "--init deepfool" in line
            for line in lines
        )
        assert has_deepfool

    def test_multi_deepfool_init_covered(self, script_content):
        """Script must measure timing for multi_deepfool initialization."""
        assert "--init multi_deepfool" in script_content

    def test_run_function_passes_module_flag(self, script_content):
        """The run() helper must invoke python with -m src.timing_cli."""
        # The run() function receives a "script" argument.
        # After migration, calls should pass "-m src.timing_cli" as the script.
        assert "-m src.timing_cli" in script_content

    def test_log_directory_created(self, script_content):
        """Script must create log directory."""
        assert "LOG_DIR" in script_content
        assert "mkdir" in script_content

    def test_timestamp_function_present(self, script_content):
        """Script must have a timestamp function for logging."""
        assert "timestamp" in script_content
