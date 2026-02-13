"""Tests for run_all_ex100.sh batch script structure.

Verifies that the PGD batch script meets requirements 4.1-4.4:
- 4.1: Uses src/main.py (not loss_curves.py) for Multi-DeepFool
- 4.2: All init methods run via single entry point (src/main.py)
- 4.3: Common parameters managed via shell variables
- 4.4: Timestamped log output to output directory
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "run_all_ex100.sh"


@pytest.fixture
def script_content():
    """Load run_all_ex100.sh content."""
    return SCRIPT_PATH.read_text(encoding="utf-8")


class TestReq41_NoLossCurves:
    """Req 4.1: Batch script shall use src/main.py, not loss_curves.py."""

    def test_no_loss_curves_invocation(self, script_content):
        """loss_curves.py must not be invoked anywhere in the script."""
        # Match any usage of loss_curves.py as a script argument
        matches = re.findall(r'loss_curves\.py', script_content)
        assert len(matches) == 0, (
            f"Found {len(matches)} references to loss_curves.py; "
            "all should use src/main.py"
        )

    def test_multi_deepfool_uses_src_main(self, script_content):
        """Multi-DeepFool runs must use src/main.py."""
        # Find all multi_deepfool run invocations
        mdf_runs = re.findall(
            r'run\s+"([^"]+)"\s+"[^"]*multi_deepfool[^"]*"',
            script_content,
        )
        assert len(mdf_runs) > 0, "No multi_deepfool runs found"
        for script_name in mdf_runs:
            assert script_name == "src/main.py", (
                f"Multi-DeepFool run uses '{script_name}' "
                "instead of 'src/main.py'"
            )


class TestReq42_SingleEntryPoint:
    """Req 4.2: All init methods via single entry point (src/main.py)."""

    def test_all_init_methods_present(self, script_content):
        """All four init methods (clean, random, deepfool, multi_deepfool)
        must be present in the script."""
        expected_inits = {"clean", "random", "deepfool", "multi_deepfool"}
        found_inits = set(
            re.findall(r'--init\s+(clean|random|deepfool|multi_deepfool)',
                       script_content)
        )
        assert found_inits == expected_inits, (
            f"Expected init methods {expected_inits}, "
            f"found {found_inits}"
        )

    def test_all_runs_use_src_main(self, script_content):
        """All run() invocations that execute PGD must use src/main.py
        as the script."""
        # Extract all run() calls with their script argument
        run_calls = re.findall(
            r'run\s+"([^"]+)"\s+"(src_[^"]+)"',
            script_content,
        )
        assert len(run_calls) > 0, "No run() calls found with src_ prefix"
        for script_name, log_name in run_calls:
            assert script_name == "src/main.py", (
                f"Run '{log_name}' uses '{script_name}' "
                "instead of 'src/main.py'"
            )

    def test_uses_python_m_src_main_or_direct(self, script_content):
        """The run() helper invokes src/main.py (directly or via -m)."""
        # run() helper calls $PYTHON "$script" "$@", so script="src/main.py"
        # Verify all run calls use "src/main.py" as script
        all_run_scripts = re.findall(
            r'run\s+"([^"]+)"\s+"[^"]+"',
            script_content,
        )
        pgd_scripts = [s for s in all_run_scripts
                       if s not in ("find_common_correct_samples.py",)]
        for script_name in pgd_scripts:
            assert script_name == "src/main.py", (
                f"Found non-src/main.py script: '{script_name}'"
            )


class TestReq43_CommonParameters:
    """Req 4.3: Common parameters managed via shell variables."""

    def test_epsilon_variables_defined(self, script_content):
        """MNIST_EPS and CIFAR10_EPS variables must be defined."""
        assert re.search(r'^MNIST_EPS=', script_content, re.MULTILINE)
        assert re.search(r'^CIFAR10_EPS=', script_content, re.MULTILINE)

    def test_alpha_variables_defined(self, script_content):
        """MNIST_ALPHA and CIFAR10_ALPHA variables must be defined."""
        assert re.search(r'^MNIST_ALPHA=', script_content, re.MULTILINE)
        assert re.search(r'^CIFAR10_ALPHA=', script_content, re.MULTILINE)

    def test_common_iteration_params_defined(self, script_content):
        """TOTAL_ITER, DF_MAX_ITER, DF_OVERSHOOT must be defined."""
        assert re.search(r'^TOTAL_ITER=', script_content, re.MULTILINE)
        assert re.search(r'^DF_MAX_ITER=', script_content, re.MULTILINE)
        assert re.search(r'^DF_OVERSHOOT=', script_content, re.MULTILINE)

    def test_restart_variables_defined(self, script_content):
        """NUM_RESTARTS_* variables must be defined for all init types."""
        for init_type in ("CLEAN", "RANDOM", "DEEPFOOL", "MULTI_DEEPFOOL"):
            var_name = f"NUM_RESTARTS_{init_type}"
            assert re.search(
                rf'^{var_name}=', script_content, re.MULTILINE
            ), f"Variable {var_name} not defined"

    def test_parameters_used_via_variables(self, script_content):
        """Runs should reference variables, not hardcoded values."""
        # Check that epsilon uses $MNIST_EPS or $CIFAR10_EPS, not literals
        epsilon_args = re.findall(
            r'--epsilon\s+(\S+)', script_content
        )
        for val in epsilon_args:
            assert val.startswith('"$') or val.startswith('$'), (
                f"--epsilon uses literal '{val}' instead of variable"
            )


class TestReq44_TimestampedLogs:
    """Req 4.4: Execution logs with timestamp saved to output directory."""

    def test_log_dir_created(self, script_content):
        """LOG_DIR must be created via mkdir."""
        assert re.search(r'mkdir\s+.*\$LOG_DIR', script_content) or \
               re.search(r'mkdir\s+.*"\$LOG_DIR"', script_content), \
            "LOG_DIR directory creation not found"

    def test_timestamp_function_defined(self, script_content):
        """A timestamp function must be defined."""
        assert re.search(r'timestamp\(\)', script_content), \
            "timestamp() function not found"

    def test_log_uses_timestamp(self, script_content):
        """Log file path must include timestamp."""
        assert re.search(r'timestamp', script_content), \
            "No timestamp usage found in log path"

    def test_run_helper_uses_tee(self, script_content):
        """run() helper must use tee to save output to log."""
        assert re.search(r'tee\s+.*\$log', script_content) or \
               re.search(r'tee\s+.*"\$log"', script_content), \
            "tee command not found in run() helper"


class TestLogNamePrefix:
    """Log names for multi_deepfool runs must use src_ prefix."""

    def test_multi_deepfool_log_names_use_src_prefix(self, script_content):
        """Multi-DeepFool run log names must have src_ prefix, not lc_."""
        mdf_log_names = re.findall(
            r'run\s+"[^"]+"\s+"([^"]*multi_deepfool[^"]*)"',
            script_content,
        )
        assert len(mdf_log_names) > 0, "No multi_deepfool log names found"
        for log_name in mdf_log_names:
            assert log_name.startswith("src_"), (
                f"Multi-DeepFool log name '{log_name}' "
                "does not start with 'src_'"
            )
            assert not log_name.startswith("lc_"), (
                f"Multi-DeepFool log name '{log_name}' "
                "still uses legacy 'lc_' prefix"
            )


class TestMultiDeepFoolCompleteness:
    """All dataset/model combinations must be covered for multi_deepfool."""

    DATASETS = ("mnist", "cifar10")
    MODELS = ("nat", "adv", "nat_and_adv", "weak_adv")

    def test_all_dataset_model_combinations_for_mdf(self, script_content):
        """Each dataset-model combination must have a multi_deepfool run."""
        mdf_log_names = re.findall(
            r'run\s+"[^"]+"\s+"([^"]*multi_deepfool[^"]*)"',
            script_content,
        )
        for dataset in self.DATASETS:
            for model in self.MODELS:
                expected_fragment = f"{dataset}_{model}_multi_deepfool"
                found = any(expected_fragment in name
                            for name in mdf_log_names)
                assert found, (
                    f"Missing multi_deepfool run for "
                    f"{dataset}/{model}"
                )

    def test_multi_deepfool_runs_have_df_params(self, script_content):
        """Multi-DeepFool runs must include --df_max_iter and
        --df_overshoot parameters."""
        # Extract multi_deepfool run blocks
        mdf_blocks = re.findall(
            r'run\s+"src/main\.py"\s+"src_[^"]*multi_deepfool[^"]*"'
            r'.*?(?=\nrun\s|$)',
            script_content,
            re.DOTALL,
        )
        assert len(mdf_blocks) > 0, "No multi_deepfool run blocks found"
        for block in mdf_blocks:
            assert "--df_max_iter" in block, (
                f"Missing --df_max_iter in block: {block[:80]}"
            )
            assert "--df_overshoot" in block, (
                f"Missing --df_overshoot in block: {block[:80]}"
            )
