"""Tests for src/timing.py (timing-optimized code paths).

TF-dependent functions are tested for signature/validation only.
Actual TF integration tests require @requires_tf marker.
"""

import importlib
import inspect
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------
class TestModuleImport:
    """Verify that src.timing can be imported and exposes required API."""

    def test_module_imports(self):
        """src.timing module should be importable."""
        mod = importlib.import_module("src.timing")
        assert mod is not None

    def test_has_random_init_timing(self):
        """Module must expose random_init_timing function."""
        from src.timing import random_init_timing
        assert callable(random_init_timing)

    def test_has_deepfool_init_timing(self):
        """Module must expose deepfool_init_timing function."""
        from src.timing import deepfool_init_timing
        assert callable(deepfool_init_timing)

    def test_has_multi_deepfool_init_timing(self):
        """Module must expose multi_deepfool_init_timing function."""
        from src.timing import multi_deepfool_init_timing
        assert callable(multi_deepfool_init_timing)

    def test_has_run_pgd_timing(self):
        """Module must expose run_pgd_timing function."""
        from src.timing import run_pgd_timing
        assert callable(run_pgd_timing)

    def test_has_measure_single_sample(self):
        """Module must expose measure_single_sample function."""
        from src.timing import measure_single_sample
        assert callable(measure_single_sample)

    def test_has_run_timing_experiment(self):
        """Module must expose run_timing_experiment function."""
        from src.timing import run_timing_experiment
        assert callable(run_timing_experiment)


# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------
class TestFunctionSignatures:
    """Verify function signatures match design.md specifications."""

    def test_random_init_timing_signature(self):
        """random_init_timing(rng, x_nat_batch, eps) -> ndarray."""
        from src.timing import random_init_timing
        sig = inspect.signature(random_init_timing)
        params = list(sig.parameters.keys())
        assert params == ["rng", "x_nat_batch", "eps"]

    def test_deepfool_init_timing_signature(self):
        """deepfool_init_timing(sess, ops, x_nat, eps, max_iter, overshoot)."""
        from src.timing import deepfool_init_timing
        sig = inspect.signature(deepfool_init_timing)
        params = list(sig.parameters.keys())
        assert params == ["sess", "ops", "x_nat", "eps", "max_iter", "overshoot"]

    def test_multi_deepfool_init_timing_signature(self):
        """multi_deepfool_init_timing(sess, ops, x_nat, eps, max_iter, overshoot, num_targets)."""
        from src.timing import multi_deepfool_init_timing
        sig = inspect.signature(multi_deepfool_init_timing)
        params = list(sig.parameters.keys())
        assert params == [
            "sess", "ops", "x_nat", "eps",
            "max_iter", "overshoot", "num_targets",
        ]

    def test_run_pgd_timing_signature(self):
        """run_pgd_timing(sess, ops, x_adv, x_nat_batch, y_batch, eps, alpha, total_iter)."""
        from src.timing import run_pgd_timing
        sig = inspect.signature(run_pgd_timing)
        params = list(sig.parameters.keys())
        assert params == [
            "sess", "ops", "x_adv", "x_nat_batch",
            "y_batch", "eps", "alpha", "total_iter",
        ]

    def test_measure_single_sample_signature(self):
        """measure_single_sample must accept all required parameters."""
        from src.timing import measure_single_sample
        sig = inspect.signature(measure_single_sample)
        params = list(sig.parameters.keys())
        expected = [
            "sess", "ops", "x_nat", "y_nat",
            "init_method", "num_restarts",
            "eps", "alpha", "total_iter",
            "df_max_iter", "df_overshoot", "seed",
        ]
        assert params == expected

    def test_run_timing_experiment_signature(self):
        """run_timing_experiment must accept all required parameters."""
        from src.timing import run_timing_experiment
        sig = inspect.signature(run_timing_experiment)
        params = list(sig.parameters.keys())
        expected = [
            "dataset", "model_src_dir", "ckpt_dir", "indices",
            "init_method", "num_restarts",
            "eps", "alpha", "total_iter",
            "df_max_iter", "df_overshoot", "seed",
        ]
        assert params == expected


# ---------------------------------------------------------------------------
# random_init_timing: TF-free, fully testable
# ---------------------------------------------------------------------------
class TestRandomInitTiming:
    """Tests for random_init_timing (no TF dependency)."""

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        from src.timing import random_init_timing
        rng = np.random.RandomState(42)
        x_nat = np.random.rand(3, 784).astype(np.float32)
        result = random_init_timing(rng, x_nat, eps=0.3)
        assert result.shape == x_nat.shape

    def test_output_dtype_float32(self):
        """Output must be float32."""
        from src.timing import random_init_timing
        rng = np.random.RandomState(42)
        x_nat = np.random.rand(1, 784).astype(np.float32)
        result = random_init_timing(rng, x_nat, eps=0.3)
        assert result.dtype == np.float32

    def test_output_within_unit_interval(self):
        """Output values must be clipped to [0, 1]."""
        from src.timing import random_init_timing
        rng = np.random.RandomState(42)
        x_nat = np.random.rand(5, 784).astype(np.float32)
        result = random_init_timing(rng, x_nat, eps=0.3)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_within_linf_ball(self):
        """Output must stay within L-inf ball of x_nat."""
        from src.timing import random_init_timing
        rng = np.random.RandomState(42)
        eps = 0.3
        x_nat = np.random.rand(5, 784).astype(np.float32)
        result = random_init_timing(rng, x_nat, eps=eps)
        linf = np.max(np.abs(result - x_nat))
        assert linf <= eps + 1e-6

    def test_different_seeds_give_different_results(self):
        """Different RNG states must produce different initializations."""
        from src.timing import random_init_timing
        x_nat = np.random.rand(1, 784).astype(np.float32)
        r1 = random_init_timing(np.random.RandomState(0), x_nat, eps=0.3)
        r2 = random_init_timing(np.random.RandomState(1), x_nat, eps=0.3)
        assert not np.allclose(r1, r2)

    def test_no_logging_attribute(self):
        """random_init_timing must not call LOGGER (check source for no logging)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.random_init_timing)
        assert "LOGGER" not in source
        assert "tqdm" not in source
        assert "print(" not in source


# ---------------------------------------------------------------------------
# No logging/tqdm/recording in timing functions (source-level checks)
# ---------------------------------------------------------------------------
class TestNoOverheadInTimingFunctions:
    """Verify that timing functions do not contain overhead code."""

    def _get_source(self, func_name):
        import src.timing as timing_mod
        func = getattr(timing_mod, func_name)
        return inspect.getsource(func)

    def test_deepfool_init_timing_no_overhead(self):
        """deepfool_init_timing must not contain LOGGER/tqdm/print."""
        source = self._get_source("deepfool_init_timing")
        assert "LOGGER" not in source
        assert "tqdm" not in source
        assert "print(" not in source

    def test_multi_deepfool_init_timing_no_overhead(self):
        """multi_deepfool_init_timing must not contain LOGGER/tqdm/print."""
        source = self._get_source("multi_deepfool_init_timing")
        assert "LOGGER" not in source
        assert "tqdm" not in source
        assert "print(" not in source

    def test_run_pgd_timing_no_overhead(self):
        """run_pgd_timing must not contain LOGGER/tqdm/print/array recording."""
        source = self._get_source("run_pgd_timing")
        assert "LOGGER" not in source
        assert "tqdm" not in source
        assert "print(" not in source
        # Must not record losses/preds arrays
        assert "losses" not in source
        assert "preds" not in source

    def test_run_pgd_timing_no_per_step_loss_eval(self):
        """run_pgd_timing must not evaluate per_ex_loss_op per iteration."""
        source = self._get_source("run_pgd_timing")
        assert "per_ex_loss_op" not in source
        assert "y_pred_op" not in source


# ---------------------------------------------------------------------------
# Shared module reuse verification
# ---------------------------------------------------------------------------
class TestSharedModuleReuse:
    """Verify that timing.py reuses shared modules from src/."""

    def test_imports_math_utils(self):
        """Must import from src.math_utils."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod)
        assert "from src.math_utils import" in source or "import src.math_utils" in source

    def test_imports_dto(self):
        """Must import from src.dto."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod)
        assert "from src.dto import" in source or "import src.dto" in source

    def test_does_not_redefine_clip_to_unit_interval(self):
        """Must not redefine clip_to_unit_interval locally."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod)
        assert "def clip_to_unit_interval" not in source

    def test_does_not_redefine_project_linf(self):
        """Must not redefine project_linf locally."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod)
        assert "def project_linf" not in source

    def test_does_not_redefine_model_ops(self):
        """Must not redefine ModelOps locally."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod)
        assert "class ModelOps" not in source


# ---------------------------------------------------------------------------
# Helper: build mock sess and ops for measure_single_sample tests
# ---------------------------------------------------------------------------
def _make_mock_sess_ops(num_classes=10, input_dim=784):
    """Create mock TF session and ModelOps for timing tests.

    The mock sess.run dispatches based on the op argument:
      - ops.logits  -> one-hot-like logits favoring class 0
      - ops.grads_all_op -> random per-class gradients
      - ops.grad_op -> random gradient of same shape as input
    """
    ops = mock.MagicMock()
    sess = mock.MagicMock()

    # Provide stable deterministic behaviour
    rng = np.random.RandomState(999)

    def _sess_run(op, feed_dict=None):
        if op is ops.logits:
            # Return logits that always predict class 0
            logits = np.zeros((1, num_classes), dtype=np.float32)
            logits[0, 0] = 10.0
            return logits
        if op is ops.grads_all_op:
            x_in = feed_dict[ops.x_ph]
            batch = x_in.shape[0]
            shape = x_in.shape[1:]
            grads = rng.randn(batch, num_classes, *shape).astype(np.float32)
            return grads
        if op is ops.grad_op:
            x_in = feed_dict[ops.x_ph]
            return rng.randn(*x_in.shape).astype(np.float32)
        raise ValueError("Unexpected op: %s" % op)

    sess.run = mock.MagicMock(side_effect=_sess_run)
    return sess, ops


# ---------------------------------------------------------------------------
# measure_single_sample: perf_counter timing, return dict
# ---------------------------------------------------------------------------
class TestMeasureSingleSample:
    """Verify measure_single_sample returns correct timing dict (Req 2.1, 2.5)."""

    def test_returns_dict_with_init_pgd_total_keys(self):
        """Return value must contain 'init', 'pgd', 'total' keys."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([3], dtype=np.int64)

        result = measure_single_sample(
            sess, ops, x_nat, y_nat,
            init_method="random", num_restarts=1,
            eps=0.3, alpha=0.01, total_iter=2,
            df_max_iter=10, df_overshoot=0.02, seed=0,
        )
        assert isinstance(result, dict)
        assert "init" in result
        assert "pgd" in result
        assert "total" in result

    def test_timing_values_are_non_negative_floats(self):
        """All timing values must be non-negative floats."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([0], dtype=np.int64)

        result = measure_single_sample(
            sess, ops, x_nat, y_nat,
            init_method="random", num_restarts=1,
            eps=0.3, alpha=0.01, total_iter=2,
            df_max_iter=10, df_overshoot=0.02, seed=42,
        )
        assert isinstance(result["init"], float)
        assert isinstance(result["pgd"], float)
        assert isinstance(result["total"], float)
        assert result["init"] >= 0.0
        assert result["pgd"] >= 0.0
        assert result["total"] >= 0.0

    def test_total_equals_init_plus_pgd(self):
        """total must equal init + pgd (Req 2.5)."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([5], dtype=np.int64)

        result = measure_single_sample(
            sess, ops, x_nat, y_nat,
            init_method="random", num_restarts=1,
            eps=0.3, alpha=0.01, total_iter=2,
            df_max_iter=10, df_overshoot=0.02, seed=0,
        )
        assert abs(result["total"] - (result["init"] + result["pgd"])) < 1e-12

    def test_uses_perf_counter(self):
        """measure_single_sample must use time.perf_counter for timing."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.measure_single_sample)
        assert "perf_counter" in source

    def test_random_init_method_accepted(self):
        """init_method='random' must work without error."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([0], dtype=np.int64)

        result = measure_single_sample(
            sess, ops, x_nat, y_nat,
            init_method="random", num_restarts=3,
            eps=0.3, alpha=0.01, total_iter=1,
            df_max_iter=10, df_overshoot=0.02, seed=0,
        )
        assert result["total"] >= 0.0

    def test_deepfool_init_method_accepted(self):
        """init_method='deepfool' must work without error."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([0], dtype=np.int64)

        result = measure_single_sample(
            sess, ops, x_nat, y_nat,
            init_method="deepfool", num_restarts=1,
            eps=0.3, alpha=0.01, total_iter=1,
            df_max_iter=2, df_overshoot=0.02, seed=0,
        )
        assert result["total"] >= 0.0

    def test_multi_deepfool_init_method_accepted(self):
        """init_method='multi_deepfool' must work without error."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([0], dtype=np.int64)

        result = measure_single_sample(
            sess, ops, x_nat, y_nat,
            init_method="multi_deepfool", num_restarts=3,
            eps=0.3, alpha=0.01, total_iter=1,
            df_max_iter=2, df_overshoot=0.02, seed=0,
        )
        assert result["total"] >= 0.0

    def test_unknown_init_method_raises_value_error(self):
        """Unknown init_method must raise ValueError."""
        from src.timing import measure_single_sample

        sess, ops = _make_mock_sess_ops()
        x_nat = np.random.rand(1, 784).astype(np.float32)
        y_nat = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError, match="Unknown init method"):
            measure_single_sample(
                sess, ops, x_nat, y_nat,
                init_method="invalid", num_restarts=1,
                eps=0.3, alpha=0.01, total_iter=1,
                df_max_iter=10, df_overshoot=0.02, seed=0,
            )

    def test_no_logging_overhead_in_measure_single_sample(self):
        """measure_single_sample must not contain LOGGER/tqdm/print."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.measure_single_sample)
        assert "LOGGER" not in source
        assert "tqdm" not in source
        assert "print(" not in source


# ---------------------------------------------------------------------------
# run_timing_experiment: warmup + per-sample measurement (Req 2.1, 2.6)
# ---------------------------------------------------------------------------
class TestRunTimingExperiment:
    """Verify run_timing_experiment structure and contracts via source inspection.

    run_timing_experiment uses lazy imports (tensorflow, src.model_loader,
    src.data_loader) inside the function body.  Mocking those local imports
    reliably is fragile, so we verify contracts through source-level checks
    and defer full integration testing to Docker/TF environments.
    """

    def test_warmup_uses_first_index(self):
        """Warmup run must use indices[0] (Req 2.6)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "indices[0]" in source

    def test_warmup_calls_measure_single_sample(self):
        """Warmup must call measure_single_sample before the main loop (Req 2.6)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        # Warmup call should appear before the "for idx in indices" loop
        warmup_pos = source.find("measure_single_sample")
        loop_pos = source.find("for idx in indices")
        assert warmup_pos != -1, "measure_single_sample not found in source"
        assert loop_pos != -1, "for idx in indices loop not found in source"
        assert warmup_pos < loop_pos, "warmup must come before the measurement loop"

    def test_collects_results_per_sample(self):
        """Results list must be built by appending per-sample measurements."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "all_results.append(result)" in source or \
               "all_results.append(" in source

    def test_returns_all_results(self):
        """Function must return the collected results list."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "return all_results" in source

    def test_return_annotation_is_list(self):
        """Return type annotation must be List[Dict[str, float]]."""
        from src.timing import run_timing_experiment
        sig = inspect.signature(run_timing_experiment)
        ret = sig.return_annotation
        # Check the annotation string representation
        ret_str = str(ret) if ret != inspect.Parameter.empty else ""
        # Accept both actual annotation and 'empty' (Python 3.6 compat)
        # The function has a type hint: -> List[Dict[str, float]]
        assert ret_str == "typing.List[typing.Dict[str, float]]" or \
               ret == inspect.Parameter.empty  # acceptable if no runtime annotation

    def test_uses_shared_model_loader(self):
        """run_timing_experiment must use src.model_loader (Req 2.2)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "load_model_module" in source
        assert "instantiate_model" in source
        assert "create_tf_session" in source
        assert "restore_checkpoint" in source

    def test_uses_shared_data_loader(self):
        """run_timing_experiment must use src.data_loader (Req 2.2)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "load_test_data" in source

    def test_uses_model_ops_from_model(self):
        """run_timing_experiment must use ModelOps.from_model (Req 2.2)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "ModelOps.from_model" in source

    def test_no_print_in_run_timing_experiment(self):
        """run_timing_experiment must not contain print statements (timing-optimized)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "print(" not in source

    def test_no_tqdm_in_run_timing_experiment(self):
        """run_timing_experiment must not contain tqdm (timing-optimized)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "tqdm" not in source

    def test_resets_default_graph(self):
        """run_timing_experiment must call reset_default_graph."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "reset_default_graph" in source

    def test_uses_saver_for_checkpoint(self):
        """run_timing_experiment must create a Saver and restore checkpoint."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod.run_timing_experiment)
        assert "Saver" in source
        assert "restore_checkpoint" in source


# ---------------------------------------------------------------------------
# Legacy measure_timing.py removal verification (Req 2.3, 5.5)
# ---------------------------------------------------------------------------
class TestLegacyMeasureTimingRemoval:
    """Verify that legacy measure_timing.py has been removed and
    all timing functionality is provided by src/timing.py and src/timing_cli.py.
    """

    def test_legacy_measure_timing_py_does_not_exist(self):
        """measure_timing.py must not exist or be empty at project root (Req 2.3)."""
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        legacy_path = os.path.join(project_root, "measure_timing.py")
        if os.path.exists(legacy_path):
            assert os.path.getsize(legacy_path) == 0, \
                "Legacy measure_timing.py must be deleted or empty after migration"

    def test_no_python_imports_of_measure_timing(self):
        """No Python file in src/ or tests/ should import measure_timing."""
        import os
        import re
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = re.compile(r"(from\s+measure_timing|import\s+measure_timing)")
        violations = []
        for dirpath, _dirnames, filenames in os.walk(project_root):
            # Skip hidden directories and non-Python files
            if any(part.startswith(".") for part in dirpath.split(os.sep)):
                continue
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                if fname == "test_timing.py":
                    continue  # skip self (contains class name references)
                with open(fpath) as f:
                    content = f.read()
                if pattern.search(content):
                    violations.append(fpath)
        assert violations == [], \
            "Files importing measure_timing: %s" % violations

    def test_src_timing_module_provides_all_legacy_functions(self):
        """src/timing.py must provide all functions that were in measure_timing.py."""
        from src.timing import (
            random_init_timing,
            deepfool_init_timing,
            multi_deepfool_init_timing,
            run_pgd_timing,
            measure_single_sample,
            run_timing_experiment,
        )
        assert callable(random_init_timing)
        assert callable(deepfool_init_timing)
        assert callable(multi_deepfool_init_timing)
        assert callable(run_pgd_timing)
        assert callable(measure_single_sample)
        assert callable(run_timing_experiment)

    def test_src_timing_cli_provides_entry_point(self):
        """src/timing_cli.py must provide CLI entry point replacing measure_timing.py main."""
        from src.timing_cli import build_timing_arg_parser, main
        assert callable(build_timing_arg_parser)
        assert callable(main)

    def test_src_timing_reuses_shared_modules_not_local_copies(self):
        """src/timing.py must not contain local copies of ModelOps/math_utils (Req 5.5)."""
        import src.timing as timing_mod
        source = inspect.getsource(timing_mod)
        # Must not redefine shared utilities locally
        assert "class ModelOps" not in source
        assert "def clip_to_unit_interval" not in source
        assert "def project_linf" not in source
        assert "def load_model_module" not in source
        assert "def load_test_data" not in source
