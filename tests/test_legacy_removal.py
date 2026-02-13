"""Tests for legacy file removal (Task 3.3).

Verify that loss_curves.py has been removed and the pipeline
operates correctly without it.
"""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent


class TestLossCurvesRemoved:
    """Verify loss_curves.py has been deleted from project root."""

    def test_loss_curves_py_does_not_exist_or_is_empty(self):
        """loss_curves.py should not exist or be empty in the project root."""
        legacy_path = PROJECT_ROOT / "loss_curves.py"
        if legacy_path.exists():
            content = legacy_path.read_text(encoding="utf-8").strip()
            assert content == "", (
                f"Legacy file still has content ({len(content)} chars): "
                f"{legacy_path}"
            )

    def test_no_python_module_imports_loss_curves(self):
        """No .py file under src/ should import from loss_curves."""
        src_dir = PROJECT_ROOT / "src"
        violations = []
        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "import loss_curves" in stripped or "from loss_curves" in stripped:
                    violations.append(f"{py_file.name}:{i}: {stripped}")
        assert violations == [], (
            f"Found imports of loss_curves in src/: {violations}"
        )


class TestPipelineModulesImportable:
    """Verify core pipeline modules remain importable after deletion."""

    def test_src_dto_importable(self):
        """src.dto should be importable."""
        from src.dto import ModelOps, PGDBatchResult, ExamplePanel
        assert ModelOps is not None
        assert PGDBatchResult is not None
        assert ExamplePanel is not None

    def test_src_math_utils_importable(self):
        """src.math_utils should be importable."""
        from src.math_utils import clip_to_unit_interval, project_linf
        assert clip_to_unit_interval is not None
        assert project_linf is not None

    def test_src_cli_importable(self):
        """src.cli should be importable."""
        from src.cli import build_arg_parser
        assert build_arg_parser is not None

    def test_src_multi_deepfool_importable(self):
        """src.multi_deepfool should be importable (replacement for loss_curves)."""
        from src.multi_deepfool import compute_perturbation_to_target
        assert compute_perturbation_to_target is not None

    def test_src_dataset_config_importable(self):
        """src.dataset_config should be importable."""
        from src.dataset_config import resolve_dataset_config
        assert resolve_dataset_config is not None

    def test_src_logging_config_importable(self):
        """src.logging_config should be importable."""
        from src.logging_config import LOGGER_NAME
        assert LOGGER_NAME is not None
