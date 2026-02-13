"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- TF availability check --------------------------------------------------
# NOTE: We do NOT attempt `import tensorflow` in this process because on ARM
# Docker emulation (linux/amd64 on Apple Silicon) TF 1.x segfaults due to
# missing AVX CPU instructions, which kills the entire test runner.
#
# Strategy (fast, no subprocess needed):
#   1. Metadata check: importlib.util.find_spec (does not import)
#   2. CPU capability check: TF 1.15 pip wheels require AVX.  If the CPU
#      flags (from /proc/cpuinfo) do not include 'avx', TF will crash with
#      SIGILL on import.  This catches ARM-emulated Docker containers.
#   3. Override: set SKIP_TF_TESTS=1 to force-skip TF tests.

import importlib.util
import os


def _cpu_has_avx():
    """Return True if the CPU advertises AVX support (Linux only)."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return "avx" in line.split()
    except (IOError, OSError):
        pass
    # Non-Linux (macOS, etc.) or unreadable: assume AVX is available.
    return True


def _check_tf_importable():
    """Check whether TensorFlow can be imported without crashing."""
    if os.environ.get("SKIP_TF_TESTS"):
        return False
    if importlib.util.find_spec("tensorflow") is None:
        return False
    if not _cpu_has_avx():
        return False
    return True


HAS_TF = _check_tf_importable()

requires_tf = pytest.mark.skipif(
    not HAS_TF, reason="TensorFlow not available"
)


# -- Marker registration ----------------------------------------------------


def pytest_configure(config):
    """Register custom markers to suppress unknown-marker warnings."""
    config.addinivalue_line(
        "markers",
        "requires_tf: mark test as requiring TensorFlow (skip when unavailable)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with @pytest.mark.requires_tf when TF is unavailable."""
    if HAS_TF:
        return
    skip_tf = pytest.mark.skip(reason="TensorFlow not available (no AVX)")
    for item in items:
        if "requires_tf" in item.keywords:
            item.add_marker(skip_tf)
