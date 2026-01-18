"""Tests for math_utils module."""

import numpy as np
import pytest

from src.math_utils import (
    clip_to_unit_interval,
    linf_distance,
    project_linf,
    scale_to_linf_ball,
)


class TestLinfDistance:
    def test_identical_arrays(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert linf_distance(x, x) == 0.0

    def test_simple_distance(self):
        x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert linf_distance(x1, x2) == 3.0

    def test_negative_values(self):
        x1 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        x2 = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        assert linf_distance(x1, x2) == 2.0

    def test_2d_array(self):
        x1 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        x2 = np.array([[0.5, 0.3], [0.1, 0.4]], dtype=np.float32)
        assert linf_distance(x1, x2) == 0.5


class TestProjectLinf:
    def test_within_ball(self):
        x = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        x_nat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = project_linf(x, x_nat, eps=0.3)
        np.testing.assert_array_equal(result, x)

    def test_outside_ball(self):
        x = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        x_nat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = project_linf(x, x_nat, eps=0.3)
        expected = np.array([0.8, 0.2, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_preserves_dtype(self):
        x = np.array([1.0, 0.0], dtype=np.float32)
        x_nat = np.array([0.5, 0.5], dtype=np.float32)
        result = project_linf(x, x_nat, eps=0.1)
        assert result.dtype == np.float32


class TestClipToUnitInterval:
    def test_within_range(self):
        x = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        result = clip_to_unit_interval(x)
        np.testing.assert_array_equal(result, x)

    def test_below_zero(self):
        x = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
        result = clip_to_unit_interval(x)
        expected = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_dtype(self):
        x = np.array([-1.0, 2.0], dtype=np.float32)
        result = clip_to_unit_interval(x)
        assert result.dtype == np.float32


class TestScaleToLinfBall:
    def test_within_ball(self):
        x = np.array([0.55, 0.45], dtype=np.float32)
        x_nat = np.array([0.5, 0.5], dtype=np.float32)
        result = scale_to_linf_ball(x, x_nat, eps=0.1)
        np.testing.assert_array_almost_equal(result, x)

    def test_outside_ball_scales_down(self):
        x = np.array([1.0, 0.5], dtype=np.float32)
        x_nat = np.array([0.5, 0.5], dtype=np.float32)
        result = scale_to_linf_ball(x, x_nat, eps=0.1)
        distance = linf_distance(result, x_nat)
        assert distance <= 0.1 + 1e-6

    def test_preserves_direction(self):
        x = np.array([1.0, 0.0], dtype=np.float32)
        x_nat = np.array([0.5, 0.5], dtype=np.float32)
        result = scale_to_linf_ball(x, x_nat, eps=0.1)
        delta_original = x - x_nat
        delta_result = result - x_nat
        ratio = delta_result / (delta_original + 1e-12)
        np.testing.assert_array_almost_equal(ratio[0], ratio[1], decimal=5)
