"""Tests for pgd module (non-TF parts)."""

import numpy as np
import pytest

from src.dto import PGDBatchResult
from src.pgd import add_jitter, build_initial_points, choose_show_restart


class TestAddJitter:
    def test_zero_jitter_returns_same(self):
        rng = np.random.RandomState(0)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = add_jitter(rng, x, jitter=0.0)
        np.testing.assert_array_equal(result, x)

    def test_negative_jitter_returns_same(self):
        rng = np.random.RandomState(0)
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = add_jitter(rng, x, jitter=-0.1)
        np.testing.assert_array_equal(result, x)

    def test_positive_jitter_adds_noise(self):
        rng = np.random.RandomState(42)
        x = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        result = add_jitter(rng, x, jitter=0.1)
        assert not np.array_equal(result, x)
        assert np.all(np.abs(result - x) <= 0.1)

    def test_jitter_bounds(self):
        rng = np.random.RandomState(0)
        x = np.zeros((100, 10), dtype=np.float32)
        jitter = 0.05
        result = add_jitter(rng, x, jitter=jitter)
        assert np.all(result >= -jitter)
        assert np.all(result <= jitter)


class TestBuildInitialPoints:
    def test_random_init_shape(self):
        rng = np.random.RandomState(0)
        x_nat_batch = np.zeros((5, 784), dtype=np.float32)
        result = build_initial_points(
            rng=rng,
            init="random",
            x_init=None,
            init_jitter=0.0,
            x_nat_batch=x_nat_batch,
            eps=0.3,
            do_clip=True,
        )
        assert result.shape == (5, 784)
        assert result.dtype == np.float32

    def test_random_init_within_eps(self):
        rng = np.random.RandomState(0)
        x_nat_batch = np.full((5, 784), 0.5, dtype=np.float32)
        eps = 0.3
        result = build_initial_points(
            rng=rng,
            init="random",
            x_init=None,
            init_jitter=0.0,
            x_nat_batch=x_nat_batch,
            eps=eps,
            do_clip=True,
        )
        diff = np.abs(result - x_nat_batch)
        assert np.all(diff <= eps + 1e-6)

    def test_random_init_clipped(self):
        rng = np.random.RandomState(0)
        x_nat_batch = np.full((5, 784), 0.9, dtype=np.float32)
        result = build_initial_points(
            rng=rng,
            init="random",
            x_init=None,
            init_jitter=0.0,
            x_nat_batch=x_nat_batch,
            eps=0.3,
            do_clip=True,
        )
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_deepfool_init_requires_x_init(self):
        rng = np.random.RandomState(0)
        x_nat_batch = np.zeros((5, 784), dtype=np.float32)
        with pytest.raises(ValueError, match="requires x_init"):
            build_initial_points(
                rng=rng,
                init="deepfool",
                x_init=None,
                init_jitter=0.0,
                x_nat_batch=x_nat_batch,
                eps=0.3,
                do_clip=True,
            )

    def test_deepfool_init_uses_x_init(self):
        rng = np.random.RandomState(0)
        x_nat_batch = np.full((3, 10), 0.5, dtype=np.float32)
        x_init = np.full((1, 10), 0.7, dtype=np.float32)
        result = build_initial_points(
            rng=rng,
            init="deepfool",
            x_init=x_init,
            init_jitter=0.0,
            x_nat_batch=x_nat_batch,
            eps=0.3,
            do_clip=True,
        )
        assert result.shape == (3, 10)
        np.testing.assert_array_almost_equal(result, np.full((3, 10), 0.7, dtype=np.float32))


class TestChooseShowRestart:
    def test_chooses_first_wrong(self):
        corrects = np.array([
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ], dtype=bool)
        losses = np.zeros((3, 3), dtype=np.float32)
        pgd_result = PGDBatchResult(
            losses=losses,
            preds=np.zeros((3, 3), dtype=np.int64),
            corrects=corrects,
            x_adv_final=np.zeros((3, 10), dtype=np.float32),
        )
        result = choose_show_restart(pgd_result)
        assert result == 1

    def test_chooses_max_loss_if_all_correct(self):
        corrects = np.array([
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ], dtype=bool)
        losses = np.array([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.5],
            [0.1, 0.2, 0.4],
        ], dtype=np.float32)
        pgd_result = PGDBatchResult(
            losses=losses,
            preds=np.zeros((3, 3), dtype=np.int64),
            corrects=corrects,
            x_adv_final=np.zeros((3, 10), dtype=np.float32),
        )
        result = choose_show_restart(pgd_result)
        assert result == 1
