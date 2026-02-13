"""Tests for src/multi_deepfool.py (TF-independent and TF-dependent parts)."""

import numpy as np
import pytest


class TestComputePerturbationToTarget:
    """Tests for compute_perturbation_to_target (pure NumPy)."""

    def test_returns_tuple_of_array_and_float(self):
        """Return type is (np.ndarray, float)."""
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 3
        input_dim = 4
        f = np.array([0.0, -2.0, -1.0], dtype=np.float32)
        grads_all = np.random.RandomState(0).randn(
            num_classes, input_dim
        ).astype(np.float32)
        start_label = 0
        target_class = 1

        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label, target_class
        )

        assert isinstance(r_flat, np.ndarray)
        assert r_flat.dtype == np.float32
        assert isinstance(r_norm, float)

    def test_output_shape_matches_flattened_input(self):
        """r_flat shape should be (input_dim,) matching flattened grad shape."""
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 4
        input_dim = 10
        f = np.zeros(num_classes, dtype=np.float32)
        f[2] = -3.0
        grads_all = np.random.RandomState(42).randn(
            num_classes, input_dim
        ).astype(np.float32)

        r_flat, _ = compute_perturbation_to_target(
            f, grads_all, start_label=0, target_class=2
        )

        assert r_flat.shape == (input_dim,)

    def test_perturbation_direction_is_toward_target(self):
        """Perturbation should move in the direction of (grad_target - grad_start)."""
        from src.multi_deepfool import compute_perturbation_to_target

        # Construct simple case where direction is clear
        num_classes = 2
        input_dim = 3
        f = np.array([0.0, -5.0], dtype=np.float32)
        grads_all = np.zeros((num_classes, input_dim), dtype=np.float32)
        grads_all[0] = [1.0, 0.0, 0.0]  # start_label gradient
        grads_all[1] = [0.0, 1.0, 0.0]  # target_class gradient

        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label=0, target_class=1
        )

        # w = grads[1] - grads[0] = [-1, 1, 0]
        w = grads_all[1].reshape(-1) - grads_all[0].reshape(-1)
        # r_flat should be proportional to w
        assert r_norm > 0.0
        # Check direction alignment (cosine similarity should be 1.0)
        cosine = float(np.dot(r_flat, w)) / (
            float(np.linalg.norm(r_flat)) * float(np.linalg.norm(w))
        )
        np.testing.assert_almost_equal(cosine, 1.0, decimal=5)

    def test_norm_is_consistent_with_vector(self):
        """r_norm should equal np.linalg.norm(r_flat)."""
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 5
        input_dim = 8
        f = np.random.RandomState(7).randn(num_classes).astype(np.float32)
        grads_all = np.random.RandomState(8).randn(
            num_classes, input_dim
        ).astype(np.float32)

        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label=2, target_class=4
        )

        expected_norm = float(np.linalg.norm(r_flat))
        np.testing.assert_almost_equal(r_norm, expected_norm, decimal=5)

    def test_zero_logit_difference_gives_zero_perturbation(self):
        """When f[target_class] == 0, perturbation should be zero."""
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 3
        input_dim = 4
        f = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        grads_all = np.random.RandomState(1).randn(
            num_classes, input_dim
        ).astype(np.float32)

        # f[target_class=1] == 0.0 -> r should be zero
        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label=0, target_class=1
        )

        np.testing.assert_array_almost_equal(
            r_flat, np.zeros(input_dim, dtype=np.float32), decimal=6
        )
        assert r_norm < 1e-6

    def test_multidimensional_grads_are_flattened(self):
        """Grads with shape (num_classes, H, W) should be handled via flatten."""
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 3
        h, w = 4, 5
        f = np.array([0.0, -2.0, -3.0], dtype=np.float32)
        grads_all = np.random.RandomState(99).randn(
            num_classes, h, w
        ).astype(np.float32)

        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label=0, target_class=2
        )

        assert r_flat.shape == (h * w,)
        assert r_norm > 0.0

    def test_formula_correctness(self):
        """Verify the DeepFool perturbation formula:
        r = (|f[target]| / (||w||^2 + 1e-12)) * w
        where w = grads[target] - grads[start].
        """
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 3
        input_dim = 4
        f = np.array([0.0, -2.5, -1.0], dtype=np.float32)
        grads_all = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        start_label = 0
        target_class = 1

        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label, target_class
        )

        # Manual computation
        w = grads_all[target_class].reshape(-1) - grads_all[start_label].reshape(-1)
        # w = [-1, 1, 0, 0]
        denom = float(np.dot(w, w) + 1e-12)
        expected_r = (abs(float(f[target_class])) / denom) * w
        expected_norm = float(np.linalg.norm(expected_r))

        np.testing.assert_array_almost_equal(r_flat, expected_r, decimal=5)
        np.testing.assert_almost_equal(r_norm, expected_norm, decimal=5)

    def test_zero_gradient_difference_gives_near_zero_perturbation(self):
        """When grad_target == grad_start (w=0), perturbation should be near zero.

        The denominator epsilon (1e-12) prevents division by zero, and
        the result should be effectively zero since w is zero.
        """
        from src.multi_deepfool import compute_perturbation_to_target

        num_classes = 3
        input_dim = 4
        f = np.array([0.0, -5.0, -3.0], dtype=np.float32)
        # Identical gradients for start and target classes
        shared_grad = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        grads_all = np.zeros((num_classes, input_dim), dtype=np.float32)
        grads_all[0] = shared_grad
        grads_all[1] = shared_grad  # same as start -> w = 0

        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label=0, target_class=1
        )

        # w = grads[1] - grads[0] = [0, 0, 0, 0]
        # r = (|f[1]| / (0 + 1e-12)) * [0,0,0,0] = [0,0,0,0]
        np.testing.assert_array_almost_equal(
            r_flat, np.zeros(input_dim, dtype=np.float32), decimal=6
        )
        assert r_norm < 1e-6
        assert r_flat.shape == (input_dim,)


def _build_simple_tf_model(num_classes, input_dim):
    """Build a minimal TF graph for testing (linear model).

    Returns (sess, ops) with a simple linear classifier.
    TF is imported lazily inside this function.
    """
    import tensorflow as tf
    from src.dto import ModelOps

    tf.compat.v1.reset_default_graph()

    x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, input_dim], name="x")
    y_ph = tf.compat.v1.placeholder(tf.int64, shape=[None], name="y")

    # Simple linear model with fixed weights
    w = tf.Variable(
        tf.eye(num_classes, input_dim, dtype=tf.float32),
        name="w",
    )
    b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32), name="b")
    logits = tf.matmul(x_ph, tf.transpose(w)) + b

    y_pred_op = tf.argmax(logits, axis=1, output_type=tf.int64)
    per_ex_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_ph, logits=logits,
    )
    loss_sum = tf.reduce_sum(per_ex_loss_op)
    grad_op = tf.gradients(loss_sum, x_ph)[0]

    grads_k = []
    for k in range(num_classes):
        grads_k.append(tf.gradients(logits[:, k], x_ph)[0])
    grads_all_op = tf.stack(grads_k, axis=1)

    ops = ModelOps(
        x_ph=x_ph,
        y_ph=y_ph,
        logits=logits,
        logits_name="logits",
        y_pred_op=y_pred_op,
        per_ex_loss_op=per_ex_loss_op,
        grad_op=grad_op,
        grads_all_op=grads_all_op,
    )

    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=cfg)
    sess.run(tf.compat.v1.global_variables_initializer())

    return sess, ops


@pytest.mark.requires_tf
class TestRunMultiDeepfoolInitPgd:
    """Tests for run_multi_deepfool_init_pgd (TF-dependent)."""

    def test_raises_valueerror_when_num_restarts_exceeds_classes_minus_one(self):
        """num_restarts > num_classes - 1 should raise ValueError."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 3
        input_dim = 4
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0  # Make class 0 the predicted label
            y_nat = np.array([0], dtype=np.int64)

            with pytest.raises(ValueError, match="Not enough target classes"):
                run_multi_deepfool_init_pgd(
                    sess=sess,
                    ops=ops,
                    x_nat=x_nat,
                    y_nat=y_nat,
                    eps=0.3,
                    alpha=0.01,
                    total_iter=5,
                    num_restarts=num_classes,  # 3 > 3-1=2
                    df_max_iter=3,
                    df_overshoot=0.02,
                    seed=0,
                )
        finally:
            sess.close()

    def test_output_losses_shape(self):
        """losses shape should be (num_restarts, total_iter+1)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = 3  # <= num_classes - 1 = 3
        total_iter = 5
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=total_iter,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.losses.shape == (num_restarts, total_iter + 1)
        finally:
            sess.close()

    def test_output_preds_shape(self):
        """preds shape should be (num_restarts, total_iter+1)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = 2
        total_iter = 5
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=total_iter,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.preds.shape == (num_restarts, total_iter + 1)
        finally:
            sess.close()

    def test_output_corrects_shape(self):
        """corrects shape should be (num_restarts, total_iter+1)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = 2
        total_iter = 5
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=total_iter,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.corrects.shape == (num_restarts, total_iter + 1)
        finally:
            sess.close()

    def test_output_corrects_dtype_is_bool(self):
        """corrects dtype should be bool."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=2,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.corrects.dtype == bool
        finally:
            sess.close()

    def test_output_x_adv_final_shape(self):
        """x_adv_final shape should be (num_restarts, *input_shape)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = 2
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.x_adv_final.shape == (num_restarts, input_dim)
        finally:
            sess.close()

    def test_output_x_df_endpoints_shape(self):
        """x_df_endpoints shape should be (num_restarts, *input_shape)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = 2
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.x_df_endpoints is not None
            assert result.x_df_endpoints.shape == (num_restarts, input_dim)
        finally:
            sess.close()

    def test_output_x_init_shape(self):
        """x_init shape should be (1, *input_shape)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=2,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.x_init is not None
            assert result.x_init.shape == (1, input_dim)
        finally:
            sess.close()

    def test_output_x_init_rank_is_int(self):
        """x_init_rank should be an int index."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = 3
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.x_init_rank is not None
            assert isinstance(result.x_init_rank, int)
            assert 0 <= result.x_init_rank < num_restarts
        finally:
            sess.close()

    def test_output_returns_pgd_batch_result(self):
        """Return type should be PGDBatchResult."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd
        from src.dto import PGDBatchResult

        num_classes = 4
        input_dim = 8
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=2,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert isinstance(result, PGDBatchResult)
        finally:
            sess.close()

    def test_losses_dtype_is_float32(self):
        """losses dtype should be float32."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=2,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.losses.dtype == np.float32
        finally:
            sess.close()

    def test_preds_dtype_is_int64(self):
        """preds dtype should be int64."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=2,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.preds.dtype == np.int64
        finally:
            sess.close()

    def test_exact_num_restarts_equals_classes_minus_one(self):
        """num_restarts == num_classes - 1 should succeed (boundary)."""
        from src.multi_deepfool import run_multi_deepfool_init_pgd

        num_classes = 4
        input_dim = 8
        num_restarts = num_classes - 1  # 3 == 4-1, boundary case
        sess, ops = _build_simple_tf_model(num_classes, input_dim)
        try:
            x_nat = np.zeros((1, input_dim), dtype=np.float32)
            x_nat[0, 0] = 5.0
            y_nat = np.array([0], dtype=np.int64)

            result = run_multi_deepfool_init_pgd(
                sess=sess,
                ops=ops,
                x_nat=x_nat,
                y_nat=y_nat,
                eps=0.3,
                alpha=0.01,
                total_iter=3,
                num_restarts=num_restarts,
                df_max_iter=3,
                df_overshoot=0.02,
                seed=0,
            )

            assert result.losses.shape[0] == num_restarts
        finally:
            sess.close()
