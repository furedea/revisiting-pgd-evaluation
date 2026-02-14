"""Timing-optimized code paths for PGD initialization and attack.

All functions in this module are designed for accurate execution-time
measurement.  They intentionally exclude logging, tqdm progress bars,
and per-step array recording so that only the algorithm cost is
captured by the caller's perf_counter brackets.

Shared modules (dto, math_utils, model_loader, data_loader) are
reused from the src package.  No utilities are re-defined locally.
"""

import time
from typing import Any, Dict, List

import numpy as np

from src.dto import ModelOps
from src.math_utils import clip_to_unit_interval, project_linf


# ------------------------------------------------------------------
# Initialization (timing-optimized, no logging / array recording)
# ------------------------------------------------------------------

def random_init_timing(
    rng: np.random.RandomState,
    x_nat_batch: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Random initialization within L-infinity ball (timing-optimized, no logging)."""
    noise = rng.uniform(low=-eps, high=eps, size=x_nat_batch.shape).astype(np.float32)
    x_adv = project_linf(x_nat_batch + noise, x_nat_batch, eps)
    return clip_to_unit_interval(x_adv)


def deepfool_init_timing(
    sess: Any,
    ops: ModelOps,
    x_nat: np.ndarray,
    eps: float,
    max_iter: int,
    overshoot: float,
) -> np.ndarray:
    """DeepFool initialization (timing-optimized, no logging / array recording).

    Finds the closest decision boundary via DeepFool and projects the
    result onto the L-inf eps-ball.
    """
    x = x_nat.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(logits0.shape[1])

    for _ in range(int(max_iter)):
        logits = sess.run(ops.logits, feed_dict={ops.x_ph: x})
        current_label = int(np.argmax(logits[0]))
        if current_label != start_label:
            break

        grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x})[0]
        f = logits[0] - logits[0, start_label]

        best_norm = float("inf")
        best_r_flat = None
        grad_start = grads_all[start_label].reshape(-1).astype(np.float32)

        for target in range(num_classes):
            if target == start_label:
                continue
            w = grads_all[target].reshape(-1).astype(np.float32) - grad_start
            denom = float(np.dot(w, w) + 1e-12)
            r_flat = (abs(float(f[target])) / denom) * w
            r_norm = float(np.linalg.norm(r_flat))
            if r_norm < best_norm:
                best_norm = r_norm
                best_r_flat = r_flat.astype(np.float32)

        if best_r_flat is None:
            break

        r = best_r_flat.reshape(x.shape[1:])
        x = x + r[np.newaxis, ...]
        x = np.clip(x, 0.0, 1.0).astype(np.float32)

    # Apply overshoot once at the end
    x = x_nat + (1.0 + overshoot) * (x - x_nat)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    x = project_linf(x, x_nat, eps)
    return clip_to_unit_interval(x)


def multi_deepfool_init_timing(
    sess: Any,
    ops: ModelOps,
    x_nat: np.ndarray,
    eps: float,
    max_iter: int,
    overshoot: float,
    num_targets: int,
) -> np.ndarray:
    """Multi-DeepFool initialization (timing-optimized, no logging / array recording).

    Generates num_targets initial points by running targeted DeepFool toward
    each of the closest non-ground-truth classes, sorted by perturbation norm.
    """
    x = x_nat.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(logits0.shape[1])

    grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x})[0]
    f = logits0[0] - logits0[0, start_label]

    target_distances = []
    grad_start = grads_all[start_label].reshape(-1).astype(np.float32)

    for target in range(num_classes):
        if target == start_label:
            continue
        w = grads_all[target].reshape(-1).astype(np.float32) - grad_start
        denom = float(np.dot(w, w) + 1e-12)
        r_flat = (abs(float(f[target])) / denom) * w
        r_norm = float(np.linalg.norm(r_flat))
        target_distances.append((r_norm, target))

    target_distances.sort(key=lambda item: item[0])
    k_actual = min(int(num_targets), len(target_distances))

    x_advs = []
    for idx in range(k_actual):
        _, target_class = target_distances[idx]
        x_curr = x_nat.astype(np.float32).copy()

        for _ in range(int(max_iter)):
            logits = sess.run(ops.logits, feed_dict={ops.x_ph: x_curr})
            current_pred = int(np.argmax(logits[0]))

            if current_pred == start_label:
                grads_all = sess.run(
                    ops.grads_all_op, feed_dict={ops.x_ph: x_curr}
                )[0]
                f_curr = logits[0] - logits[0, start_label]
                grad_start = grads_all[start_label].reshape(-1).astype(np.float32)
                w = grads_all[target_class].reshape(-1).astype(np.float32) - grad_start
                denom = float(np.dot(w, w) + 1e-12)
                r_flat = (abs(float(f_curr[target_class])) / denom) * w
                r = r_flat.reshape(x_curr.shape[1:])
                x_curr = x_curr + r[np.newaxis, ...]
                x_curr = np.clip(x_curr, 0.0, 1.0).astype(np.float32)
            else:
                break

        x_final = x_nat + (1.0 + overshoot) * (x_curr - x_nat)
        x_final = np.clip(x_final, 0.0, 1.0).astype(np.float32)
        x_final = project_linf(x_final, x_nat, eps)
        x_final = clip_to_unit_interval(x_final)
        x_advs.append(x_final)

    return np.concatenate(x_advs, axis=0)


# ------------------------------------------------------------------
# PGD loop (timing-optimized, no tqdm / logging / loss-pred recording)
# ------------------------------------------------------------------

def run_pgd_timing(
    sess: Any,
    ops: ModelOps,
    x_adv: np.ndarray,
    x_nat_batch: np.ndarray,
    y_batch: np.ndarray,
    eps: float,
    alpha: float,
    total_iter: int,
) -> np.ndarray:
    """PGD iterations (timing-optimized, no progress bar/logging/array recording)."""
    for _ in range(int(total_iter)):
        grad = sess.run(
            ops.grad_op,
            feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch},
        )
        x_adv = x_adv + float(alpha) * np.sign(grad).astype(np.float32)
        x_adv = project_linf(x_adv, x_nat_batch, float(eps))
        x_adv = clip_to_unit_interval(x_adv)
    return x_adv


# ------------------------------------------------------------------
# Measurement helpers
# ------------------------------------------------------------------

def measure_single_sample(
    sess: Any,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    init_method: str,
    num_restarts: int,
    eps: float,
    alpha: float,
    total_iter: int,
    df_max_iter: int,
    df_overshoot: float,
    seed: int,
) -> Dict[str, float]:
    """Measure timing for a single sample.

    Returns:
        {"init": float, "pgd": float, "total": float}
    """
    rng = np.random.RandomState(int(seed))

    # Prepare batch (not included in timing)
    x_nat_batch = np.repeat(x_nat.astype(np.float32), int(num_restarts), axis=0)
    y_batch = np.repeat(y_nat.astype(np.int64), int(num_restarts), axis=0)

    # --- Measure initialization time ---
    t0 = time.perf_counter()

    if init_method == "random":
        x_adv = random_init_timing(rng, x_nat_batch, float(eps))
    elif init_method == "deepfool":
        x_adv = deepfool_init_timing(
            sess, ops, x_nat, float(eps), int(df_max_iter), float(df_overshoot),
        )
        if int(num_restarts) > 1:
            x_adv = np.repeat(x_adv, int(num_restarts), axis=0)
    elif init_method == "multi_deepfool":
        x_adv = multi_deepfool_init_timing(
            sess, ops, x_nat, float(eps),
            int(df_max_iter), float(df_overshoot), int(num_restarts),
        )
    else:
        raise ValueError("Unknown init method: %s" % init_method)

    t_init = time.perf_counter() - t0

    # --- Measure PGD time ---
    t0 = time.perf_counter()
    run_pgd_timing(
        sess, ops, x_adv, x_nat_batch, y_batch,
        float(eps), float(alpha), int(total_iter),
    )
    t_pgd = time.perf_counter() - t0

    return {"init": t_init, "pgd": t_pgd, "total": t_init + t_pgd}


def run_timing_experiment(
    dataset: str,
    model_src_dir: str,
    ckpt_dir: str,
    indices: List[int],
    init_method: str,
    num_restarts: int,
    eps: float,
    alpha: float,
    total_iter: int,
    df_max_iter: int,
    df_overshoot: float,
    seed: int,
) -> List[Dict[str, float]]:
    """Run timing experiment with warmup for all samples.

    Loads model, restores checkpoint, runs warmup, then measures
    each sample individually.
    """
    import tensorflow as tf

    from src.data_loader import load_test_data
    from src.model_loader import (
        create_tf_session,
        instantiate_model,
        load_model_module,
        restore_checkpoint,
    )

    tf.compat.v1.reset_default_graph()
    model_module = load_model_module(model_src_dir, dataset)
    model = instantiate_model(model_module, mode_default="eval")
    ops = ModelOps.from_model(model)

    saver = tf.compat.v1.train.Saver()
    all_results = []  # type: List[Dict[str, float]]

    with create_tf_session() as sess:
        restore_checkpoint(sess, saver, ckpt_dir)
        x_test, y_test = load_test_data(dataset, model_src_dir)

        # Warmup run (discard timing)
        idx = indices[0]
        x_nat = x_test[idx:idx + 1].astype(np.float32)
        y_nat = y_test[idx:idx + 1].astype(np.int64)
        measure_single_sample(
            sess, ops, x_nat, y_nat, init_method,
            int(num_restarts), float(eps), float(alpha),
            int(total_iter), int(df_max_iter), float(df_overshoot),
            int(seed),
        )

        # Actual measurements
        for idx in indices:
            x_nat = x_test[idx:idx + 1].astype(np.float32)
            y_nat = y_test[idx:idx + 1].astype(np.int64)
            result = measure_single_sample(
                sess, ops, x_nat, y_nat, init_method,
                int(num_restarts), float(eps), float(alpha),
                int(total_iter), int(df_max_iter), float(df_overshoot),
                int(seed),
            )
            all_results.append(result)

    return all_results
