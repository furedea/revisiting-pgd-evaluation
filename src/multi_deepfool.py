"""Multi-DeepFool initialization for PGD attack.

Generates (K-1) diverse initial points by running DeepFool toward each
non-ground-truth target class, then selects the max-loss starting point.
"""

from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.dto import ModelOps, PGDBatchResult
from src.logging_config import LOGGER
from src.math_utils import clip_to_unit_interval, project_linf


def compute_perturbation_to_target(
    f: np.ndarray,
    grads_all: np.ndarray,
    start_label: int,
    target_class: int,
) -> Tuple[np.ndarray, float]:
    """Compute perturbation toward target class from gradients.

    Uses the DeepFool formula:
        w = grad(target) - grad(start)
        r = (|f[target]| / (||w||^2 + eps)) * w

    Args:
        f: Logit differences (num_classes,).
        grads_all: Per-class gradients (num_classes, *input_shape).
        start_label: Original predicted class.
        target_class: Target class index.

    Returns:
        r_flat: Perturbation vector (flattened, float32).
        r_norm: L2 norm of perturbation.
    """
    grad_start = grads_all[start_label].reshape(-1).astype(np.float32)
    w = grads_all[target_class].reshape(-1).astype(np.float32) - grad_start
    denom = float(np.dot(w, w) + 1e-12)
    r_flat = (abs(float(f[target_class])) / denom) * w
    r_norm = float(np.linalg.norm(r_flat))
    return r_flat.astype(np.float32), r_norm


def deepfool_single_target(
    sess,  # tf.compat.v1.Session
    ops: ModelOps,
    x0: np.ndarray,
    target_class: int,
    start_label: int,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """DeepFool targeting a specific class.

    Iteratively perturbs x toward target_class until the prediction
    changes or max_iter is reached, then applies overshoot.

    Args:
        sess: TensorFlow session.
        ops: ModelOps with graph operations.
        x0: Original input (1, *input_shape).
        target_class: Target class index.
        start_label: Original predicted class.
        max_iter: Maximum DeepFool iterations.
        overshoot: Overshoot factor applied at the end.
        clip_min: Minimum pixel value.
        clip_max: Maximum pixel value.

    Returns:
        Adversarial example (1, *input_shape) as float32.
    """
    x = x0.astype(np.float32).copy()

    for i in range(int(max_iter)):
        logits = sess.run(ops.logits, feed_dict={ops.x_ph: x})
        current_label = int(np.argmax(logits[0]))

        # Stop if reached target or left start region
        if current_label == target_class:
            LOGGER.info(
                "[deepfool_single] reached target=%d at iter=%d",
                target_class, i,
            )
            break
        if current_label != start_label and current_label != target_class:
            LOGGER.info(
                "[deepfool_single] diverted to class=%d at iter=%d",
                current_label, i,
            )
            break

        grads_all = sess.run(
            ops.grads_all_op, feed_dict={ops.x_ph: x}
        )[0]
        f = logits[0] - logits[0, start_label]

        # Compute perturbation toward target class
        r_flat, _ = compute_perturbation_to_target(
            f, grads_all, start_label, target_class
        )

        r = r_flat.reshape(x.shape[1:])
        x = x + r[np.newaxis, ...]
        x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

    # Apply overshoot once at the end (per original paper)
    x = x0 + (1.0 + float(overshoot)) * (x - x0)
    x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

    return x.astype(np.float32)


def multi_deepfool_with_trace(
    sess,  # tf.compat.v1.Session
    ops: ModelOps,
    x0: np.ndarray,
    y_nat: np.ndarray,
    top_k: int,
    max_iter: int,
    overshoot: float,
    eps: float,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Multi-DeepFool with trajectory recording.

    Runs targeted DeepFool toward each of the top_k closest non-ground-truth
    classes and records loss/prediction trajectories.

    Args:
        sess: TensorFlow session.
        ops: ModelOps with graph operations.
        x0: Original input (1, *input_shape).
        y_nat: Ground-truth label (1,).
        top_k: Number of target classes (must be <= num_classes - 1).
        max_iter: Maximum DeepFool iterations per target.
        overshoot: Overshoot factor applied at the end of each target.
        eps: L-infinity constraint radius.

    Returns:
        x_advs: Tuple of (top_k,) adversarial examples projected to eps-ball.
        losses: (top_k, max_iter+1) loss trajectory.
        preds: (top_k, max_iter+1) prediction trajectory.

    Raises:
        ValueError: If top_k exceeds available target classes.
    """
    x = x0.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(ops.logits.shape[-1])

    LOGGER.info(
        "[multi_deepfool_trace] start label=%d top_k=%d "
        "max_iter=%d overshoot=%f",
        start_label, top_k, int(max_iter), float(overshoot),
    )

    # Collect initial perturbations for all target classes
    grads_all = sess.run(
        ops.grads_all_op, feed_dict={ops.x_ph: x}
    )[0]
    f = logits0[0] - logits0[0, start_label]

    # Compute distance to each target class
    target_distances = []
    for target in range(num_classes):
        if target == start_label:
            continue
        r_flat, r_norm = compute_perturbation_to_target(
            f, grads_all, start_label, target
        )
        target_distances.append((r_norm, target, r_flat))

    # Sort by distance and take top K
    target_distances.sort(key=lambda item: item[0])
    k_actual = min(int(top_k), len(target_distances))

    if k_actual < int(top_k):
        raise ValueError(
            "Not enough target classes: requested %d, "
            "but only %d available (num_classes=%d)"
            % (top_k, k_actual, num_classes)
        )

    LOGGER.info(
        "[multi_deepfool_trace] selecting %d closest targets",
        k_actual,
    )

    # Initialize arrays for trajectory recording
    losses = np.zeros((k_actual, int(max_iter) + 1), dtype=np.float32)
    preds = np.zeros((k_actual, int(max_iter) + 1), dtype=np.int64)

    # Record initial state (iter 0 = x_nat)
    l0, p0 = sess.run(
        [ops.per_ex_loss_op, ops.y_pred_op],
        feed_dict={ops.x_ph: x0, ops.y_ph: y_nat},
    )
    losses[:, 0] = float(l0[0])
    preds[:, 0] = int(p0[0])

    # Generate trajectory for each target
    x_advs = []
    for idx in range(k_actual):
        _, target_class, _ = target_distances[idx]

        LOGGER.info(
            "[multi_deepfool_trace] target %d/%d: class=%d",
            idx + 1, k_actual, target_class,
        )

        # DeepFool iterations with trajectory recording
        x_curr = x0.astype(np.float32).copy()

        for i in range(int(max_iter)):
            logits = sess.run(
                ops.logits, feed_dict={ops.x_ph: x_curr}
            )
            current_pred = int(np.argmax(logits[0]))

            # Update if still at start label
            if current_pred == start_label:
                grads_all = sess.run(
                    ops.grads_all_op, feed_dict={ops.x_ph: x_curr}
                )[0]
                f_curr = logits[0] - logits[0, start_label]

                r_flat, _ = compute_perturbation_to_target(
                    f_curr, grads_all, start_label, target_class
                )

                r = r_flat.reshape(x_curr.shape[1:])
                x_curr = x_curr + r[np.newaxis, ...]
                x_curr = clip_to_unit_interval(x_curr)

            # Project to eps-ball and record
            x_proj = clip_to_unit_interval(
                project_linf(x_curr, x0, float(eps))
            )

            lt, pt = sess.run(
                [ops.per_ex_loss_op, ops.y_pred_op],
                feed_dict={ops.x_ph: x_proj, ops.y_ph: y_nat},
            )
            losses[idx, i + 1] = float(lt[0])
            preds[idx, i + 1] = int(pt[0])

        # Apply overshoot once at the end (per original paper)
        x_curr = x0 + (1.0 + float(overshoot)) * (x_curr - x0)
        x_curr = clip_to_unit_interval(x_curr)

        # Final point (projected to eps-ball)
        x_final = clip_to_unit_interval(
            project_linf(x_curr, x0, float(eps))
        )
        x_advs.append(x_final)

    return tuple(x_advs), losses, preds


def run_multi_deepfool_init_pgd(
    sess,  # tf.compat.v1.Session
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    eps: float,
    alpha: float,
    total_iter: int,
    num_restarts: int,
    df_max_iter: int,
    df_overshoot: float,
    seed: int,
) -> PGDBatchResult:
    """Run Multi-DeepFool initialization followed by PGD.

    Generates (num_restarts) diverse initial points via Multi-DeepFool,
    then runs PGD from each endpoint. Records only the PGD trajectory:
    - iter 0: DeepFool endpoint (PGD start point)
    - iter 1..total_iter: PGD iterations

    Args:
        sess: TensorFlow session.
        ops: ModelOps with graph operations.
        x_nat: Original input (1, *input_shape).
        y_nat: Ground-truth label (1,).
        eps: L-infinity constraint radius.
        alpha: PGD step size.
        total_iter: Number of PGD iterations.
        num_restarts: Number of restarts (must be <= num_classes - 1).
        df_max_iter: Maximum DeepFool iterations per target.
        df_overshoot: Overshoot factor for DeepFool.
        seed: Random seed (unused but kept for interface consistency).

    Returns:
        PGDBatchResult with:
            losses: (num_restarts, total_iter+1)
            preds: (num_restarts, total_iter+1)
            corrects: (num_restarts, total_iter+1) as bool
            x_adv_final: (num_restarts, *input_shape)
            x_df_endpoints: (num_restarts, *input_shape)
            x_init: (1, *input_shape) -- max-loss DeepFool point
            x_init_rank: int -- index of max-loss restart

    Raises:
        ValueError: If num_restarts > num_classes - 1.
    """
    restarts = int(num_restarts)
    df_iter = int(df_max_iter)
    pgd_iter = int(total_iter)

    LOGGER.info(
        "[multi_deepfool_pgd] restarts=%d df_iter=%d pgd_iter=%d",
        restarts, df_iter, pgd_iter,
    )

    # Initialize arrays for PGD trajectory only
    losses = np.zeros((restarts, pgd_iter + 1), dtype=np.float32)
    preds = np.zeros((restarts, pgd_iter + 1), dtype=np.int64)

    # Run Multi-DeepFool with trace (validates num_restarts internally)
    x_advs, df_losses, df_preds = multi_deepfool_with_trace(
        sess=sess,
        ops=ops,
        x0=x_nat,
        y_nat=y_nat,
        top_k=restarts,
        max_iter=df_iter,
        overshoot=float(df_overshoot),
        eps=float(eps),
    )

    # Find max-loss x_init from eps-constrained outputs
    adv_losses = []
    for x_adv_i in x_advs:
        loss_val = sess.run(
            ops.per_ex_loss_op,
            feed_dict={ops.x_ph: x_adv_i, ops.y_ph: y_nat},
        )
        adv_losses.append(float(loss_val[0]))
    x_init_rank = int(np.argmax(adv_losses))
    x_init = x_advs[x_init_rank].astype(np.float32)

    LOGGER.info(
        "[multi_deepfool_pgd] x_init selected: rank=%d loss=%.6g",
        x_init_rank, adv_losses[x_init_rank],
    )

    # Prepare batch for PGD
    y_batch = np.repeat(y_nat.astype(np.int64), restarts, axis=0)
    x_nat_batch = np.repeat(x_nat.astype(np.float32), restarts, axis=0)

    # Stack DeepFool endpoints into batch (PGD start points)
    x_df_endpoints = np.concatenate(
        [x.astype(np.float32) for x in x_advs], axis=0
    )
    x_adv = x_df_endpoints.copy()

    # Record iter 0: DeepFool endpoint loss/pred
    losses[:, 0] = df_losses[:, -1]
    preds[:, 0] = df_preds[:, -1]

    # Run PGD from Multi-DeepFool endpoints
    for t in tqdm(
        range(1, pgd_iter + 1), desc="PGD", unit="iter", leave=False
    ):
        grad = sess.run(
            ops.grad_op,
            feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch},
        )
        x_adv = x_adv + float(alpha) * np.sign(grad).astype(np.float32)
        x_adv = clip_to_unit_interval(
            project_linf(x_adv, x_nat_batch, float(eps))
        )

        lt, pt = sess.run(
            [ops.per_ex_loss_op, ops.y_pred_op],
            feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch},
        )
        losses[:, t] = lt.astype(np.float32)
        preds[:, t] = pt.astype(np.int64)

    corrects = (preds == y_batch[:, None]).astype(bool)

    return PGDBatchResult(
        losses=losses,
        preds=preds,
        corrects=corrects,
        x_adv_final=x_adv.astype(np.float32),
        x_df_endpoints=x_df_endpoints,
        x_init=x_init,
        x_init_rank=x_init_rank,
    )
