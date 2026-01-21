"""PGD attack implementation."""

from typing import Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.dto import ModelOps, PGDBatchResult
from src.math_utils import clip_to_unit_interval, project_linf


def add_jitter(rng: np.random.RandomState, x: np.ndarray, jitter: float) -> np.ndarray:
    """Add uniform noise to input."""
    if float(jitter) <= 0.0:
        return x
    noise = rng.uniform(low=-jitter, high=jitter, size=x.shape).astype(np.float32)
    return x + noise


def build_initial_points(
    rng: np.random.RandomState,
    init: str,
    x_init: Optional[np.ndarray],
    init_jitter: float,
    x_nat_batch: np.ndarray,
    eps: float,
    do_clip: bool,
) -> np.ndarray:
    """Build initial adversarial points for PGD."""
    if init == "deepfool":
        if x_init is None:
            raise ValueError("init='deepfool' requires x_init.")
        x_adv = np.repeat(x_init.astype(np.float32), x_nat_batch.shape[0], axis=0)
        x_adv = add_jitter(rng=rng, x=x_adv, jitter=float(init_jitter))
        x_adv = project_linf(x_adv, x_nat_batch, float(eps))
        return clip_to_unit_interval(x_adv) if do_clip else x_adv

    noise = rng.uniform(low=-eps, high=eps, size=x_nat_batch.shape).astype(np.float32)
    x_adv = project_linf(x_nat_batch + noise, x_nat_batch, float(eps))
    return clip_to_unit_interval(x_adv) if do_clip else x_adv


def run_pgd_batch(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    eps: float,
    alpha: float,
    steps: int,
    num_restarts: int,
    seed: int,
    do_clip: bool,
    init: str,
    x_init: Optional[np.ndarray],
    init_jitter: float,
) -> PGDBatchResult:
    """Run PGD attack with multiple restarts."""
    rng = np.random.RandomState(int(seed))
    restarts = int(num_restarts)
    total_steps = int(steps)

    y_batch = np.repeat(y_nat.astype(np.int64), restarts, axis=0)
    x_nat_batch = np.repeat(x_nat.astype(np.float32), restarts, axis=0)

    losses = np.zeros((restarts, total_steps + 1), dtype=np.float32)
    preds = np.zeros((restarts, total_steps + 1), dtype=np.int64)

    x_adv = build_initial_points(
        rng=rng,
        init=str(init),
        x_init=x_init,
        init_jitter=float(init_jitter),
        x_nat_batch=x_nat_batch,
        eps=float(eps),
        do_clip=bool(do_clip),
    )

    l0, p0 = sess.run(
        [ops.per_ex_loss_op, ops.y_pred_op],
        feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch},
    )
    losses[:, 0] = l0.astype(np.float32)
    preds[:, 0] = p0.astype(np.int64)

    for t in tqdm(range(1, total_steps + 1), desc="PGD", unit="step", leave=False):
        grad = sess.run(ops.grad_op, feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch})
        x_adv = x_adv + float(alpha) * np.sign(grad).astype(np.float32)
        x_adv = project_linf(x_adv, x_nat_batch, float(eps))
        x_adv = clip_to_unit_interval(x_adv) if do_clip else x_adv

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
    )


def choose_show_restart(pgd_result: PGDBatchResult) -> int:
    """Choose which restart to show in visualization."""
    wrong_at_end = np.where(~pgd_result.corrects[:, -1])[0]
    return int(wrong_at_end[0]) if len(wrong_at_end) > 0 else int(np.argmax(pgd_result.losses[:, -1]))
