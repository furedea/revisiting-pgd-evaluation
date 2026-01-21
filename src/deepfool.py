"""DeepFool algorithm for adversarial initialization."""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.dto import ModelOps
from src.logging_config import LOGGER
from src.math_utils import clip_to_unit_interval, project_linf, scale_to_linf_ball


def deepfool_init_point(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x0: np.ndarray,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
    verbose: bool,
) -> np.ndarray:
    """Compute DeepFool adversarial point."""
    x = x0.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(logits0.shape[1])

    if verbose:
        LOGGER.info(
            f"[deepfool] start label={start_label} "
            f"max_iter={int(max_iter)} overshoot={float(overshoot)}"
        )

    for i in range(int(max_iter)):
        logits = sess.run(ops.logits, feed_dict={ops.x_ph: x})
        current_label = int(np.argmax(logits[0]))
        if current_label != start_label:
            if verbose:
                LOGGER.info(f"[deepfool] stop iter={int(i)} label {start_label}->{current_label}")
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
            if verbose:
                LOGGER.info("[deepfool] stop (no direction found)")
            break

        r = best_r_flat.reshape(x.shape[1:])
        x = x + r[np.newaxis, ...]
        x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

    # Apply overshoot once at the end (per original paper)
    x = x0 + (1.0 + float(overshoot)) * (x - x0)
    x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

    return x.astype(np.float32)


def deepfool_init_point_with_trace(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x0: np.ndarray,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
    verbose: bool,
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """DeepFool that also returns trajectory points (including x0 and each updated x)."""
    x = x0.astype(np.float32).copy()
    trace = [x.copy()]

    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(logits0.shape[1])

    if verbose:
        LOGGER.info(
            f"[deepfool] start label={start_label} "
            f"max_iter={int(max_iter)} overshoot={float(overshoot)}"
        )

    for i in range(int(max_iter)):
        logits = sess.run(ops.logits, feed_dict={ops.x_ph: x})
        current_label = int(np.argmax(logits[0]))
        if current_label != start_label:
            if verbose:
                LOGGER.info(f"[deepfool] stop iter={int(i)} label {start_label}->{current_label}")
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
            if verbose:
                LOGGER.info("[deepfool] stop (no direction found)")
            break

        r = best_r_flat.reshape(x.shape[1:])
        x = x + r[np.newaxis, ...]
        x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

        trace.append(x.copy())

    # Apply overshoot once at the end (per original paper)
    x = x0 + (1.0 + float(overshoot)) * (x - x0)
    x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

    return x.astype(np.float32), tuple(trace)


def select_maxloss_within_eps(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    xs: Tuple[np.ndarray, ...],
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    eps: float,
    project: str = "clip",
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Pick argmax loss among projected points onto Linf-ball."""
    best_x = None
    best_loss = None

    for x in xs:
        x = x.astype(np.float32)

        if project == "clip":
            x_proj = project_linf(x, x_nat, float(eps))
        elif project == "scale":
            x_proj = scale_to_linf_ball(x, x_nat, float(eps))
            x_proj = project_linf(x_proj, x_nat, float(eps))
        else:
            raise ValueError(f"Unknown project: {project}")

        x_proj = clip_to_unit_interval(x_proj)

        loss_vec = sess.run(
            ops.per_ex_loss_op,
            feed_dict={ops.x_ph: x_proj, ops.y_ph: y_nat},
        )
        loss = float(loss_vec[0])

        if (best_loss is None) or (loss > best_loss):
            best_loss = loss
            best_x = x_proj

    return best_x, best_loss


def build_deepfool_init(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    df_max_iter: int,
    df_overshoot: float,
    df_project: str,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build DeepFool initialization point with projection."""
    if df_project == "maxloss":
        x_df, trace = deepfool_init_point_with_trace(
            sess=sess,
            ops=ops,
            x0=x_nat,
            max_iter=int(df_max_iter),
            overshoot=float(df_overshoot),
            clip_min=0.0,
            clip_max=1.0,
            verbose=True,
        )
    else:
        x_df = deepfool_init_point(
            sess=sess,
            ops=ops,
            x0=x_nat,
            max_iter=int(df_max_iter),
            overshoot=float(df_overshoot),
            clip_min=0.0,
            clip_max=1.0,
            verbose=True,
        )
        trace = None

    if df_project == "clip":
        x_init = project_linf(x_df, x_nat, eps)

    elif df_project == "scale":
        x_init = scale_to_linf_ball(x_df, x_nat, eps)
        x_init = project_linf(x_init, x_nat, eps)

    elif df_project == "maxloss":
        assert trace is not None
        best_x, best_loss = select_maxloss_within_eps(
            sess=sess,
            ops=ops,
            xs=trace,
            x_nat=x_nat,
            y_nat=y_nat,
            eps=eps,
        )
        if best_x is None:
            LOGGER.info("[deepfool] maxloss: no trace point within eps; fallback to scale")
            x_init = project_linf(scale_to_linf_ball(x_df, x_nat, eps), x_nat, eps)
        else:
            LOGGER.info(f"[deepfool] maxloss: picked loss={best_loss:.6g} within eps")
            x_init = best_x.astype(np.float32)
            x_init = project_linf(x_init, x_nat, eps)

    else:
        raise ValueError(f"Unknown df_project: {df_project}")

    x_init = clip_to_unit_interval(x_init)
    return x_df, x_init
