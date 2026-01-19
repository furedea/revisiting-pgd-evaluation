"""
PGD loss/correctness visualization (MNIST/CIFAR10, 1..5 panels) with optional
DeepFool init.

- Single-file script.
- Python 3.6.9 compatible.
- No external libs beyond: numpy, matplotlib, tensorflow, tqdm.
"""

import os
import sys
import argparse
import inspect
import importlib.util
import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import numpy as np
import tensorflow as tf
from tqdm import tqdm


matplotlib.use("Agg")
tf.compat.v1.disable_eager_execution()


# ============================================================
# logging
# ============================================================
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# DTOs (immutable by convention)
# ============================================================
class ModelOps:
    """Graph ops required for prediction/loss/grad and DeepFool."""

    __slots__ = (
        "x_ph",
        "y_ph",
        "logits",
        "logits_name",
        "y_pred_op",
        "per_ex_loss_op",
        "grad_op",
        "grads_all_op",
    )

    def __init__(
        self,
        x_ph: Any,
        y_ph: Any,
        logits: Any,
        logits_name: str,
        y_pred_op: Any,
        per_ex_loss_op: Any,
        grad_op: Any,
        grads_all_op: Any,
    ) -> None:
        self.x_ph = x_ph
        self.y_ph = y_ph
        self.logits = logits
        self.logits_name = logits_name
        self.y_pred_op = y_pred_op
        self.per_ex_loss_op = per_ex_loss_op
        self.grad_op = grad_op
        self.grads_all_op = grads_all_op

    @classmethod
    def from_model(cls, model: Any) -> "ModelOps":
        x_ph = model.x_input
        y_ph = model.y_input

        if hasattr(model, "pre_softmax"):
            logits = model.pre_softmax
            logits_name = "pre_softmax"
        elif hasattr(model, "logits"):
            logits = model.logits
            logits_name = "logits"
        else:
            raise AttributeError("Model has neither 'pre_softmax' nor 'logits'.")

        y_pred_op = tf.argmax(logits, axis=1, output_type=tf.int64)

        per_ex_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_ph,
            logits=logits,
        )
        loss_sum = tf.reduce_sum(per_ex_loss_op)
        grad_op = tf.gradients(loss_sum, x_ph)[0]

        num_classes = int(logits.shape[-1])
        grads_k = []
        for k in range(num_classes):
            grads_k.append(tf.gradients(logits[:, k], x_ph)[0])
        grads_all_op = tf.stack(grads_k, axis=1)

        return cls(
            x_ph=x_ph,
            y_ph=y_ph,
            logits=logits,
            logits_name=logits_name,
            y_pred_op=y_pred_op,
            per_ex_loss_op=per_ex_loss_op,
            grad_op=grad_op,
            grads_all_op=grads_all_op,
        )


class InitSanityMetrics:
    __slots__ = (
        "true_label",
        "nat_pred",
        "nat_loss",
        "df_pred",
        "linf_df",
        "df_loss",
        "init_pred",
        "linf_init",
        "init_loss",
    )

    def __init__(
        self,
        true_label: int,
        nat_pred: int,
        nat_loss: float,
        df_pred: Optional[int],
        linf_df: Optional[float],
        df_loss: Optional[float],
        init_pred: Optional[int],
        linf_init: Optional[float],
        init_loss: Optional[float],
    ) -> None:
        self.true_label = int(true_label)
        self.nat_pred = int(nat_pred)
        self.nat_loss = float(nat_loss)
        self.df_pred = None if df_pred is None else int(df_pred)
        self.linf_df = None if linf_df is None else float(linf_df)
        self.df_loss = None if df_loss is None else float(df_loss)
        self.init_pred = None if init_pred is None else int(init_pred)
        self.linf_init = None if linf_init is None else float(linf_init)
        self.init_loss = None if init_loss is None else float(init_loss)


class PGDBatchResult:
    """Batched PGD result for one example (R restarts)."""

    __slots__ = ("losses", "preds", "corrects", "x_adv_final")

    def __init__(
        self,
        losses: np.ndarray,
        preds: np.ndarray,
        corrects: np.ndarray,
        x_adv_final: np.ndarray,
    ) -> None:
        self.losses = losses
        self.preds = preds
        self.corrects = corrects
        self.x_adv_final = x_adv_final


class ExamplePanel:
    """All info needed to render one panel (one test example)."""

    __slots__ = (
        "x_nat",
        "y_nat",
        "x_df",
        "x_adv_show",
        "show_restart",
        "pred_end",
        "pgd",
        "sanity",
        "test_idx",
    )

    def __init__(
        self,
        x_nat: np.ndarray,
        y_nat: np.ndarray,
        x_adv_show: np.ndarray,
        show_restart: int,
        pred_end: int,
        pgd: PGDBatchResult,
        sanity: Optional[InitSanityMetrics],
        x_df: Optional[np.ndarray] = None,
        test_idx: int = -1,
    ) -> None:
        self.x_nat = x_nat
        self.y_nat = y_nat
        self.x_df = x_df
        self.x_adv_show = x_adv_show
        self.show_restart = show_restart
        self.pred_end = pred_end
        self.pgd = pgd
        self.sanity = sanity
        self.test_idx = test_idx


# ============================================================
# import / model creation
# ============================================================
def add_sys_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def load_module_from_path(module_name: str, file_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec: {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instantiate_model(model_module: Any, mode_default: str) -> Any:
    if not hasattr(model_module, "Model"):
        raise AttributeError("model.py has no Model class.")
    model_cls = model_module.Model

    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.keys())  # includes 'self'
    return model_cls() if len(params) <= 1 else model_cls(mode_default)


def latest_ckpt(ckpt_dir: str) -> str:
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")
    return ckpt


def load_model_module(model_src_dir: str, dataset: str) -> Any:
    add_sys_path(model_src_dir)
    model_py = os.path.join(model_src_dir, "model.py")
    if not os.path.exists(model_py):
        raise FileNotFoundError(f"model.py not found: {model_py}")
    return load_module_from_path(f"challenge_model_{dataset}", model_py)


# ============================================================
# data loaders
# ============================================================
def load_mnist_flattened(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    x_test = mnist.test.images.astype(np.float32)
    y_test = mnist.test.labels.astype(np.int64)
    return x_test, y_test


def load_cifar10_float01() -> Tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.reshape(-1).astype(np.int64)
    return x_test, y_test


# ============================================================
# math helpers
# ============================================================
def linf_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(np.max(np.abs(x1.astype(np.float32) - x2.astype(np.float32))))


def project_linf(x: np.ndarray, x_nat: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(x, x_nat - eps, x_nat + eps).astype(np.float32)


def clip_to_unit_interval(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ============================================================
# selection / diagnostics
# ============================================================
def log_nat_sanity( # TODO: name
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
) -> None:
    pred = sess.run(ops.y_pred_op, feed_dict={ops.x_ph: x_nat, ops.y_ph: y_nat})
    loss_vec = sess.run(ops.per_ex_loss_op, feed_dict={ops.x_ph: x_nat, ops.y_ph: y_nat})

    true_label = int(y_nat.reshape(-1)[0])
    pred_label = int(pred[0])
    loss0 = float(loss_vec[0])

    LOGGER.info(f"[clean] true={true_label} pred={pred_label} loss={loss0:.6g}")


def log_init_sanity(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    x_df: Optional[np.ndarray],
    x_init: Optional[np.ndarray],
    eps: float,
) -> Optional[InitSanityMetrics]:
    """DeepFool init の妥当性チェック（PGD直前に1回だけ出す想定）."""
    y0 = int(y_nat.reshape(-1)[0])

    def eval_point(x: np.ndarray) -> Tuple[int, float]:
        pred, loss_vec = sess.run(
            [ops.y_pred_op, ops.per_ex_loss_op],
            feed_dict={ops.x_ph: x, ops.y_ph: y_nat},
        )
        return int(pred[0]), float(loss_vec[0])

    nat_pred, nat_loss = eval_point(x_nat)

    df_pred: Optional[int] = None
    df_loss: Optional[float] = None
    linf_df: Optional[float] = None
    if x_df is not None:
        df_pred, df_loss = eval_point(x_df)
        linf_df = linf_distance(x_df, x_nat)

    init_pred: Optional[int] = None
    init_loss: Optional[float] = None
    linf_init: Optional[float] = None
    if x_init is not None:
        init_pred, init_loss = eval_point(x_init)
        linf_init = linf_distance(x_init, x_nat)

    LOGGER.info(
        "init_sanity true=%d | nat(pred=%d loss=%.6f) | "
        "df(pred=%s loss=%s linf_df=%.6f eps=%.6f) | "
        "init(pred=%s loss=%s linf_init=%s)",
        y0,
        nat_pred,
        nat_loss,
        "NA" if df_pred is None else str(df_pred),
        "NA" if df_loss is None else f"{df_loss:.6f}",
        -1.0 if linf_df is None else float(linf_df),
        float(eps),
        "NA" if init_pred is None else str(init_pred),
        "NA" if init_loss is None else f"{init_loss:.6f}",
        "NA" if linf_init is None else f"{float(linf_init):.6f}",
    )

    if x_df is None and x_init is None:
        return None

    return InitSanityMetrics(
        true_label=y0,
        nat_pred=nat_pred,
        nat_loss=nat_loss,
        df_pred=df_pred,
        linf_df=linf_df,
        df_loss=df_loss,
        init_pred=init_pred,
        linf_init=linf_init,
        init_loss=init_loss,
    )


# ============================================================
# Multi-DeepFool helpers
# ============================================================
def compute_perturbation_to_target(
    f: np.ndarray,
    grads_all: np.ndarray,
    start_label: int,
    target_class: int,
) -> Tuple[np.ndarray, float]:
    """Compute perturbation toward target class from gradients."""
    grad_start = grads_all[start_label].reshape(-1).astype(np.float32)
    w = grads_all[target_class].reshape(-1).astype(np.float32) - grad_start
    denom = float(np.dot(w, w) + 1e-12)
    r_flat = (abs(float(f[target_class])) / denom) * w
    r_norm = float(np.linalg.norm(r_flat))
    return r_flat.astype(np.float32), r_norm


def deepfool_single_target(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x0: np.ndarray,
    target_class: int,
    start_label: int,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """DeepFool targeting a specific class."""
    x = x0.astype(np.float32).copy()

    for i in range(int(max_iter)):
        logits = sess.run(ops.logits, feed_dict={ops.x_ph: x})
        current_label = int(np.argmax(logits[0]))

        # Stop if reached target or left start region
        if current_label == target_class:
            LOGGER.info(f"[deepfool_single] reached target={target_class} at iter={i}")
            break
        if current_label != start_label and current_label != target_class:
            LOGGER.info(f"[deepfool_single] diverted to class={current_label} at iter={i}")
            break

        grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x})[0]
        f = logits[0] - logits[0, start_label]

        # Compute perturbation toward target class
        r_flat, _ = compute_perturbation_to_target(f, grads_all, start_label, target_class)

        r = r_flat.reshape(x.shape[1:])
        x = x + (1.0 + float(overshoot)) * r[np.newaxis, ...]
        x = np.clip(x, float(clip_min), float(clip_max)).astype(np.float32)

    return x.astype(np.float32)


def multi_deepfool_init_points(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x0: np.ndarray,
    top_k: int,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
) -> Tuple[np.ndarray, ...]:
    """Multi-DeepFool with multiple target classes (top-K closest decision boundaries).

    Returns tuple of (top_k,) adversarial examples, one per target class.
    """
    x = x0.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(ops.logits.shape[-1])

    LOGGER.info(
        f"[multi_deepfool] start label={start_label} top_k={top_k} "
        f"max_iter={int(max_iter)} overshoot={float(overshoot)}"
    )

    # Collect initial perturbations for all target classes
    grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x})[0]
    f = logits0[0] - logits0[0, start_label]

    # Compute distance to each target class
    target_distances = []
    for target in range(num_classes):
        if target == start_label:
            continue
        r_flat, r_norm = compute_perturbation_to_target(f, grads_all, start_label, target)
        target_distances.append((r_norm, target, r_flat))

    # Sort by distance and take top K
    target_distances.sort(key=lambda x: x[0])
    k_actual = min(int(top_k), len(target_distances))

    if k_actual < int(top_k):
        raise ValueError(
            f"Not enough target classes: requested {top_k}, "
            f"but only {k_actual} available (num_classes={num_classes})"
        )

    LOGGER.info(f"[multi_deepfool] selecting {k_actual} closest targets")

    # Generate one adversarial example per target
    x_advs = []
    for idx in range(k_actual):
        _, target_class, initial_r = target_distances[idx]

        LOGGER.info(f"[multi_deepfool] target {idx+1}/{k_actual}: class={target_class}")

        # Run DeepFool targeting this specific class
        x_adv = deepfool_single_target(
            sess=sess,
            ops=ops,
            x0=x0,
            target_class=target_class,
            start_label=start_label,
            max_iter=max_iter,
            overshoot=overshoot,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        x_advs.append(x_adv)

    return tuple(x_advs)


def multi_deepfool_with_trace(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x0: np.ndarray,
    y_nat: np.ndarray,
    top_k: int,
    max_iter: int,
    overshoot: float,
    clip_min: float,
    clip_max: float,
    eps: float,
    do_clip: bool,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Multi-DeepFool with trajectory recording.

    Records loss/pred at each iteration for each target class, clipped to eps-ball.

    Returns:
        x_advs: tuple of (top_k,) final adversarial examples (clipped to eps)
        losses: (top_k, max_iter+1) - loss at each iteration
        preds: (top_k, max_iter+1) - pred at each iteration
    """
    x = x0.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(ops.logits.shape[-1])

    LOGGER.info(
        f"[multi_deepfool_trace] start label={start_label} top_k={top_k} "
        f"max_iter={int(max_iter)} overshoot={float(overshoot)}"
    )

    # Collect initial perturbations for all target classes
    grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x})[0]
    f = logits0[0] - logits0[0, start_label]

    # Compute distance to each target class
    target_distances = []
    for target in range(num_classes):
        if target == start_label:
            continue
        r_flat, r_norm = compute_perturbation_to_target(f, grads_all, start_label, target)
        target_distances.append((r_norm, target, r_flat))

    # Sort by distance and take top K
    target_distances.sort(key=lambda x: x[0])
    k_actual = min(int(top_k), len(target_distances))

    if k_actual < int(top_k):
        raise ValueError(
            f"Not enough target classes: requested {top_k}, "
            f"but only {k_actual} available (num_classes={num_classes})"
        )

    LOGGER.info(f"[multi_deepfool_trace] selecting {k_actual} closest targets")

    # Initialize arrays for trajectory recording
    losses = np.zeros((k_actual, int(max_iter) + 1), dtype=np.float32)
    preds = np.zeros((k_actual, int(max_iter) + 1), dtype=np.int64)

    # Record initial state (step 0 = x_nat)
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

        LOGGER.info(f"[multi_deepfool_trace] target {idx+1}/{k_actual}: class={target_class}")

        # DeepFool iterations with trajectory recording
        x_curr = x0.astype(np.float32).copy()
        curr_label = start_label

        for i in range(int(max_iter)):
            # Compute gradients and logits
            logits = sess.run(ops.logits, feed_dict={ops.x_ph: x_curr})
            current_pred = int(np.argmax(logits[0]))

            # Update if still at start label
            if current_pred == start_label:
                grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x_curr})[0]
                f_curr = logits[0] - logits[0, start_label]

                r_flat, _ = compute_perturbation_to_target(
                    f_curr, grads_all, start_label, target_class
                )

                r = r_flat.reshape(x_curr.shape[1:])
                x_curr = x_curr + (1.0 + float(overshoot)) * r[np.newaxis, ...]
                x_curr = np.clip(x_curr, float(clip_min), float(clip_max)).astype(np.float32)

            # Project to eps-ball and record
            x_proj = project_linf(x_curr, x0, float(eps))
            x_proj = clip_to_unit_interval(x_proj) if do_clip else x_proj

            lt, pt = sess.run(
                [ops.per_ex_loss_op, ops.y_pred_op],
                feed_dict={ops.x_ph: x_proj, ops.y_ph: y_nat},
            )
            losses[idx, i + 1] = float(lt[0])
            preds[idx, i + 1] = int(pt[0])

        # Final point (projected to eps-ball)
        x_final = project_linf(x_curr, x0, float(eps))
        x_final = clip_to_unit_interval(x_final) if do_clip else x_final
        x_advs.append(x_final)

    return tuple(x_advs), losses, preds


def run_multi_deepfool_init_pgd(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    eps: float,
    alpha: float,
    total_steps: int,
    num_restarts: int,
    df_max_iter: int,
    df_overshoot: float,
    seed: int,
    do_clip: bool,
) -> PGDBatchResult:
    """Run multi_deepfool followed by PGD.

    total_steps is the total number of iterations:
    - step 0: x_nat (initial point)
    - step 1 to df_max_iter: multi_deepfool iterations
    - step df_max_iter+1 to total_steps: PGD iterations

    Returns PGDBatchResult with losses/preds of shape (num_restarts, total_steps+1).
    """
    restarts = int(num_restarts)
    df_iter = int(df_max_iter)
    pgd_steps = int(total_steps) - df_iter

    if pgd_steps < 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= df_max_iter ({df_max_iter})"
        )

    LOGGER.info(
        f"[multi_deepfool_pgd] restarts={restarts} df_iter={df_iter} "
        f"pgd_steps={pgd_steps} total_steps={total_steps}"
    )

    # Initialize arrays for full trajectory
    losses = np.zeros((restarts, int(total_steps) + 1), dtype=np.float32)
    preds = np.zeros((restarts, int(total_steps) + 1), dtype=np.int64)

    # Run multi_deepfool with trace
    x_advs, df_losses, df_preds = multi_deepfool_with_trace(
        sess=sess,
        ops=ops,
        x0=x_nat,
        y_nat=y_nat,
        top_k=restarts,
        max_iter=df_iter,
        overshoot=float(df_overshoot),
        clip_min=0.0,
        clip_max=1.0,
        eps=float(eps),
        do_clip=bool(do_clip),
    )

    # Copy multi_deepfool trajectory (step 0 to df_max_iter)
    losses[:, : df_iter + 1] = df_losses
    preds[:, : df_iter + 1] = df_preds

    # Prepare batch for PGD
    y_batch = np.repeat(y_nat.astype(np.int64), restarts, axis=0)
    x_nat_batch = np.repeat(x_nat.astype(np.float32), restarts, axis=0)

    # Stack x_advs into batch
    x_adv = np.concatenate([x.astype(np.float32) for x in x_advs], axis=0)

    # Run PGD from multi_deepfool endpoints
    for t in tqdm(range(1, pgd_steps + 1), desc="PGD", unit="step", leave=False):
        grad = sess.run(ops.grad_op, feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch})
        x_adv = x_adv + float(alpha) * np.sign(grad).astype(np.float32)
        x_adv = project_linf(x_adv, x_nat_batch, float(eps))
        x_adv = clip_to_unit_interval(x_adv) if do_clip else x_adv

        lt, pt = sess.run(
            [ops.per_ex_loss_op, ops.y_pred_op],
            feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch},
        )
        losses[:, df_iter + t] = lt.astype(np.float32)
        preds[:, df_iter + t] = pt.astype(np.int64)

    corrects = (preds == y_batch[:, None]).astype(bool)

    return PGDBatchResult(
        losses=losses,
        preds=preds,
        corrects=corrects,
        x_adv_final=x_adv.astype(np.float32),
    )


def find_correct_indices(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
    start_idx: int,
    max_tries: int,
) -> Tuple[int, ...]:
    n = int(len(x_test))
    idx = int(start_idx)
    found_indices = []
    tries = 0

    while len(found_indices) < int(k) and tries < min(int(max_tries), n):
        x = x_test[idx : idx + 1]
        y = y_test[idx : idx + 1]
        pred = sess.run(ops.y_pred_op, feed_dict={ops.x_ph: x, ops.y_ph: y})
        if int(pred[0]) == int(y.reshape(-1)[0]):
            found_indices.append(idx)
        idx = (idx + 1) % n
        tries += 1

    if len(found_indices) < int(k):
        raise RuntimeError(f"Could not find {k} correct examples (found {len(found_indices)}).")

    return tuple(int(i) for i in found_indices)


# ============================================================
# PGD
# ============================================================
def build_initial_points(
    rng: np.random.RandomState,
    init: str,
    x_init: Optional[Tuple[np.ndarray, ...]],
    x_nat_batch: np.ndarray,
    eps: float,
    do_clip: bool,
) -> np.ndarray:
    if init == "multi_deepfool":
        if x_init is None:
            raise ValueError("init='multi_deepfool' requires x_init (tuple of init points).")
        if not isinstance(x_init, tuple):
            raise TypeError("init='multi_deepfool' requires x_init to be a tuple.")

        num_restarts = x_nat_batch.shape[0]
        num_init_points = len(x_init)

        # Cycle through init points to fill all restarts
        x_adv_list = []
        for r in range(num_restarts):
            idx = r % num_init_points
            x_adv_list.append(x_init[idx].astype(np.float32))

        x_adv = np.concatenate(x_adv_list, axis=0)
        x_adv = project_linf(x_adv, x_nat_batch, float(eps))
        return clip_to_unit_interval(x_adv) if do_clip else x_adv

    # random init
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
    x_init: Optional[Tuple[np.ndarray, ...]],
) -> PGDBatchResult:
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
    wrong_at_end = np.where(~pgd_result.corrects[:, -1])[0]
    return int(wrong_at_end[0]) if len(wrong_at_end) > 0 else int(np.argmax(pgd_result.losses[:, -1]))


# ============================================================
# saving / plotting
# ============================================================
def save_panel_arrays(
    out_dir: str,
    base: str,
    panel_index: int,
    panel: ExamplePanel,
) -> None:
    arrays_dir = os.path.join(out_dir, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)

    np.save(os.path.join(arrays_dir, f"{base}_p{panel_index}_losses.npy"), panel.pgd.losses)
    np.save(os.path.join(arrays_dir, f"{base}_p{panel_index}_preds.npy"), panel.pgd.preds)
    np.save(
        os.path.join(arrays_dir, f"{base}_p{panel_index}_corrects.npy"),
        panel.pgd.corrects.astype(np.uint8),
    )


def format_panel_metadata(panel: ExamplePanel, panel_index: int, args: argparse.Namespace) -> str:
    """Format metadata for a single panel."""
    losses = panel.pgd.losses
    preds = panel.pgd.preds
    corrects = panel.pgd.corrects
    true_label = int(panel.y_nat.reshape(-1)[0])
    attack_success_rate = float(np.sum(~corrects[:, -1])) / float(corrects.shape[0])
    initial_losses = losses[:, 0]
    final_losses = losses[:, -1]

    nat_pred_line = f"nat_pred={panel.sanity.nat_pred}\n" if panel.sanity is not None else ""
    nat_loss_line = f"nat_loss={panel.sanity.nat_loss:.6f}\n" if panel.sanity is not None else ""

    content = (
        f"[PANEL_{panel_index}]\n"
        f"test_index={panel.test_idx}\n"
        f"true_label={true_label}\n"
        f"restart_shown={panel.show_restart}\n"
        f"pred_end={panel.pred_end}\n"
        f"{nat_pred_line}final_preds={','.join(str(int(p)) for p in preds[:, -1])}\n"
        f"attack_success_rate={attack_success_rate:.6f}\n"
        f"{nat_loss_line}initial_loss_min={float(np.min(initial_losses)):.6f}\n"
        f"initial_loss_max={float(np.max(initial_losses)):.6f}\n"
        f"initial_loss_mean={float(np.mean(initial_losses)):.6f}\n"
        f"initial_loss_median={float(np.median(initial_losses)):.6f}\n"
        f"final_loss_min={float(np.min(final_losses)):.6f}\n"
        f"final_loss_max={float(np.max(final_losses)):.6f}\n"
        f"final_loss_mean={float(np.mean(final_losses)):.6f}\n"
        f"final_loss_median={float(np.median(final_losses)):.6f}\n"
    )

    if panel.sanity is not None and args.init == "multi_deepfool":
        df_lines = ""
        if panel.sanity.df_pred is not None:
            df_lines += f"df_pred={panel.sanity.df_pred}\n"
        if panel.sanity.df_loss is not None:
            df_lines += f"df_loss={panel.sanity.df_loss:.6f}\n"
        if panel.sanity.linf_df is not None:
            df_lines += f"linf_df={panel.sanity.linf_df:.6f}\n"
        if panel.sanity.init_pred is not None:
            df_lines += f"init_pred={panel.sanity.init_pred}\n"
        if panel.sanity.init_loss is not None:
            df_lines += f"init_loss={panel.sanity.init_loss:.6f}\n"
        if panel.sanity.linf_init is not None:
            df_lines += f"linf_init={panel.sanity.linf_init:.6f}\n"
        content += df_lines

    return content


def plot_panels(
    dataset: str,
    panels: Tuple[ExamplePanel, ...],
    out_png: str,
    title: str,
    alpha_line: float,
    init_sanity_plot: bool,
    eps: float,
) -> None:
    num_panels = int(len(panels))
    if num_panels < 1 or num_panels > 5:
        raise ValueError(f"len(panels) must be 1..5, got {num_panels}")

    cmap = ListedColormap(["#440154", "#FDE725"])

    # ---- sanity row is shown only if at least one sample has linf_df > eps
    has_df_over_eps = any(
        (p.sanity is not None)
        and (p.sanity.linf_df is not None)
        and (float(p.sanity.linf_df) > float(eps))
        for p in panels
    )

    # user flag + multi_deepfool + df_over_eps only
    show_sanity_row = bool(init_sanity_plot) and bool(has_df_over_eps)

    # nrows = 4 if bool(init_sanity_plot) else 3
    # height_ratios = [3.2, 1.5, 1.6, 1.6] if nrows == 4 else [3.2, 1.5, 1.6]
    nrows = 4 if show_sanity_row else 3
    height_ratios = [3.2, 1.5, 1.6, 1.6] if show_sanity_row else [3.2, 1.5, 1.6]

    # fig_w = 4.2 * num_panels
    fig_w = min(3.6 * num_panels, 16.0) # TODO
    fig_h = 10.5 if nrows == 4 else 9.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=14, y=0.995)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=num_panels,
        height_ratios=height_ratios,
        hspace=0.65 if nrows == 4 else 0.55, # 問題があれば修正
        wspace=0.35,
    )

    fig.legend(
        handles=[
            Patch(facecolor="#FDE725", edgecolor="none", label="Correct"),
            Patch(facecolor="#440154", edgecolor="none", label="Wrong"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )

    for col, panel in enumerate(panels):
        losses = panel.pgd.losses
        corrects = panel.pgd.corrects
        restarts, steps_plus1 = losses.shape
        xs = np.arange(steps_plus1)

        ax1 = fig.add_subplot(gs[0, col])
        for r in range(restarts):
            ax1.plot(xs, losses[r], linewidth=1, alpha=float(alpha_line))
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Cross-entropy Loss" if col == 0 else "")
        ax1.tick_params(labelbottom=True)

        # --- disable scientific notation like "1e-5" on y-axis (top plot)
        sf = ScalarFormatter(useOffset=False)
        sf.set_scientific(False)
        ax1.yaxis.set_major_formatter(sf)
        ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax1.yaxis.get_offset_text().set_visible(False)

        ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
        ax2.imshow(
            corrects.astype(np.int8),
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("restart (run)" if col == 0 else "")
        ax2.set_yticks([0, restarts - 1])
        ax2.set_yticklabels(["0", str(restarts - 1)] if col == 0 else ["", ""])
        ax2.tick_params(labelbottom=True)

        # Display 3 images if x_df exists, otherwise 2
        num_imgs = 3 if panel.x_df is not None else 2
        sub = gs[2, col].subgridspec(1, num_imgs, wspace=0.08)
        ax3a = fig.add_subplot(sub[0, 0])
        if num_imgs == 3:
            ax3b = fig.add_subplot(sub[0, 1])
            ax3c = fig.add_subplot(sub[0, 2])
        else:
            ax3c = fig.add_subplot(sub[0, 1])

        ax3a.axis("off")
        ax3c.axis("off")
        ax3a.set_title("x_nat", fontsize=11, pad=6)
        ax3c.set_title("x_adv", fontsize=11, pad=6)

        if num_imgs == 3:
            ax3b.axis("off")
            ax3b.set_title("x_df", fontsize=11, pad=6)

        if dataset == "mnist":
            ax3a.imshow(
                np.squeeze(panel.x_nat).reshape(28, 28),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            if num_imgs == 3 and panel.x_df is not None:
                ax3b.imshow(
                    np.squeeze(panel.x_df).reshape(28, 28),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
            ax3c.imshow(
                np.squeeze(panel.x_adv_show).reshape(28, 28),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
        else:
            ax3a.imshow(
                np.clip(np.squeeze(panel.x_nat).reshape(32, 32, 3), 0.0, 1.0),
                vmin=0.0,
                vmax=1.0,
            )
            if num_imgs == 3 and panel.x_df is not None:
                ax3b.imshow(
                    np.clip(np.squeeze(panel.x_df).reshape(32, 32, 3), 0.0, 1.0),
                    vmin=0.0,
                    vmax=1.0,
                )
            ax3c.imshow(
                np.clip(np.squeeze(panel.x_adv_show).reshape(32, 32, 3), 0.0, 1.0),
                vmin=0.0,
                vmax=1.0,
            )

    # --- appended sanity plot (bottom, spanning all columns)
    # --- init sanity plot (bottom row, per-sample mini plots)
    if show_sanity_row:
        # ----------------------------
        # Collect ranges (df/init only)
        # ----------------------------
        y_vals = []
        x_max = float(eps)

        for p in panels:
            if p.sanity is None:
                continue
            if p.sanity.linf_df is None:
                continue
            if float(p.sanity.linf_df) <= float(eps):
                continue  # ★df>eps の列だけでレンジを作る

            x_max = max(x_max, float(p.sanity.linf_df))
            if p.sanity.linf_init is not None:
                x_max = max(x_max, float(p.sanity.linf_init))
            if p.sanity.df_loss is not None:
                y_vals.append(float(p.sanity.df_loss))
            if p.sanity.init_loss is not None:
                y_vals.append(float(p.sanity.init_loss))

        if x_max <= 0.0:
            x_max = 1.0

        # robust y-limits (linear)
        if len(y_vals) > 0:
            y_lo = float(np.percentile(y_vals, 5))
            y_hi = float(np.percentile(y_vals, 95))
            pad = 0.20 * max(1e-12, (y_hi - y_lo))
            y0 = max(0.0, y_lo - pad)
            y1 = y_hi + pad
            if y1 <= y0:
                y1 = y0 + 1.0
            y_lim = (y0, y1)
        else:
            y_lim = (0.0, 1.0)

        # ----------------------------
        # Draw per-sample mini plots
        # ----------------------------
        for col, p in enumerate(panels):
            ax = fig.add_subplot(gs[3, col])

            ok = (
                (p.sanity is not None)
                and (p.sanity.linf_df is not None)
                and (float(p.sanity.linf_df) > float(eps))
                and (p.sanity.df_loss is not None)
                and (p.sanity.linf_init is not None)
                and (p.sanity.init_loss is not None)
                and (p.sanity.df_pred is not None)
                and (p.sanity.init_pred is not None)
            )
            if not ok:
                ax.axis("off")   # ★詰めない：この列は空白として残す
                continue

            ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
            ax.axvline(float(eps), linestyle=":", linewidth=1.2, alpha=0.8)
            sf = ScalarFormatter(useOffset=False)
            sf.set_scientific(False)
            ax.yaxis.set_major_formatter(sf)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.yaxis.get_offset_text().set_visible(False)

            ax.set_xlim(0.0, x_max)

            ax.set_xlabel("Linf")
            ax.set_ylabel("Cross-entropy Loss")

            if p.sanity is None:
                ax.set_title(f"s{col+1} (no sanity)", fontsize=11)
                continue

            # Title contains all label info (NO annotate)
            t = p.sanity.true_label
            natp = p.sanity.nat_pred
            dfp = p.sanity.df_pred
            inp = p.sanity.init_pred
            ax.set_title(f"label: true={t} nat={natp} df={dfp} init={inp}", fontsize=11)

            # require df/init points
            if (
                p.sanity.linf_df is None or p.sanity.df_loss is None or p.sanity.df_pred is None or
                p.sanity.linf_init is None or p.sanity.init_loss is None or p.sanity.init_pred is None
            ):
                continue

            x_df = float(p.sanity.linf_df)
            y_df = float(p.sanity.df_loss)
            x_in = float(p.sanity.linf_init)
            y_in = float(p.sanity.init_loss)

            # --- per-panel y-limits (like top plots)
            y_min = min(y_df, y_in)
            y_max = max(y_df, y_in)

            # padding (avoid zero range)
            span = max(1e-12, y_max - y_min)
            pad = 0.20 * span

            y0 = max(0.0, y_min - pad)   # CE lossなので下は0に寄せる
            y1 = y_max + pad

            # 2点がほぼ同じ値で潰れるケース対策
            if y1 <= y0:
                y1 = y0 + 1e-3

            ax.set_ylim(y0, y1)

            # Ensure we draw df -> init (expected: right AND down if loss decreases)
            ax.plot([x_df, x_in], [y_df, y_in], linewidth=2.0)

            # Markers: df ▲, init ■
            ax.plot([x_df], [y_df], marker="^", linestyle="None", markersize=10)
            ax.plot([x_in], [y_in], marker="s", linestyle="None", markersize=10)

            ax.annotate(
                f"{y_in:.6f}",
                xy=(x_in, y_in),
                xytext=(6, 10),              # ← -10 じゃなく +10
                textcoords="offset points",
                ha="left",
                va="bottom",                 # ← top じゃなく bottom
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.6),
            )

        # ----------------------------
        # ONE legend for the whole figure (outside axes)
        # ----------------------------
        shape_legend = [
            Line2D([0], [0], marker="^", linestyle="None", markersize=10, label="x_df"),
            Line2D([0], [0], marker="s", linestyle="None", markersize=10, label="x_init"),
            Line2D([0], [0], linestyle=":", linewidth=1.2, label="eps"),
        ]
        fig.legend(
            handles=shape_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        # Give room for the bottom legend
        # (later you already call fig.subplots_adjust; update bottom there too)

    fig.subplots_adjust(top=0.90 if nrows == 4 else 0.88, bottom=0.11, left=0.06, right=0.99)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    LOGGER.info(f"[save] figure={out_png}")


# ============================================================
# args / naming
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--model_src_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=1)
    ap.add_argument("--max_tries", type=int, default=20000)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--num_restarts", type=int, default=20)

    ap.add_argument("--no_clip", action="store_true") # TODO: いる？
    ap.add_argument("--tag", default="naturally_trained")
    ap.add_argument("--alpha_line", type=float, default=0.9)
    ap.add_argument("--check_only", action="store_true")

    ap.add_argument("--init", choices=["random", "multi_deepfool"], default="random")
    ap.add_argument("--df_max_iter", type=int, default=50)
    ap.add_argument("--df_overshoot", type=float, default=0.02)
    ap.add_argument("--init_sanity_plot", action="store_true")

    return ap


def validate_args(args: argparse.Namespace) -> None:
    if int(args.n_examples) < 1 or int(args.n_examples) > 5:
        raise ValueError("--n_examples must be 1..5")
    if str(args.init) == "multi_deepfool" and int(args.df_max_iter) <= 0:
        raise ValueError("--df_max_iter must be > 0")



def format_indices_part(indices: Tuple[int, ...]) -> str:
    return f"idx{indices[0]}" if len(indices) == 1 else f"indices{'-'.join(str(i) for i in indices)}"


def format_base_name(args: argparse.Namespace, indices: Tuple[int, ...]) -> str:
    idx_part = format_indices_part(indices)
    df_part = (
        f"_dfiter{args.df_max_iter}_dfo{args.df_overshoot}"
        if args.init == "multi_deepfool"
        else ""
    )
    clip_part = "_noclip" if args.no_clip else ""

    return (
        f"{args.dataset}_{args.tag}_{args.init}_{idx_part}_k{args.steps}"
        f"_eps{args.epsilon}_a{args.alpha}_r{args.num_restarts}_seed{args.seed}"
        f"{df_part}{clip_part}"
    )


def format_title(args: argparse.Namespace) -> str:
    return f"{args.dataset.upper()} loss curves ({args.tag}, {args.init}-init)"


# ============================================================
# orchestration (split)
# ============================================================
def create_tf_session() -> tf.compat.v1.Session:
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=cfg)


def restore_checkpoint(
    sess: tf.compat.v1.Session,
    saver: tf.compat.v1.train.Saver,
    ckpt_dir: str,
) -> str:
    ckpt = latest_ckpt(ckpt_dir)
    saver.restore(sess, ckpt)
    LOGGER.info(f"[restore] {ckpt}")
    return ckpt


def load_test_data(
    args: argparse.Namespace,
    model_src_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if args.dataset == "mnist":
        data_dir = os.path.join(model_src_dir, "MNIST_data")
        return load_mnist_flattened(data_dir)
    return load_cifar10_float01()


def run_check_only(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    start_idx: int,
) -> None:
    x_nat = x_test[start_idx : start_idx + 1].astype(np.float32)
    y_nat = y_test[start_idx : start_idx + 1].astype(np.int64)
    log_nat_sanity(sess, ops, x_nat, y_nat)
    LOGGER.info({"action": "exit", "reason": "check_only"})


def run_one_example(
    args: argparse.Namespace,
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    idx: int,
) -> ExamplePanel:
    x_nat = x_test[idx : idx + 1].astype(np.float32)
    y_nat = y_test[idx : idx + 1].astype(np.int64)

    log_nat_sanity(sess, ops, x_nat, y_nat)

    do_clip = not bool(args.no_clip)

    x_df = None

    if args.init == "multi_deepfool":
        # Run multi_deepfool + PGD with trajectory recording
        pgd = run_multi_deepfool_init_pgd(
            sess=sess,
            ops=ops,
            x_nat=x_nat,
            y_nat=y_nat,
            eps=float(args.epsilon),
            alpha=float(args.alpha),
            total_steps=int(args.steps),
            num_restarts=int(args.num_restarts),
            df_max_iter=int(args.df_max_iter),
            df_overshoot=float(args.df_overshoot),
            seed=int(args.seed),
            do_clip=do_clip,
        )
        # For visualization, x_df is the first restart's final point
        x_df = pgd.x_adv_final[0:1].astype(np.float32)
    else:
        # Random init: run standard PGD
        pgd = run_pgd_batch(
            sess=sess,
            ops=ops,
            x_nat=x_nat,
            y_nat=y_nat,
            eps=float(args.epsilon),
            alpha=float(args.alpha),
            steps=int(args.steps),
            num_restarts=int(args.num_restarts),
            seed=int(args.seed),
            do_clip=do_clip,
            init=str(args.init),
            x_init=None,
        )

    # Sanity logging (x_df and x_init for multi_deepfool case)
    sanity = log_init_sanity(
        sess=sess,
        ops=ops,
        x_nat=x_nat,
        y_nat=y_nat,
        x_df=x_df,
        x_init=x_df,  # For multi_deepfool, same as x_df
        eps=float(args.epsilon),
    )

    show_restart = choose_show_restart(pgd)
    x_adv_show = pgd.x_adv_final[show_restart : show_restart + 1].astype(np.float32)
    pred_end = int(pgd.preds[show_restart, -1])

    wrong_end = int(np.sum(~pgd.corrects[:, -1]))
    loss_end = pgd.losses[:, -1]
    LOGGER.info(
        f"[pgd] wrong_end={wrong_end}/{int(args.num_restarts)} "
        f"loss_end(med/min/max)={float(np.median(loss_end)):.4g}/"
        f"{float(np.min(loss_end)):.4g}/{float(np.max(loss_end)):.4g}"
    )

    return ExamplePanel(
        x_nat=x_nat,
        y_nat=y_nat,
        x_adv_show=x_adv_show,
        show_restart=int(show_restart),
        pred_end=int(pred_end),
        pgd=pgd,
        sanity=sanity,
        x_df=x_df,
        test_idx=int(idx),
    )


def run_all_examples(
    args: argparse.Namespace,
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: Tuple[int, ...],
) -> Tuple[ExamplePanel, ...]:
    panels = []
    for idx in indices:
        panels.append(run_one_example(args, sess, ops, x_test, y_test, int(idx)))
    return tuple(panels)


def save_all_outputs(
    args: argparse.Namespace,
    base: str,
    panels: Tuple[ExamplePanel, ...],
) -> None:
    # Save arrays for each panel
    for i, panel in enumerate(panels, start=1):
        save_panel_arrays(
            out_dir=str(args.out_dir),
            base=str(base),
            panel_index=int(i),
            panel=panel,
        )

    # Save unified metadata file
    metadata_dir = os.path.join(args.out_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_txt = os.path.join(metadata_dir, f"{base}_meta.txt")

    df_params = ""
    if args.init == "multi_deepfool":
        df_params = (
            f"df_max_iter={args.df_max_iter}\n"
            f"df_overshoot={args.df_overshoot}\n"
        )

    content = (
        "[PARAMETERS]\n"
        f"dataset={args.dataset}\n"
        f"epsilon={args.epsilon}\n"
        f"alpha={args.alpha}\n"
        f"steps={args.steps}\n"
        f"num_restarts={args.num_restarts}\n"
        f"seed={args.seed}\n"
        f"init={args.init}\n"
        f"clip={not args.no_clip}\n"
        f"{df_params}\n"
    )

    for i, panel in enumerate(panels, start=1):
        content += format_panel_metadata(panel, i, args)

    with open(meta_txt, "w", encoding="utf-8") as f:
        f.write(content)

    LOGGER.info(f"[save] metadata={meta_txt}")


def render_figure(
    args: argparse.Namespace,
    base: str,
    title: str,
    panels: Tuple[ExamplePanel, ...],
) -> str:
    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_png = os.path.join(figures_dir, f"{base}.png")
    plot_panels(
        dataset=str(args.dataset),
        panels=panels,
        out_png=out_png,
        title=str(title),
        alpha_line=float(args.alpha_line),
        init_sanity_plot=bool(args.init_sanity_plot) and (str(args.init) == "multi_deepfool"),
        eps=float(args.epsilon),
    )
    return out_png


def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    model_src_dir = os.path.abspath(args.model_src_dir)

    tf.compat.v1.reset_default_graph()
    model_module = load_model_module(model_src_dir, str(args.dataset))
    model = instantiate_model(model_module, mode_default="eval")
    ops = ModelOps.from_model(model)

    saver = tf.compat.v1.train.Saver()

    with create_tf_session() as sess:
        restore_checkpoint(sess, saver, str(args.ckpt_dir))
        x_test, y_test = load_test_data(args, model_src_dir)

        if bool(args.check_only):
            run_check_only(sess, ops, x_test, y_test, int(args.start_idx))
            return

        indices = find_correct_indices(
            sess=sess,
            ops=ops,
            x_test=x_test,
            y_test=y_test,
            k=int(args.n_examples),
            start_idx=int(args.start_idx),
            max_tries=int(args.max_tries),
        )
        base = format_base_name(args, indices)
        title = format_title(args)

        df_part = (
            f" df_iter={args.df_max_iter} df_overshoot={args.df_overshoot}"
            if args.init == "multi_deepfool"
            else ""
        )
        LOGGER.info(
            f"[run] dataset={args.dataset} indices={indices} tag={args.tag} init={args.init}"
            f" eps={args.epsilon} alpha={args.alpha} steps={args.steps} restarts={args.num_restarts}"
            f" seed={args.seed}{df_part} out={args.out_dir}"
        )

        panels = run_all_examples(args, sess, ops, x_test, y_test, indices)

    save_all_outputs(args, base, panels)
    out_png = render_figure(args, base, title, panels)

    LOGGER.info(f"[done] figure={out_png}")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
