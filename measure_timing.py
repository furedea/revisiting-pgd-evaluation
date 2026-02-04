"""
Measure execution time for PGD initialization methods.

Usage:
    python measure_timing.py \
        --dataset mnist \
        --model nat \
        --init random \
        --num_restarts 1 \
        --common_indices_file docs/common_correct_indices_mnist_n10.json \
        --out_dir outputs/timing
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# ============================================================
# Model loading
# ============================================================
def load_model_module(model_src_dir: str, dataset: str) -> Any:
    """Load model module from source directory."""
    import importlib.util
    import sys

    model_path = os.path.join(model_src_dir, "model.py")
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    return module


def instantiate_model(model_module: Any, mode_default: str = "eval") -> Any:
    """Instantiate model from module."""
    import inspect

    model_class = model_module.Model
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters

    if "mode" in params:
        return model_class(mode=mode_default)
    return model_class()


def create_tf_session() -> tf.compat.v1.Session:
    """Create TensorFlow session with GPU memory growth."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def restore_checkpoint(
    sess: tf.compat.v1.Session, saver: tf.compat.v1.train.Saver, ckpt_dir: str
) -> None:
    """Restore model from checkpoint."""
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    saver.restore(sess, ckpt)


class ModelOps:
    """Graph ops required for prediction/loss/grad."""

    __slots__ = (
        "x_ph", "y_ph", "logits", "y_pred_op",
        "per_ex_loss_op", "grad_op", "grads_all_op",
    )

    def __init__(
        self, x_ph: Any, y_ph: Any, logits: Any,
        y_pred_op: Any, per_ex_loss_op: Any, grad_op: Any, grads_all_op: Any,
    ) -> None:
        self.x_ph = x_ph
        self.y_ph = y_ph
        self.logits = logits
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
        elif hasattr(model, "logits"):
            logits = model.logits
        else:
            raise AttributeError("Model has neither 'pre_softmax' nor 'logits'.")

        y_pred_op = tf.argmax(logits, axis=1, output_type=tf.int64)
        per_ex_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_ph, logits=logits
        )
        grad_op = tf.gradients(per_ex_loss_op, x_ph)[0]
        grads_all_op = tf.gradients(logits, x_ph)

        return cls(
            x_ph=x_ph, y_ph=y_ph, logits=logits,
            y_pred_op=y_pred_op, per_ex_loss_op=per_ex_loss_op,
            grad_op=grad_op, grads_all_op=grads_all_op,
        )


# ============================================================
# Data loading
# ============================================================
def load_test_data(dataset: str, model_src_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data for the specified dataset."""
    if dataset == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        data_dir = os.path.join(model_src_dir, "data")
        mnist = input_data.read_data_sets(data_dir, one_hot=False)
        x_test = mnist.test.images.astype(np.float32)  # (?, 784) flattened
        y_test = mnist.test.labels.astype(np.int64)
    else:  # cifar10
        from tensorflow.keras.datasets import cifar10
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype(np.float32) / 255.0
        y_test = y_test.reshape(-1).astype(np.int64)
    return x_test, y_test


def load_common_indices(file_path: str) -> List[int]:
    """Load pre-computed common correct indices from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    if "selected_indices" in data:
        return data["selected_indices"]
    return data["common_correct_indices"]


# ============================================================
# Math utilities
# ============================================================
def clip_to_unit_interval(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def project_linf(x: np.ndarray, x_nat: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(x, x_nat - eps, x_nat + eps).astype(np.float32)


# ============================================================
# Initialization methods (timing-focused, minimal overhead)
# ============================================================
def random_init(
    rng: np.random.RandomState,
    x_nat_batch: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Random initialization within L-infinity ball."""
    noise = rng.uniform(low=-eps, high=eps, size=x_nat_batch.shape).astype(np.float32)
    x_adv = project_linf(x_nat_batch + noise, x_nat_batch, eps)
    return clip_to_unit_interval(x_adv)


def deepfool_init(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    eps: float,
    max_iter: int,
    overshoot: float,
) -> np.ndarray:
    """DeepFool initialization (single target - closest boundary)."""
    x = x_nat.astype(np.float32).copy()
    logits0 = sess.run(ops.logits, feed_dict={ops.x_ph: x})
    start_label = int(np.argmax(logits0[0]))
    num_classes = int(logits0.shape[1])

    for _ in range(max_iter):
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

    x = x_nat + (1.0 + overshoot) * (x - x_nat)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    x = project_linf(x, x_nat, eps)
    return clip_to_unit_interval(x)


def multi_deepfool_init(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    eps: float,
    max_iter: int,
    overshoot: float,
    num_targets: int,
) -> np.ndarray:
    """Multi-DeepFool initialization (multiple targets)."""
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

    target_distances.sort(key=lambda x: x[0])
    k_actual = min(num_targets, len(target_distances))

    x_advs = []
    for idx in range(k_actual):
        _, target_class = target_distances[idx]
        x_curr = x_nat.astype(np.float32).copy()

        for _ in range(max_iter):
            logits = sess.run(ops.logits, feed_dict={ops.x_ph: x_curr})
            current_pred = int(np.argmax(logits[0]))

            if current_pred == start_label:
                grads_all = sess.run(ops.grads_all_op, feed_dict={ops.x_ph: x_curr})[0]
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


# ============================================================
# PGD attack (timing-focused, no logging/tqdm)
# ============================================================
def run_pgd(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_adv: np.ndarray,
    x_nat_batch: np.ndarray,
    y_batch: np.ndarray,
    eps: float,
    alpha: float,
    total_iter: int,
) -> np.ndarray:
    """Run PGD iterations (no overhead)."""
    for _ in range(total_iter):
        grad = sess.run(ops.grad_op, feed_dict={ops.x_ph: x_adv, ops.y_ph: y_batch})
        x_adv = x_adv + alpha * np.sign(grad).astype(np.float32)
        x_adv = project_linf(x_adv, x_nat_batch, eps)
        x_adv = clip_to_unit_interval(x_adv)
    return x_adv


# ============================================================
# Timing measurement
# ============================================================
def measure_single_sample(
    sess: tf.compat.v1.Session,
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
    """Measure timing for a single init method on a single sample."""
    rng = np.random.RandomState(seed)

    # Prepare batch (not included in timing)
    x_nat_batch = np.repeat(x_nat.astype(np.float32), num_restarts, axis=0)
    y_batch = np.repeat(y_nat.astype(np.int64), num_restarts, axis=0)

    # --- Measure initialization time ---
    t0 = time.perf_counter()

    if init_method == "random":
        x_adv = random_init(rng, x_nat_batch, eps)
    elif init_method == "deepfool":
        x_adv = deepfool_init(sess, ops, x_nat, eps, df_max_iter, df_overshoot)
        # For single DeepFool, we only have 1 init point (expand if num_restarts > 1)
        if num_restarts > 1:
            x_adv = np.repeat(x_adv, num_restarts, axis=0)
    elif init_method == "multi_deepfool":
        x_adv = multi_deepfool_init(
            sess, ops, x_nat, eps, df_max_iter, df_overshoot, num_restarts
        )
    else:
        raise ValueError(f"Unknown init method: {init_method}")

    t_init = time.perf_counter() - t0

    # --- Measure PGD time ---
    t0 = time.perf_counter()
    run_pgd(sess, ops, x_adv, x_nat_batch, y_batch, eps, alpha, total_iter)
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
    """Run timing experiment for all samples."""
    tf.compat.v1.reset_default_graph()
    model_module = load_model_module(model_src_dir, dataset)
    model = instantiate_model(model_module, mode_default="eval")
    ops = ModelOps.from_model(model)

    saver = tf.compat.v1.train.Saver()
    all_results = []

    with create_tf_session() as sess:
        restore_checkpoint(sess, saver, ckpt_dir)
        x_test, y_test = load_test_data(dataset, model_src_dir)

        # Warmup run (discard timing)
        print("[INFO] Warmup run...")
        idx = indices[0]
        x_nat = x_test[idx:idx+1].astype(np.float32)
        y_nat = y_test[idx:idx+1].astype(np.int64)
        _ = measure_single_sample(
            sess, ops, x_nat, y_nat, init_method, num_restarts,
            eps, alpha, total_iter, df_max_iter, df_overshoot, seed
        )

        # Actual measurements
        for i, idx in enumerate(indices):
            print(f"[INFO] Sample {i+1}/{len(indices)} (idx={idx})")
            x_nat = x_test[idx:idx+1].astype(np.float32)
            y_nat = y_test[idx:idx+1].astype(np.int64)

            result = measure_single_sample(
                sess, ops, x_nat, y_nat, init_method, num_restarts,
                eps, alpha, total_iter, df_max_iter, df_overshoot, seed
            )
            all_results.append(result)
            print(f"       init={result['init']:.4f}s, pgd={result['pgd']:.4f}s, total={result['total']:.4f}s")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure PGD initialization timing")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--model", choices=["nat", "adv", "nat_and_adv", "weak_adv"], required=True)
    parser.add_argument("--init", choices=["random", "deepfool", "multi_deepfool"], required=True)
    parser.add_argument("--num_restarts", type=int, required=True)
    parser.add_argument("--common_indices_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--exp_name", required=True, help="Experiment name for subdirectory")
    parser.add_argument("--total_iter", type=int, default=100)
    parser.add_argument("--df_max_iter", type=int, default=50)
    parser.add_argument("--df_overshoot", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    if dataset == "mnist":
        model_src_dir = "model_src/mnist_challenge"
        ckpt_dir = f"model_src/mnist_challenge/models/{model}"
        eps = 0.3
        alpha = 0.01
    else:
        model_src_dir = "model_src/cifar10_challenge"
        ckpt_dir = f"model_src/cifar10_challenge/models/{model}"
        eps = 8.0 / 255.0
        alpha = 2.0 / 255.0

    indices = load_common_indices(args.common_indices_file)

    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Model: {model}")
    print(f"[INFO] Init: {args.init}")
    print(f"[INFO] Restarts: {args.num_restarts}")
    print(f"[INFO] Samples: {len(indices)}")

    results = run_timing_experiment(
        dataset=dataset,
        model_src_dir=model_src_dir,
        ckpt_dir=ckpt_dir,
        indices=indices,
        init_method=args.init,
        num_restarts=args.num_restarts,
        eps=eps,
        alpha=alpha,
        total_iter=args.total_iter,
        df_max_iter=args.df_max_iter,
        df_overshoot=args.df_overshoot,
        seed=args.seed,
    )

    # Save results
    timing_dir = os.path.join(args.out_dir, "timing", args.exp_name)
    os.makedirs(timing_dir, exist_ok=True)
    out_file = os.path.join(
        timing_dir,
        f"timing_{dataset}_{model}_{args.init}_n{args.num_restarts}.json"
    )

    output = {
        "dataset": dataset,
        "model": model,
        "init": args.init,
        "num_restarts": args.num_restarts,
        "indices": indices,
        "total_iter": args.total_iter,
        "df_max_iter": args.df_max_iter,
        "df_overshoot": args.df_overshoot,
        "eps": eps,
        "alpha": alpha,
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Saved: {out_file}")

    # Summary
    init_times = [r["init"] for r in results]
    pgd_times = [r["pgd"] for r in results]
    total_times = [r["total"] for r in results]

    print(f"\n[SUMMARY] {dataset}/{model}/{args.init} (n={args.num_restarts})")
    print(f"  Init:  mean={np.mean(init_times):.4f}s, std={np.std(init_times):.4f}s")
    print(f"  PGD:   mean={np.mean(pgd_times):.4f}s, std={np.std(pgd_times):.4f}s")
    print(f"  Total: mean={np.mean(total_times):.4f}s, std={np.std(total_times):.4f}s")


if __name__ == "__main__":
    main()
