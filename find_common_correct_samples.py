"""Find samples correctly classified by all models for cross-model comparison."""

import argparse
import json
import os
from typing import List, Set

import numpy as np
import tensorflow as tf

from src.data_loader import load_test_data
from src.dto import ModelOps
from src.model_loader import (
    create_tf_session,
    instantiate_model,
    load_model_module,
    restore_checkpoint,
)


def get_correct_indices_for_model(
    model_src_dir: str,
    ckpt_dir: str,
    dataset: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Set[int]:
    """Get indices of correctly classified samples for a single model."""
    tf.compat.v1.reset_default_graph()
    model_module = load_model_module(model_src_dir, dataset)
    model = instantiate_model(model_module, mode_default="eval")
    ops = ModelOps.from_model(model)

    saver = tf.compat.v1.train.Saver()
    correct_indices: Set[int] = set()

    with create_tf_session() as sess:
        restore_checkpoint(sess, saver, ckpt_dir)

        batch_size = 100
        n_samples = len(x_test)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = x_test[start:end]
            y_batch = y_test[start:end]

            preds = sess.run(ops.y_pred_op, feed_dict={ops.x_ph: x_batch, ops.y_ph: y_batch})

            for i, (pred, true_label) in enumerate(zip(preds, y_batch)):
                if int(pred) == int(true_label):
                    correct_indices.add(start + i)

    return correct_indices


def find_common_correct_indices(
    dataset: str,
    model_src_dir: str,
    models: List[str],
) -> List[int]:
    """Find indices of samples correctly classified by all models."""
    x_test, y_test = load_test_data(dataset, model_src_dir)

    common_indices: Set[int] = set(range(len(x_test)))

    for model_name in models:
        ckpt_dir = os.path.join(model_src_dir, "models", model_name)
        print(f"Processing model: {model_name} (ckpt: {ckpt_dir})")

        correct_indices = get_correct_indices_for_model(
            model_src_dir=model_src_dir,
            ckpt_dir=ckpt_dir,
            dataset=dataset,
            x_test=x_test,
            y_test=y_test,
        )

        print(f"  Correct: {len(correct_indices)} / {len(x_test)}")
        common_indices = common_indices.intersection(correct_indices)
        print(f"  Common so far: {len(common_indices)}")

    return sorted(common_indices)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Find samples correctly classified by all models."
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--out_dir",
        default="docs",
        help="Output directory for JSON file (default: docs)",
    )
    parser.add_argument(
        "--models",
        default="nat,adv,nat_and_adv,weak_adv",
        help="Comma-separated list of model names (default: nat,adv,nat_and_adv,weak_adv)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = str(args.dataset)
    out_dir = str(args.out_dir)
    models = [m.strip() for m in str(args.models).split(",")]

    if dataset == "mnist":
        model_src_dir = "model_src/mnist_challenge"
    else:
        model_src_dir = "model_src/cifar10_challenge"

    print(f"Finding common correct indices for {dataset}")
    print(f"Models: {models}")

    common_indices = find_common_correct_indices(
        dataset=dataset,
        model_src_dir=model_src_dir,
        models=models,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"common_correct_indices_{dataset}.json")

    result = {
        "dataset": dataset,
        "models": models,
        "common_correct_indices": common_indices,
        "total_count": len(common_indices),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(common_indices)} common correct indices to: {out_file}")


if __name__ == "__main__":
    main()
