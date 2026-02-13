"""Find samples correctly classified by all models for cross-model comparison."""

import argparse
import json
import os
from typing import List, Set

import numpy as np

from src.dataset_config import resolve_dataset_config


def validate_model_src_dir(model_src_dir: str) -> None:
    """Validate that the model source directory exists.

    Args:
        model_src_dir: Path to the model source directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not os.path.isdir(model_src_dir):
        raise FileNotFoundError(
            f"Model source directory not found: {model_src_dir}\n"
            f"Ensure the model source is placed at the expected location."
        )


def validate_checkpoint_dir(ckpt_dir: str) -> None:
    """Validate that the checkpoint directory exists.

    Args:
        ckpt_dir: Path to the checkpoint directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"Ensure the model checkpoint is placed at the expected location."
        )


def get_correct_indices_for_model(
    model_src_dir: str,
    ckpt_dir: str,
    dataset: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Set[int]:
    """Get indices of correctly classified samples for a single model."""
    # Lazy import: TF 1.15.5 crashes on ARM at import time.
    import tensorflow as tf

    from src.dto import ModelOps
    from src.model_loader import (
        create_tf_session,
        instantiate_model,
        load_model_module,
        restore_checkpoint,
    )

    validate_checkpoint_dir(ckpt_dir)

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

            preds = sess.run(
                ops.y_pred_op,
                feed_dict={ops.x_ph: x_batch, ops.y_ph: y_batch},
            )

            for i, (pred, true_label) in enumerate(zip(preds, y_batch)):
                if int(pred) == int(true_label):
                    correct_indices.add(start + i)

    return correct_indices


def find_common_correct_indices(
    dataset: str,
    model_src_dir: str,
    models: List[str],
    num_classes: int = 10,
    samples_per_class: int = 1,
    seed: int = 0,
) -> List[int]:
    """Find samples per class that are correctly classified by all models.

    For each class (0 to num_classes-1), randomly select N samples
    from those that all models classify correctly.
    """
    # Lazy import: data_loader depends on TF at import time.
    from src.data_loader import load_test_data

    validate_model_src_dir(model_src_dir)

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

    # Randomly select samples per class
    rng = np.random.RandomState(seed)
    selected_indices: List[int] = []

    for class_label in range(num_classes):
        # Get all valid indices for this class
        class_indices = [
            idx for idx in common_indices if y_test[idx] == class_label
        ]

        if len(class_indices) >= samples_per_class:
            # Randomly select without replacement
            chosen = rng.choice(
                class_indices, size=samples_per_class, replace=False
            )
            selected_indices.extend(chosen.tolist())
            print(
                f"  Class {class_label}: randomly selected "
                f"{samples_per_class} from {len(class_indices)} candidates"
            )
        else:
            # Not enough samples, take all available
            selected_indices.extend(class_indices)
            print(
                f"  Warning: Class {class_label} only has "
                f"{len(class_indices)}/{samples_per_class} samples"
            )

    return selected_indices


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for finding common correct samples.

    Returns:
        Configured ArgumentParser with dataset, output, model,
        and sampling options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Find test samples correctly classified by all specified models. "
            "Outputs a JSON file containing the selected sample indices for "
            "cross-model comparison experiments."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        required=True,
        help="Dataset name (mnist or cifar10).",
    )
    parser.add_argument(
        "--out_dir",
        default="docs",
        help=(
            "Output directory for the JSON file containing selected indices. "
            "(default: docs)"
        ),
    )
    parser.add_argument(
        "--models",
        default="nat,adv,nat_and_adv,weak_adv",
        help=(
            "Comma-separated list of model names whose checkpoints reside "
            "under <model_src_dir>/models/<name>/. "
            "(default: nat,adv,nat_and_adv,weak_adv)"
        ),
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1,
        help=(
            "Number of samples to randomly select per class. "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sample selection. (default: 0)",
    )
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = str(args.dataset)
    out_dir = str(args.out_dir)
    models = [m.strip() for m in str(args.models).split(",")]
    samples_per_class = int(args.samples_per_class)
    seed = int(args.seed)
    num_classes = 10

    # Resolve model_src_dir from centralized dataset config
    cfg = resolve_dataset_config(dataset)
    model_src_dir = cfg.model_src_dir

    print(f"Finding common correct indices for {dataset}")
    print(f"Models: {models}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Seed: {seed}")

    selected_indices = find_common_correct_indices(
        dataset=dataset,
        model_src_dir=model_src_dir,
        models=models,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    total_samples = num_classes * samples_per_class
    out_file = os.path.join(
        out_dir, f"common_correct_indices_{dataset}_n{total_samples}.json"
    )

    result = {
        "dataset": dataset,
        "models": models,
        "selected_indices": selected_indices,
        "num_classes": num_classes,
        "samples_per_class": samples_per_class,
        "seed": seed,
        "total_count": len(selected_indices),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"\nSaved {len(selected_indices)} common correct indices to: "
        f"{out_file}"
    )


if __name__ == "__main__":
    main()
