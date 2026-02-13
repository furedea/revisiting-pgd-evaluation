"""CLI entry point for timing measurement of PGD initialization methods.

Usage:
    python -m src.timing_cli \\
        --dataset mnist \\
        --model nat \\
        --init random \\
        --num_restarts 1 \\
        --common_indices_file docs/common_correct_indices_mnist_n10.json \\
        --out_dir outputs \\
        --exp_name timing_ex100

Dataset-specific parameters (epsilon, alpha, model_src_dir) are
auto-resolved from the dataset name via src.dataset_config.
"""

import argparse
import json
import os
from typing import List

import numpy as np

from src.dataset_config import resolve_dataset_config
from src.timing import run_timing_experiment


def build_timing_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for timing CLI.

    Required arguments:
        --dataset: Dataset name (mnist or cifar10)
        --model: Model name (nat, adv, nat_and_adv, weak_adv)
        --init: Initialization method (random, deepfool, multi_deepfool)
        --num_restarts: Number of restarts for PGD
        --common_indices_file: JSON file with pre-computed common correct indices
        --out_dir: Output directory
        --exp_name: Experiment name for subdirectory

    Optional arguments:
        --total_iter: Total PGD iterations (default: 100)
        --df_max_iter: Max DeepFool iterations (default: 50)
        --df_overshoot: DeepFool overshoot (default: 0.02)
        --seed: Random seed (default: 0)
    """
    ap = argparse.ArgumentParser(
        description="Measure PGD initialization timing",
    )

    ap.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        required=True,
        help="Dataset name",
    )
    ap.add_argument(
        "--model",
        choices=["nat", "adv", "nat_and_adv", "weak_adv"],
        required=True,
        help="Model name",
    )
    ap.add_argument(
        "--init",
        choices=["random", "deepfool", "multi_deepfool"],
        required=True,
        help="Initialization method",
    )
    ap.add_argument(
        "--num_restarts",
        type=int,
        required=True,
        help="Number of restarts for PGD",
    )
    ap.add_argument(
        "--common_indices_file",
        required=True,
        help="JSON file with pre-computed common correct indices",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory",
    )
    ap.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name for subdirectory",
    )
    ap.add_argument(
        "--total_iter",
        type=int,
        default=100,
        help="Total PGD iterations (default: 100)",
    )
    ap.add_argument(
        "--df_max_iter",
        type=int,
        default=50,
        help="Max DeepFool iterations (default: 50)",
    )
    ap.add_argument(
        "--df_overshoot",
        type=float,
        default=0.02,
        help="DeepFool overshoot (default: 0.02)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    return ap


def _load_common_indices(file_path: str) -> List[int]:
    """Load pre-computed common correct indices from JSON file.

    Supports both 'selected_indices' and 'common_correct_indices' keys
    for backward compatibility.
    """
    with open(file_path) as f:
        data = json.load(f)
    if "selected_indices" in data:
        return data["selected_indices"]
    return data["common_correct_indices"]


def main() -> None:
    """Main entry point for timing CLI."""
    parser = build_timing_arg_parser()
    args = parser.parse_args()

    # Auto-resolve dataset-specific parameters
    config = resolve_dataset_config(args.dataset)
    eps = config.epsilon
    alpha = config.alpha
    model_src_dir = config.model_src_dir
    ckpt_dir = os.path.join(model_src_dir, "models", args.model)

    # Load common indices
    indices = _load_common_indices(args.common_indices_file)

    print("[INFO] Dataset: %s" % args.dataset)
    print("[INFO] Model: %s" % args.model)
    print("[INFO] Init: %s" % args.init)
    print("[INFO] Restarts: %d" % args.num_restarts)
    print("[INFO] Samples: %d" % len(indices))

    # Run timing experiment
    results = run_timing_experiment(
        dataset=args.dataset,
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

    # Save results as JSON (legacy-compatible format, Req 7.4)
    timing_dir = os.path.join(args.out_dir, "timing", args.exp_name)
    os.makedirs(timing_dir, exist_ok=True)
    out_file = os.path.join(
        timing_dir,
        "timing_%s_%s_%s_n%d.json" % (
            args.dataset, args.model, args.init, args.num_restarts,
        ),
    )

    output = {
        "dataset": args.dataset,
        "model": args.model,
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

    print("\n[INFO] Saved: %s" % out_file)

    # Summary statistics
    init_times = [r["init"] for r in results]
    pgd_times = [r["pgd"] for r in results]
    total_times = [r["total"] for r in results]

    print("\n[SUMMARY] %s/%s/%s (n=%d)" % (
        args.dataset, args.model, args.init, args.num_restarts,
    ))
    print("  Init:  mean=%.4fs, std=%.4fs" % (
        float(np.mean(init_times)), float(np.std(init_times)),
    ))
    print("  PGD:   mean=%.4fs, std=%.4fs" % (
        float(np.mean(pgd_times)), float(np.std(pgd_times)),
    ))
    print("  Total: mean=%.4fs, std=%.4fs" % (
        float(np.mean(total_times)), float(np.std(total_times)),
    ))


if __name__ == "__main__":
    main()
