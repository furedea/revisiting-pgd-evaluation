"""
Analyze timing results and generate LaTeX table.

Loads per-sample timing JSON files produced by the timing measurement module,
computes statistics (mean, std) for each initialization method, and generates
LaTeX comparison tables.

Usage:
    python analyze_timing.py --input_dir outputs/timing/timing_ex100 --out_dir outputs
    python analyze_timing.py --input_dir outputs/timing/timing_ex100 --dataset mnist
    python analyze_timing.py --help
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.dataset_config import resolve_dataset_config


def infer_exp_name(input_dir: str) -> str:
    """Infer experiment name from input directory path.

    Extracts the final path component as the experiment name,
    stripping any trailing slashes.

    Args:
        input_dir: Path to the timing results directory.

    Returns:
        Experiment name string (e.g., "timing_ex100").
    """
    return os.path.basename(os.path.normpath(input_dir))


def validate_input_dir(input_dir: str) -> None:
    """Validate that input directory exists and contains timing JSON files.

    Args:
        input_dir: Path to the timing results directory.

    Raises:
        FileNotFoundError: If directory does not exist or contains no
            timing JSON files, with a clear message including the path.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            f"Please verify the path exists."
        )

    has_timing_files = any(
        fname.startswith("timing_") and fname.endswith(".json")
        for fname in os.listdir(input_dir)
    )
    if not has_timing_files:
        raise FileNotFoundError(
            f"No timing JSON files found in: {input_dir}\n"
            f"Expected files matching 'timing_*.json' pattern.\n"
            f"Example: timing_mnist_nat_random_n1.json"
        )


def load_timing_results(input_dir: str) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Load timing results from individual JSON files.

    Scans the input directory for files matching 'timing_*.json',
    parses each file, and groups results by dataset and method key.

    Args:
        input_dir: Path to the directory containing timing JSON files.

    Returns:
        Nested dict: results[dataset][method_key] = list of timing dicts.
        Each timing dict has keys "init", "pgd", "total" with float values.
        method_key is formatted as "{init_method}_n{num_restarts}".
    """
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    if not os.path.isdir(input_dir):
        return results

    for filename in os.listdir(input_dir):
        if filename.startswith("timing_") and filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath) as f:
                data = json.load(f)

            dataset = data["dataset"]
            init_method = data["init"]
            num_restarts = data["num_restarts"]
            method_key = f"{init_method}_n{num_restarts}"

            if dataset not in results:
                results[dataset] = {}

            if method_key not in results[dataset]:
                results[dataset][method_key] = []

            # Extend with all sample results
            results[dataset][method_key].extend(data["results"])

    return results


def compute_statistics(timing_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and std for timing results.

    Args:
        timing_list: List of timing dicts, each with "init", "pgd", "total".

    Returns:
        Dict with keys: init_mean, init_std, pgd_mean, pgd_std,
        total_mean, total_std, n_samples. All values are Python floats.
    """
    init_times = [r["init"] for r in timing_list]
    pgd_times = [r["pgd"] for r in timing_list]
    total_times = [r["total"] for r in timing_list]

    return {
        "init_mean": float(np.mean(init_times)),
        "init_std": float(np.std(init_times)),
        "pgd_mean": float(np.mean(pgd_times)),
        "pgd_std": float(np.std(pgd_times)),
        "total_mean": float(np.mean(total_times)),
        "total_std": float(np.std(total_times)),
        "n_samples": len(timing_list),
    }


def generate_comparison_table(
    data: Dict[str, List[Dict[str, float]]], dataset: str, exp_name: str
) -> str:
    """Generate comparison table (random vs DeepFool-based).

    Args:
        data: Dict[method_key, List[timing_dict]] for a single dataset.
        dataset: Dataset name (e.g., "mnist").
        exp_name: Experiment name for table caption.

    Returns:
        LaTeX table as a string.
    """
    available = set(data.keys())

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(f"  \\caption{{{dataset.upper()}における初期化手法の実行時間比較（{exp_name}）}}")
    lines.append(f"  \\label{{table:{dataset}_timing_comparison_{exp_name}}}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{c|l|ccc|c}")
    lines.append("    \\hline")
    lines.append("    リスタート数 & 初期化手法 & 初期点生成 (s) & PGD (s) & 合計 (s) & 比率 \\\\")
    lines.append("    \\hline")

    # n=1 comparison
    if "random_n1" in available and "deepfool_n1" in available:
        stats_r1 = compute_statistics(data["random_n1"])
        stats_d1 = compute_statistics(data["deepfool_n1"])
        ratio_1 = stats_d1["total_mean"] / stats_r1["total_mean"] if stats_r1["total_mean"] > 0 else 0

        lines.append(
            f"    \\multirow{{2}}{{*}}{{1}} & ランダム & "
            f"{stats_r1['init_mean']:.4f} & {stats_r1['pgd_mean']:.4f} & "
            f"{stats_r1['total_mean']:.4f} & 1.00 \\\\"
        )
        lines.append(
            f"     & DeepFool & "
            f"{stats_d1['init_mean']:.4f} & {stats_d1['pgd_mean']:.4f} & "
            f"{stats_d1['total_mean']:.4f} & {ratio_1:.2f} \\\\"
        )
        lines.append("    \\hline")

    # n=9 comparison
    if "random_n9" in available and "multi_deepfool_n9" in available:
        stats_r9 = compute_statistics(data["random_n9"])
        stats_m9 = compute_statistics(data["multi_deepfool_n9"])
        ratio_9 = stats_m9["total_mean"] / stats_r9["total_mean"] if stats_r9["total_mean"] > 0 else 0

        lines.append(
            f"    \\multirow{{2}}{{*}}{{9}} & ランダム & "
            f"{stats_r9['init_mean']:.4f} & {stats_r9['pgd_mean']:.4f} & "
            f"{stats_r9['total_mean']:.4f} & 1.00 \\\\"
        )
        lines.append(
            f"     & Multi-DeepFool & "
            f"{stats_m9['init_mean']:.4f} & {stats_m9['pgd_mean']:.4f} & "
            f"{stats_m9['total_mean']:.4f} & {ratio_9:.2f} \\\\"
        )
        lines.append("    \\hline")

    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_summary(data: Dict[str, List[Dict[str, float]]], dataset: str) -> None:
    """Print timing summary to console.

    Args:
        data: Dict[method_key, List[timing_dict]] for a single dataset.
        dataset: Dataset name (e.g., "mnist").
    """
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'=' * 80}")
    print(f"{'Method':<25} {'N':<8} {'Init (s)':<12} {'PGD (s)':<12} {'Total (s)':<12}")
    print("-" * 80)

    methods = ["random_n1", "deepfool_n1", "random_n9", "multi_deepfool_n9"]
    stats_dict = {}

    for method in methods:
        if method in data:
            stats = compute_statistics(data[method])
            stats_dict[method] = stats
            print(
                f"{method:<25} "
                f"{stats['n_samples']:<8} "
                f"{stats['init_mean']:.4f}       "
                f"{stats['pgd_mean']:.4f}       "
                f"{stats['total_mean']:.4f}"
            )

    print("-" * 80)
    print("\nComparison:")

    if "random_n1" in stats_dict and "deepfool_n1" in stats_dict:
        ratio_1 = stats_dict["deepfool_n1"]["total_mean"] / stats_dict["random_n1"]["total_mean"]
        print(f"  n=1: deepfool / random = {ratio_1:.2f}x")

    if "random_n9" in stats_dict and "multi_deepfool_n9" in stats_dict:
        ratio_9 = stats_dict["multi_deepfool_n9"]["total_mean"] / stats_dict["random_n9"]["total_mean"]
        print(f"  n=9: multi_deepfool / random = {ratio_9:.2f}x")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for timing analysis.

    Returns:
        Configured ArgumentParser with --input_dir, --out_dir,
        and --dataset arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze timing results from PGD attack experiments. "
            "Loads per-sample timing JSON files, computes statistics "
            "(mean, std) for each initialization method (random, deepfool, "
            "multi_deepfool), and generates LaTeX comparison tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analyze_timing.py --input_dir outputs/timing/timing_ex100\n"
            "  python analyze_timing.py --input_dir outputs/timing/timing_ex100 "
            "--dataset mnist\n"
            "  python analyze_timing.py --input_dir outputs/timing/timing_ex100 "
            "--out_dir /custom/output"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing timing JSON files (e.g., outputs/timing/timing_ex100)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Base output directory for analysis results (default: outputs)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["mnist", "cifar10"],
        help="Filter results to a specific dataset (default: process all datasets)",
    )
    return parser


def main() -> None:
    """Main entry point for timing analysis."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate input directory
    validate_input_dir(args.input_dir)

    # Load timing results
    all_data = load_timing_results(args.input_dir)

    if not all_data:
        print("[ERROR] No timing results found in: " + args.input_dir)
        return

    # Infer experiment name from input directory
    exp_name = infer_exp_name(args.input_dir)

    # Filter by dataset if specified
    if args.dataset is not None:
        if args.dataset not in all_data:
            print(f"[ERROR] Dataset '{args.dataset}' not found in timing results.")
            print(f"Available datasets: {sorted(all_data.keys())}")
            return
        all_data = {args.dataset: all_data[args.dataset]}

    # Resolve dataset config for informational display
    for dataset in sorted(all_data.keys()):
        try:
            cfg = resolve_dataset_config(dataset)
            print(f"[INFO] Dataset '{dataset}' config: "
                  f"epsilon={cfg.epsilon}, alpha={cfg.alpha}")
        except ValueError:
            pass

    # Create output directory with exp_name
    result_dir = os.path.join(args.out_dir, "timing_analysis", exp_name)
    os.makedirs(result_dir, exist_ok=True)

    for dataset in sorted(all_data.keys()):
        data = all_data[dataset]

        # Print summary
        print_summary(data, dataset)

        # Generate and save LaTeX table
        table = generate_comparison_table(data, dataset, exp_name)
        out_file = os.path.join(result_dir, f"{dataset}_timing_comparison.tex")

        with open(out_file, "w") as f:
            f.write(table)
        print(f"\n[SAVE] {out_file}")

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
