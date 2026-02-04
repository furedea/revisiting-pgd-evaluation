"""
Analyze timing results and generate LaTeX table.

Usage:
    python analyze_timing.py --input_dir outputs/timing/timing_ex10 --out_dir outputs
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def load_timing_results(input_dir: str) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Load timing results from individual JSON files.

    Returns:
        Dict[dataset, Dict[method_key, List[timing_dict]]]
        where method_key is like "random_n1", "deepfool_n1", etc.
    """
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

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
    """Compute mean and std for timing results."""
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


def generate_comparison_table(data: Dict[str, List[Dict[str, float]]], dataset: str) -> str:
    """Generate comparison table (random vs DeepFool-based)."""
    # Check which methods are available
    available = set(data.keys())

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(f"  \\caption{{{dataset.upper()}における初期化手法の実行時間比較}}")
    lines.append(f"  \\label{{table:{dataset}_timing_comparison}}")
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
    """Print timing summary to console."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze timing results")
    parser.add_argument("--input_dir", required=True, help="Directory containing timing JSON files")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    all_data = load_timing_results(args.input_dir)

    if not all_data:
        print("[ERROR] No timing results found")
        return

    # Create output directory
    result_dir = os.path.join(args.out_dir, "timing_analysis")
    os.makedirs(result_dir, exist_ok=True)

    for dataset in sorted(all_data.keys()):
        data = all_data[dataset]

        # Print summary
        print_summary(data, dataset)

        # Generate and save LaTeX table
        table = generate_comparison_table(data, dataset)
        out_file = os.path.join(result_dir, f"{dataset}_timing_comparison.tex")

        with open(out_file, "w") as f:
            f.write(table)
        print(f"\n[SAVE] {out_file}")

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
