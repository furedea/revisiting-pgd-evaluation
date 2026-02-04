"""
Analyze timing results and generate LaTeX table.

Usage:
    python analyze_timing.py --input_dir outputs/timing --out_dir outputs
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def load_timing_results(input_dir: str) -> Dict[str, Dict]:
    """Load timing results from JSON files."""
    results = {}
    for filename in os.listdir(input_dir):
        if filename.startswith("timing_") and filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath) as f:
                data = json.load(f)
            dataset = data["dataset"]
            results[dataset] = data
    return results


def compute_statistics(
    results: List[Dict[str, Dict[str, float]]], method: str
) -> Dict[str, float]:
    """Compute mean and std for a method across samples."""
    init_times = [r[method]["init"] for r in results]
    pgd_times = [r[method]["pgd"] for r in results]
    total_times = [r[method]["total"] for r in results]

    return {
        "init_mean": float(np.mean(init_times)),
        "init_std": float(np.std(init_times)),
        "pgd_mean": float(np.mean(pgd_times)),
        "pgd_std": float(np.std(pgd_times)),
        "total_mean": float(np.mean(total_times)),
        "total_std": float(np.std(total_times)),
    }


def generate_latex_table(data: Dict, dataset: str) -> str:
    """Generate LaTeX table for timing comparison."""
    results = data["results"]
    n_samples = len(results)

    methods = [
        ("random_n1", "random", 1),
        ("deepfool_n1", "deepfool", 1),
        ("random_n9", "random", 9),
        ("multi_deepfool_n9", "multi\\_deepfool", 9),
    ]

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(f"  \\caption{{{dataset.upper()}における実行時間比較（{n_samples}サンプル）}}")
    lines.append(f"  \\label{{table:{dataset}_timing}}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{l|c|ccc}")
    lines.append("    \\hline")
    lines.append("    初期化手法 & リスタート数 & 初期化時間 (s) & PGD時間 (s) & 合計時間 (s) \\\\")
    lines.append("    \\hline")

    for method_key, method_name, n_restarts in methods:
        stats = compute_statistics(results, method_key)
        lines.append(
            f"    {method_name} & {n_restarts} & "
            f"{stats['init_mean']:.3f} & "
            f"{stats['pgd_mean']:.3f} & "
            f"{stats['total_mean']:.3f} \\\\"
        )

    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_comparison_table(data: Dict, dataset: str) -> str:
    """Generate comparison table (random vs DeepFool-based)."""
    results = data["results"]
    n_samples = len(results)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(f"  \\caption{{{dataset.upper()}における初期化手法の実行時間比較}}")
    lines.append(f"  \\label{{table:{dataset}_timing_comparison}}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{c|l|ccc|c}")
    lines.append("    \\hline")
    lines.append("    リスタート数 & 初期化手法 & 初期化 (s) & PGD (s) & 合計 (s) & 比率 \\\\")
    lines.append("    \\hline")

    # n=1 comparison
    stats_r1 = compute_statistics(results, "random_n1")
    stats_d1 = compute_statistics(results, "deepfool_n1")
    ratio_1 = stats_d1["total_mean"] / stats_r1["total_mean"] if stats_r1["total_mean"] > 0 else 0

    lines.append(
        f"    \\multirow{{2}}{{*}}{{1}} & random & "
        f"{stats_r1['init_mean']:.3f} & {stats_r1['pgd_mean']:.3f} & "
        f"{stats_r1['total_mean']:.3f} & 1.00x \\\\"
    )
    lines.append(
        f"     & deepfool & "
        f"{stats_d1['init_mean']:.3f} & {stats_d1['pgd_mean']:.3f} & "
        f"{stats_d1['total_mean']:.3f} & {ratio_1:.2f}x \\\\"
    )
    lines.append("    \\hline")

    # n=9 comparison
    stats_r9 = compute_statistics(results, "random_n9")
    stats_m9 = compute_statistics(results, "multi_deepfool_n9")
    ratio_9 = stats_m9["total_mean"] / stats_r9["total_mean"] if stats_r9["total_mean"] > 0 else 0

    lines.append(
        f"    \\multirow{{2}}{{*}}{{9}} & random & "
        f"{stats_r9['init_mean']:.3f} & {stats_r9['pgd_mean']:.3f} & "
        f"{stats_r9['total_mean']:.3f} & 1.00x \\\\"
    )
    lines.append(
        f"     & multi\\_deepfool & "
        f"{stats_m9['init_mean']:.3f} & {stats_m9['pgd_mean']:.3f} & "
        f"{stats_m9['total_mean']:.3f} & {ratio_9:.2f}x \\\\"
    )
    lines.append("    \\hline")

    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_summary(data: Dict, dataset: str) -> None:
    """Print timing summary to console."""
    results = data["results"]
    n_samples = len(results)

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset.upper()} ({n_samples} samples)")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'Init (s)':<12} {'PGD (s)':<12} {'Total (s)':<12}")
    print("-" * 70)

    methods = ["random_n1", "deepfool_n1", "random_n9", "multi_deepfool_n9"]
    for method in methods:
        stats = compute_statistics(results, method)
        print(
            f"{method:<20} "
            f"{stats['init_mean']:.4f}       "
            f"{stats['pgd_mean']:.4f}       "
            f"{stats['total_mean']:.4f}"
        )

    print("-" * 70)
    print("\nComparison:")
    stats_r1 = compute_statistics(results, "random_n1")
    stats_d1 = compute_statistics(results, "deepfool_n1")
    stats_r9 = compute_statistics(results, "random_n9")
    stats_m9 = compute_statistics(results, "multi_deepfool_n9")

    ratio_1 = stats_d1["total_mean"] / stats_r1["total_mean"] if stats_r1["total_mean"] > 0 else 0
    ratio_9 = stats_m9["total_mean"] / stats_r9["total_mean"] if stats_r9["total_mean"] > 0 else 0

    print(f"  n=1: deepfool / random = {ratio_1:.2f}x")
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

    for dataset, data in all_data.items():
        # Print summary
        print_summary(data, dataset)

        # Generate and save LaTeX tables
        table1 = generate_latex_table(data, dataset)
        table2 = generate_comparison_table(data, dataset)

        out_file1 = os.path.join(result_dir, f"{dataset}_timing_table.tex")
        out_file2 = os.path.join(result_dir, f"{dataset}_timing_comparison.tex")

        with open(out_file1, "w") as f:
            f.write(table1)
        print(f"\n[SAVE] {out_file1}")

        with open(out_file2, "w") as f:
            f.write(table2)
        print(f"[SAVE] {out_file2}")

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
