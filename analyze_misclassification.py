"""
Misclassification analysis for PGD attacks with multiple samples.

Generates:
1. Heatmap: X-axis = PGD iterations, Y-axis = model/init combinations,
   Color = fraction of trials misclassified by that iteration
2. Table: LaTeX table with mean statistics across all samples

Usage:
    python analyze_misclassification.py --input_dir outputs/arrays/run_all_ex10/ --out_dir outputs
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

MODEL_ORDER = ["nat", "nat_and_adv", "adv", "weak_adv"]
INIT_ORDER = ["clean", "random", "deepfool", "multi_deepfool"]

MODEL_DISPLAY = {
    "nat": "nat",
    "nat_and_adv": "nat\\_and\\_adv",
    "adv": "adv",
    "weak_adv": "weak\\_adv",
}

INIT_DISPLAY = {
    "clean": "clean",
    "random": "random",
    "deepfool": "deepfool",
    "multi_deepfool": "multi\\_deepfool",
}

RESTART_COUNTS = {
    "clean": 1,
    "random": 20,
    "deepfool": 1,
    "multi_deepfool": 9,
}

NOT_MISCLASSIFIED = -1


class CorrectsData:
    """Container for correctness data from a single experiment."""

    __slots__ = ("filepath", "dataset", "model", "init", "panel_index", "corrects")

    def __init__(
        self,
        filepath: str,
        dataset: str,
        model: str,
        init: str,
        panel_index: int,
        corrects: np.ndarray,
    ) -> None:
        self.filepath = filepath
        self.dataset = dataset
        self.model = model
        self.init = init
        self.panel_index = panel_index
        self.corrects = corrects  # shape: (restarts, iterations+1), bool

    @property
    def num_restarts(self) -> int:
        return self.corrects.shape[0]

    @property
    def num_iterations(self) -> int:
        return self.corrects.shape[1] - 1


def parse_filename(filepath: str) -> Optional[Tuple[str, str, str, int]]:
    """Parse filename to extract dataset, model, init, panel_index."""
    basename = os.path.basename(filepath)

    panel_match = re.search(r"_p(\d+)_corrects\.npy$", basename)
    if not panel_match:
        return None
    panel_index = int(panel_match.group(1))

    prefix = basename[: panel_match.start()]

    if prefix.startswith("mnist_"):
        dataset = "mnist"
        rest = prefix[6:]
    elif prefix.startswith("cifar10_"):
        dataset = "cifar10"
        rest = prefix[8:]
    else:
        return None

    model_patterns = ["nat_and_adv", "weak_adv", "nat", "adv"]
    init_patterns = ["multi_deepfool", "deepfool", "random", "clean"]

    model = None
    init = None

    for mp in model_patterns:
        if rest.startswith(mp + "_"):
            model = mp
            rest = rest[len(mp) + 1 :]
            break

    if model is None:
        return None

    for ip in init_patterns:
        if rest.startswith(ip):
            init = ip
            break

    if init is None:
        return None

    return dataset, model, init, panel_index


def load_corrects_files(input_dir: str) -> List[CorrectsData]:
    """Load all *_corrects.npy files from input directory."""
    data_list = []

    for filename in os.listdir(input_dir):
        if not filename.endswith("_corrects.npy"):
            continue

        filepath = os.path.join(input_dir, filename)
        parsed = parse_filename(filepath)

        if parsed is None:
            print(f"[WARN] Could not parse: {filename}")
            continue

        dataset, model, init, panel_index = parsed
        corrects = np.load(filepath)

        data_list.append(
            CorrectsData(
                filepath=filepath,
                dataset=dataset,
                model=model,
                init=init,
                panel_index=panel_index,
                corrects=corrects,
            )
        )

    print(f"[INFO] Loaded {len(data_list)} corrects files from {input_dir}")
    return data_list


def compute_first_misclassification(corrects: np.ndarray) -> np.ndarray:
    """Compute iteration where first misclassification occurs for each restart.

    Args:
        corrects: shape (restarts, iterations+1), where True/1 means correct classification

    Returns:
        first_wrong: shape (restarts,)
            >= 0: iteration index of first misclassification
            NOT_MISCLASSIFIED (-1): never misclassified (attack failed)
    """
    corrects_bool = corrects.astype(bool)
    wrong = ~corrects_bool
    any_wrong = np.any(wrong, axis=1)
    first_wrong_iter = np.argmax(wrong, axis=1).astype(np.int32)
    first_wrong_iter[~any_wrong] = NOT_MISCLASSIFIED

    return first_wrong_iter


def aggregate_all_trials(
    data_list: List[CorrectsData],
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Aggregate all trials by dataset, model, and init.

    Returns:
        result[dataset][model][init] = array of first misclassification iterations
            (flattened across all samples and restarts)
    """
    result: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    for data in data_list:
        if data.dataset not in result:
            result[data.dataset] = {}
        if data.model not in result[data.dataset]:
            result[data.dataset][data.model] = {}
        if data.init not in result[data.dataset][data.model]:
            result[data.dataset][data.model][data.init] = []

        first_wrong = compute_first_misclassification(data.corrects)
        result[data.dataset][data.model][data.init].extend(first_wrong.tolist())

    # Convert to numpy arrays
    final: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for dataset in result:
        final[dataset] = {}
        for model in result[dataset]:
            final[dataset][model] = {}
            for init in result[dataset][model]:
                final[dataset][model][init] = np.array(result[dataset][model][init])

    return final


def compute_cumulative_misclassification_rate(
    first_wrong_iters: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute cumulative misclassification rate at each iteration.

    Args:
        first_wrong_iters: array of first misclassification iterations
            (-1 means never misclassified)
        max_iter: maximum iteration number

    Returns:
        rates: array of shape (max_iter+1,) with cumulative misclassification rate
            at each iteration (0 to max_iter)
    """
    n_total = len(first_wrong_iters)
    if n_total == 0:
        return np.zeros(max_iter + 1)

    rates = np.zeros(max_iter + 1)
    for t in range(max_iter + 1):
        # Count trials that have misclassified by iteration t
        n_misclassified = np.sum((first_wrong_iters >= 0) & (first_wrong_iters <= t))
        rates[t] = n_misclassified / n_total

    return rates


def plot_heatmap(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    dataset: str,
    max_iter: int = 100,
) -> None:
    """Plot heatmap of cumulative misclassification rate.

    X-axis: PGD iterations (0 to max_iter)
    Y-axis: model/init combinations
    Color: fraction of trials misclassified by that iteration
    """
    # Build row order: model/init combinations
    row_labels = []
    row_data = []

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue
            first_wrong = stats[model][init]
            rates = compute_cumulative_misclassification_rate(first_wrong, max_iter)
            row_labels.append(f"{model}/{init}")
            row_data.append(rates)

    if not row_data:
        print(f"[WARN] No data for heatmap: {dataset}")
        return

    # Create heatmap matrix
    heatmap_matrix = np.array(row_data)  # shape: (n_rows, max_iter+1)

    # Plot
    fig, ax = plt.subplots(figsize=(14, len(row_labels) * 0.5 + 2))

    im = ax.imshow(
        heatmap_matrix,
        aspect="auto",
        cmap="RdYlGn_r",  # Red = high misclassification, Green = low
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cumulative Misclassification Rate", fontsize=12)

    # Axis labels
    ax.set_xlabel("PGD Iteration", fontsize=12)
    ax.set_ylabel("Model / Init", fontsize=12)
    ax.set_title(f"Misclassification Heatmap ({dataset.upper()})", fontsize=14)

    # Y-axis ticks
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # X-axis ticks (every 10 iterations)
    x_ticks = list(range(0, max_iter + 1, 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Heatmap: {out_path}")


def generate_latex_table(
    stats: Dict[str, Dict[str, np.ndarray]],
    dataset: str,
    n_samples: int,
) -> str:
    """Generate LaTeX table with misclassification statistics.

    Format matches the thesis table format.
    """
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(f"  \\caption{{{dataset.upper()}における誤分類統計（{n_samples}サンプル平均）}}")
    lines.append(f"  \\label{{table:{dataset}_misclassification_ex{n_samples}}}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{l|l|c|cccc}")
    lines.append("    \\hline")
    lines.append("    & & & & \\multicolumn{3}{c}{初回誤分類反復数} \\\\")
    lines.append("    \\cline{5-7}")
    lines.append("    モデル & 初期化 & リスタート数 & 攻撃成功率 & 平均 & 中央値 & P95 \\\\")
    lines.append("    \\hline")

    for model in MODEL_ORDER:
        if model not in stats:
            continue

        first_model = True
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue

            first_wrong = stats[model][init]
            n_total = len(first_wrong)
            misclassified = first_wrong[first_wrong >= 0]
            n_misclassified = len(misclassified)
            attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0
            restart_count = RESTART_COUNTS.get(init, 1)

            model_col = f"\\multirow{{4}}{{*}}{{{MODEL_DISPLAY[model]}}}" if first_model else ""
            first_model = False

            if n_misclassified > 0:
                mean_val = np.mean(misclassified)
                median_val = np.median(misclassified)
                p95_val = np.percentile(misclassified, 95)
                lines.append(
                    f"    {model_col} & {INIT_DISPLAY[init]} & {restart_count} & "
                    f"{attack_rate:.0f}\\% & {mean_val:.1f} & {median_val:.1f} & {p95_val:.1f} \\\\"
                )
            else:
                lines.append(
                    f"    {model_col} & {INIT_DISPLAY[init]} & {restart_count} & "
                    f"{attack_rate:.0f}\\% & N/A & N/A & N/A \\\\"
                )

        lines.append("    \\hline")

    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def save_latex_table(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    dataset: str,
    n_samples: int,
) -> None:
    """Save LaTeX table to file."""
    latex_content = generate_latex_table(stats, dataset, n_samples)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"[SAVE] LaTeX table: {out_path}")


def print_summary(
    stats: Dict[str, Dict[str, np.ndarray]],
    dataset: str,
    n_samples: int,
) -> None:
    """Print summary statistics to console."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset.upper()} ({n_samples} samples)")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Init':<15} {'Restarts':<10} {'Success%':<10} {'Mean':<8} {'Median':<8} {'P95':<8}")
    print("-" * 80)

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue

            first_wrong = stats[model][init]
            n_total = len(first_wrong)
            misclassified = first_wrong[first_wrong >= 0]
            n_misclassified = len(misclassified)
            attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0
            restart_count = RESTART_COUNTS.get(init, 1)

            if n_misclassified > 0:
                mean_val = np.mean(misclassified)
                median_val = np.median(misclassified)
                p95_val = np.percentile(misclassified, 95)
                print(
                    f"{model:<15} {init:<15} {restart_count:<10} {attack_rate:<10.0f} "
                    f"{mean_val:<8.1f} {median_val:<8.1f} {p95_val:<8.1f}"
                )
            else:
                print(
                    f"{model:<15} {init:<15} {restart_count:<10} {attack_rate:<10.0f} "
                    f"{'N/A':<8} {'N/A':<8} {'N/A':<8}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze misclassification from multiple samples"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_corrects.npy files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/work/outputs",
        help="Base output directory (default: /work/outputs)",
    )
    args = parser.parse_args()

    result_dir = os.path.join(args.out_dir, "misclassification_analysis")
    os.makedirs(result_dir, exist_ok=True)

    data_list = load_corrects_files(args.input_dir)
    if not data_list:
        print("[ERROR] No corrects files found")
        return

    # Aggregate all trials
    all_stats = aggregate_all_trials(data_list)

    datasets = sorted(all_stats.keys())

    for dataset in datasets:
        stats = all_stats[dataset]

        # Count unique samples (panel indices)
        dataset_data = [d for d in data_list if d.dataset == dataset]
        n_samples = len(set(d.panel_index for d in dataset_data))

        # Print summary
        print_summary(stats, dataset, n_samples)

        # Create output directory for this dataset
        dataset_dir = os.path.join(result_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        # Generate heatmap
        plot_heatmap(
            stats,
            os.path.join(dataset_dir, f"{dataset}_misclassification_heatmap.png"),
            dataset,
        )

        # Generate LaTeX table
        save_latex_table(
            stats,
            os.path.join(dataset_dir, f"{dataset}_misclassification_table.tex"),
            dataset,
            n_samples,
        )

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
