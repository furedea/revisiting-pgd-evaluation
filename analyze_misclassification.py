"""
Misclassification analysis for PGD attacks with multiple samples.

Generates:
1. Heatmap: X-axis = PGD iterations, Y-axis = model/init combinations,
   Color = fraction of trials misclassified at that iteration (averaged over samples)
2. Table: LaTeX table with mean statistics averaged over samples

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


class SampleStats:
    """Statistics for a single sample."""

    __slots__ = ("attack_success_rate", "mean", "median", "p95", "misclassification_rates")

    def __init__(
        self,
        attack_success_rate,  # type: float
        mean,  # type: Optional[float]
        median,  # type: Optional[float]
        p95,  # type: Optional[float]
        misclassification_rates,  # type: np.ndarray
    ):
        self.attack_success_rate = attack_success_rate
        self.mean = mean
        self.median = median
        self.p95 = p95
        self.misclassification_rates = misclassification_rates  # shape: (max_iter+1,)


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


def compute_sample_stats(corrects: np.ndarray, max_iter: int = 100) -> SampleStats:
    """Compute statistics for a single sample.

    Args:
        corrects: shape (restarts, iterations+1)
        max_iter: maximum iteration number

    Returns:
        SampleStats with attack_success_rate, mean, median, p95, and misclassification_rates
    """
    first_wrong = compute_first_misclassification(corrects)
    n_total = len(first_wrong)
    misclassified = first_wrong[first_wrong >= 0]
    n_misclassified = len(misclassified)

    attack_success_rate = n_misclassified / n_total if n_total > 0 else 0.0

    if n_misclassified > 0:
        mean = float(np.mean(misclassified))
        median = float(np.median(misclassified))
        p95 = float(np.percentile(misclassified, 95))
    else:
        mean = None
        median = None
        p95 = None

    # Compute misclassification rate at each iteration
    rates = np.zeros(max_iter + 1)
    for t in range(max_iter + 1):
        n_wrong_at_t = np.sum((first_wrong >= 0) & (first_wrong <= t))
        rates[t] = n_wrong_at_t / n_total if n_total > 0 else 0.0

    return SampleStats(
        attack_success_rate=attack_success_rate,
        mean=mean,
        median=median,
        p95=p95,
        misclassification_rates=rates,
    )


def aggregate_by_sample(
    data_list: List[CorrectsData],
    max_iter: int = 100,
) -> Dict[str, Dict[str, Dict[str, List[SampleStats]]]]:
    """Aggregate statistics by sample for each dataset/model/init.

    Returns:
        result[dataset][model][init] = list of SampleStats (one per sample)
    """
    result: Dict[str, Dict[str, Dict[str, List[SampleStats]]]] = {}

    for data in data_list:
        if data.dataset not in result:
            result[data.dataset] = {}
        if data.model not in result[data.dataset]:
            result[data.dataset][data.model] = {}
        if data.init not in result[data.dataset][data.model]:
            result[data.dataset][data.model][data.init] = []

        sample_stats = compute_sample_stats(data.corrects, max_iter)
        result[data.dataset][data.model][data.init].append(sample_stats)

    return result


class AveragedStats:
    """Statistics averaged over multiple samples."""

    __slots__ = ("attack_success_rate", "mean", "median", "p95", "misclassification_rates", "n_samples")

    def __init__(
        self,
        attack_success_rate,  # type: float
        mean,  # type: Optional[float]
        median,  # type: Optional[float]
        p95,  # type: Optional[float]
        misclassification_rates,  # type: np.ndarray
        n_samples,  # type: int
    ):
        self.attack_success_rate = attack_success_rate
        self.mean = mean
        self.median = median
        self.p95 = p95
        self.misclassification_rates = misclassification_rates  # shape: (max_iter+1,)
        self.n_samples = n_samples


def average_sample_stats(sample_stats_list: List[SampleStats]) -> AveragedStats:
    """Average statistics over multiple samples.

    For numeric stats (mean, median, p95), average only over samples where attack succeeded.
    """
    n_samples = len(sample_stats_list)
    if n_samples == 0:
        return AveragedStats(
            attack_success_rate=0.0,
            mean=None,
            median=None,
            p95=None,
            misclassification_rates=np.zeros(101),
            n_samples=0,
        )

    # Average attack success rate
    attack_success_rate = np.mean([s.attack_success_rate for s in sample_stats_list])

    # Average numeric stats (only from samples with successful attacks)
    means = [s.mean for s in sample_stats_list if s.mean is not None]
    medians = [s.median for s in sample_stats_list if s.median is not None]
    p95s = [s.p95 for s in sample_stats_list if s.p95 is not None]

    avg_mean = float(np.mean(means)) if means else None
    avg_median = float(np.mean(medians)) if medians else None
    avg_p95 = float(np.mean(p95s)) if p95s else None

    # Average misclassification rates
    all_rates = np.array([s.misclassification_rates for s in sample_stats_list])
    avg_rates = np.mean(all_rates, axis=0)

    return AveragedStats(
        attack_success_rate=attack_success_rate,
        mean=avg_mean,
        median=avg_median,
        p95=avg_p95,
        misclassification_rates=avg_rates,
        n_samples=n_samples,
    )


def compute_averaged_stats(
    sample_stats: Dict[str, Dict[str, Dict[str, List[SampleStats]]]],
) -> Dict[str, Dict[str, Dict[str, AveragedStats]]]:
    """Compute averaged statistics for each dataset/model/init."""
    result: Dict[str, Dict[str, Dict[str, AveragedStats]]] = {}

    for dataset in sample_stats:
        result[dataset] = {}
        for model in sample_stats[dataset]:
            result[dataset][model] = {}
            for init in sample_stats[dataset][model]:
                stats_list = sample_stats[dataset][model][init]
                result[dataset][model][init] = average_sample_stats(stats_list)

    return result


def plot_heatmap(
    stats: Dict[str, Dict[str, AveragedStats]],
    out_path: str,
    dataset: str,
    max_iter: int = 100,
) -> None:
    """Plot heatmap of misclassification rate (averaged over samples).

    X-axis: PGD iterations (0 to max_iter)
    Y-axis: model/init combinations
    Color: fraction of trials misclassified at that iteration
    """
    row_labels = []
    row_data = []

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue
            avg_stats = stats[model][init]
            row_labels.append(f"{model}/{init}")
            row_data.append(avg_stats.misclassification_rates)

    if not row_data:
        print(f"[WARN] No data for heatmap: {dataset}")
        return

    heatmap_matrix = np.array(row_data)

    fig, ax = plt.subplots(figsize=(14, len(row_labels) * 0.5 + 2))

    im = ax.imshow(
        heatmap_matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Misclassification Rate (10-sample avg)", fontsize=12)

    ax.set_xlabel("PGD Iteration", fontsize=12)
    ax.set_ylabel("Model / Init", fontsize=12)
    ax.set_title(f"Misclassification Heatmap ({dataset.upper()}, 10-sample average)", fontsize=14)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    x_ticks = list(range(0, max_iter + 1, 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Heatmap: {out_path}")


def generate_latex_table(
    stats: Dict[str, Dict[str, AveragedStats]],
    dataset: str,
    n_samples: int,
) -> str:
    """Generate LaTeX table with misclassification statistics (averaged over samples)."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(f"  \\caption{{{dataset.upper()}における誤分類統計（{n_samples}サンプル平均）}}")
    lines.append(f"  \\label{{table:{dataset}_misclassification_ex{n_samples}}}")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{l|l|c|cccc}")
    lines.append("    \\hline")
    lines.append("    & & & & \\multicolumn{3}{c}{誤分類達成反復数} \\\\")
    lines.append("    \\cline{5-7}")
    lines.append("    モデル & 初期化 & リスタート数 & 誤分類達成率 & 平均 & 中央値 & P95 \\\\")
    lines.append("    \\hline")

    for model in MODEL_ORDER:
        if model not in stats:
            continue

        first_model = True
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue

            avg_stats = stats[model][init]
            restart_count = RESTART_COUNTS.get(init, 1)

            model_col = f"\\multirow{{4}}{{*}}{{{MODEL_DISPLAY[model]}}}" if first_model else ""
            first_model = False

            attack_rate_pct = avg_stats.attack_success_rate * 100

            if avg_stats.mean is not None:
                lines.append(
                    f"    {model_col} & {INIT_DISPLAY[init]} & {restart_count} & "
                    f"{attack_rate_pct:.0f}\\% & {avg_stats.mean:.1f} & {avg_stats.median:.1f} & {avg_stats.p95:.1f} \\\\"
                )
            else:
                lines.append(
                    f"    {model_col} & {INIT_DISPLAY[init]} & {restart_count} & "
                    f"{attack_rate_pct:.0f}\\% & --- & --- & --- \\\\"
                )

        lines.append("    \\hline")

    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def save_latex_table(
    stats: Dict[str, Dict[str, AveragedStats]],
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
    stats: Dict[str, Dict[str, AveragedStats]],
    dataset: str,
    n_samples: int,
) -> None:
    """Print summary statistics to console."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset.upper()} ({n_samples} samples, averaged)")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Init':<15} {'Restarts':<10} {'Success%':<10} {'Mean':<8} {'Median':<8} {'P95':<8}")
    print("-" * 80)

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue

            avg_stats = stats[model][init]
            restart_count = RESTART_COUNTS.get(init, 1)
            attack_rate_pct = avg_stats.attack_success_rate * 100

            if avg_stats.mean is not None:
                print(
                    f"{model:<15} {init:<15} {restart_count:<10} {attack_rate_pct:<10.0f} "
                    f"{avg_stats.mean:<8.1f} {avg_stats.median:<8.1f} {avg_stats.p95:<8.1f}"
                )
            else:
                print(
                    f"{model:<15} {init:<15} {restart_count:<10} {attack_rate_pct:<10.0f} "
                    f"{'---':<8} {'---':<8} {'---':<8}"
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

    # Extract exp_name from input_dir (e.g., "outputs/arrays/run_all_ex10" -> "run_all_ex10")
    exp_name = os.path.basename(os.path.normpath(args.input_dir))

    result_dir = os.path.join(args.out_dir, "misclassification_analysis", exp_name)
    os.makedirs(result_dir, exist_ok=True)

    data_list = load_corrects_files(args.input_dir)
    if not data_list:
        print("[ERROR] No corrects files found")
        return

    # Aggregate by sample first
    sample_stats = aggregate_by_sample(data_list)

    # Compute averaged statistics
    averaged_stats = compute_averaged_stats(sample_stats)

    datasets = sorted(averaged_stats.keys())

    for dataset in datasets:
        stats = averaged_stats[dataset]

        # Count unique samples
        dataset_data = [d for d in data_list if d.dataset == dataset]
        n_samples = len(set(d.panel_index for d in dataset_data))

        # Print summary
        print_summary(stats, dataset, n_samples)

        # Create output directory
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
