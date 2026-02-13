"""Misclassification analysis for PGD attacks with multiple samples.

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
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.dataset_config import resolve_dataset_config

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

NOT_MISCLASSIFIED = -1


def infer_experiment_name(input_dir: str) -> str:
    """Infer experiment name from input directory path.

    Extracts the last component of the normalized path as the experiment name.

    Args:
        input_dir: Path to the input directory
            (e.g., "outputs/arrays/run_all_ex10/" -> "run_all_ex10")

    Returns:
        Experiment name string
    """
    return os.path.basename(os.path.normpath(input_dir))


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
    """Load all *_corrects.npy files from input directory.

    Args:
        input_dir: Path to directory containing *_corrects.npy files

    Returns:
        List of CorrectsData objects

    Raises:
        FileNotFoundError: if input_dir does not exist
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            f"Please verify the path and try again."
        )

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


def compute_combined_stats(
    data_list: List[CorrectsData], max_iter: int = 100
) -> AveragedStats:
    """Compute statistics by combining all samples and restarts.

    All (sample, restart) combinations are pooled together to compute
    mean, median, and p95 directly from the combined data.
    """
    if not data_list:
        return AveragedStats(
            attack_success_rate=0.0,
            mean=None,
            median=None,
            p95=None,
            misclassification_rates=np.zeros(max_iter + 1),
            n_samples=0,
        )

    n_samples = len(data_list)

    # Collect all first_wrong values from all samples and restarts
    all_first_wrong = []
    for data in data_list:
        first_wrong = compute_first_misclassification(data.corrects)
        all_first_wrong.extend(first_wrong.tolist())

    all_first_wrong = np.array(all_first_wrong)
    n_total = len(all_first_wrong)
    misclassified = all_first_wrong[all_first_wrong >= 0]
    n_misclassified = len(misclassified)

    # Compute attack success rate over all (sample, restart) combinations
    attack_success_rate = n_misclassified / n_total if n_total > 0 else 0.0

    # Compute statistics from pooled data
    if n_misclassified > 0:
        combined_mean = float(np.mean(misclassified))
        combined_median = float(np.median(misclassified))
        combined_p95 = float(np.percentile(misclassified, 95))
    else:
        combined_mean = None
        combined_median = None
        combined_p95 = None

    # Compute misclassification rate at each iteration (over all combinations)
    rates = np.zeros(max_iter + 1)
    for t in range(max_iter + 1):
        n_wrong_at_t = np.sum((all_first_wrong >= 0) & (all_first_wrong <= t))
        rates[t] = n_wrong_at_t / n_total if n_total > 0 else 0.0

    return AveragedStats(
        attack_success_rate=attack_success_rate,
        mean=combined_mean,
        median=combined_median,
        p95=combined_p95,
        misclassification_rates=rates,
        n_samples=n_samples,
    )


def aggregate_corrects_by_group(
    data_list: List[CorrectsData],
) -> Dict[str, Dict[str, Dict[str, List[CorrectsData]]]]:
    """Group CorrectsData by dataset/model/init.

    Returns:
        result[dataset][model][init] = list of CorrectsData
    """
    result: Dict[str, Dict[str, Dict[str, List[CorrectsData]]]] = {}

    for data in data_list:
        if data.dataset not in result:
            result[data.dataset] = {}
        if data.model not in result[data.dataset]:
            result[data.dataset][data.model] = {}
        if data.init not in result[data.dataset][data.model]:
            result[data.dataset][data.model][data.init] = []

        result[data.dataset][data.model][data.init].append(data)

    return result


def compute_combined_stats_for_all(
    grouped_data: Dict[str, Dict[str, Dict[str, List[CorrectsData]]]],
    max_iter: int = 100,
) -> Dict[str, Dict[str, Dict[str, AveragedStats]]]:
    """Compute combined statistics for each dataset/model/init."""
    result: Dict[str, Dict[str, Dict[str, AveragedStats]]] = {}

    for dataset in grouped_data:
        result[dataset] = {}
        for model in grouped_data[dataset]:
            result[dataset][model] = {}
            for init in grouped_data[dataset][model]:
                data_list = grouped_data[dataset][model][init]
                result[dataset][model][init] = compute_combined_stats(data_list, max_iter)

    return result


def _collect_heatmap_rows(
    stats: Dict[str, Dict[str, AveragedStats]],
) -> Tuple[List[str], List[str], List[np.ndarray]]:
    """Collect model names, init names, and rate arrays for heatmap rows."""
    row_models: List[str] = []
    row_inits: List[str] = []
    row_data: List[np.ndarray] = []

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue
            row_models.append(model)
            row_inits.append(init)
            row_data.append(stats[model][init].misclassification_rates)

    return row_models, row_inits, row_data


def _draw_model_labels(
    ax_model: plt.Axes,
    row_models: List[str],
    n_rows: int,
) -> List[float]:
    """Draw model labels centered per group. Returns separator y positions."""
    separators: List[float] = []
    prev_model = None
    group_start = 0

    for i, model in enumerate(row_models):
        if model != prev_model:
            if prev_model is not None:
                mid_y = (group_start + i - 1) / 2.0
                ax_model.text(
                    0.5, mid_y, prev_model,
                    ha="center", va="center", fontsize=20, fontweight="bold",
                )
                separators.append(i - 0.5)
            prev_model = model
            group_start = i

    if prev_model is not None:
        mid_y = (group_start + n_rows - 1) / 2.0
        ax_model.text(
            0.5, mid_y, prev_model,
            ha="center", va="center", fontsize=20, fontweight="bold",
        )

    return separators


def _draw_init_labels(ax_init: plt.Axes, row_inits: List[str]) -> None:
    """Draw init labels for each row."""
    for i, init_name in enumerate(row_inits):
        ax_init.text(
            0.5, i, init_name,
            ha="center", va="center", fontsize=20,
        )


def plot_heatmap(
    stats: Dict[str, Dict[str, AveragedStats]],
    out_path: str,
    dataset: str,
    max_iter: int = 100,
    n_samples: int = 10,
) -> None:
    """Plot heatmap of misclassification rate (averaged over samples).

    X-axis: PGD iterations (0 to max_iter)
    Y-axis: model and init shown as two separate label columns
    Color: fraction of trials misclassified at that iteration
    """
    row_models, row_inits, row_data = _collect_heatmap_rows(stats)

    if not row_data:
        print(f"[WARN] No data for heatmap: {dataset}")
        return

    heatmap_matrix = np.array(row_data)
    n_rows = len(row_models)

    fig_h = max(n_rows * 0.5 + 2, 4)
    fig = plt.figure(figsize=(14, fig_h))

    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[0.30, 0.28, 1.0, 0.04],
        wspace=0.05,
    )

    ax_model = fig.add_subplot(gs[0, 0])
    ax_init = fig.add_subplot(gs[0, 1])
    ax_hmap = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3])

    # Heatmap
    im = ax_hmap.imshow(
        heatmap_matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_hmap.set_yticks([])

    x_ticks = list(range(0, max_iter + 1, 10))
    ax_hmap.set_xticks(x_ticks)
    ax_hmap.set_xticklabels([str(x) for x in x_ticks], fontsize=18)
    ax_hmap.set_xlabel("PGD Iteration", fontsize=22)
    ax_hmap.set_title(
        f"Misclassification Rate Heatmap ({dataset.upper()})",
        fontsize=24, pad=12,
    )

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Misclassification Rate", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # Configure label axes to match heatmap y-axis
    for ax_label in [ax_model, ax_init]:
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(n_rows - 0.5, -0.5)
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        for spine in ax_label.spines.values():
            spine.set_visible(False)

    # Model labels (centered per group) and separators
    separators = _draw_model_labels(ax_model, row_models, n_rows)
    # Group separators + top/bottom borders
    all_lines = [-0.5] + separators + [n_rows - 0.5]
    for y in all_lines:
        for ax in [ax_model, ax_init, ax_hmap]:
            ax.axhline(y=y, color="black", linewidth=1.5)

    # Init labels (one per row)
    _draw_init_labels(ax_init, row_inits)

    # Column headers
    ax_model.set_title("Model", fontsize=22, fontweight="bold", pad=10)
    ax_init.set_title("Init", fontsize=22, fontweight="bold", pad=10)

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
    lines.append("  \\begin{tabular}{l|l|cccc}")
    lines.append("    \\hline")
    lines.append("    & & & \\multicolumn{3}{c}{誤分類達成反復数} \\\\")
    lines.append("    \\cline{4-6}")
    lines.append("    モデル & 初期化 & 誤分類達成率 & 平均 & 中央値 & P95 \\\\")
    lines.append("    \\hline")

    for model in MODEL_ORDER:
        if model not in stats:
            continue

        first_model = True
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue

            avg_stats = stats[model][init]

            model_col = f"\\multirow{{4}}{{*}}{{{MODEL_DISPLAY[model]}}}" if first_model else ""
            first_model = False

            attack_rate_pct = avg_stats.attack_success_rate * 100

            # Show "---" if attack success rate < 1% or mean is None
            if avg_stats.mean is not None and attack_rate_pct >= 1.0:
                lines.append(
                    f"    {model_col} & {INIT_DISPLAY[init]} & "
                    f"{attack_rate_pct:.0f}\\% & {avg_stats.mean:.1f} & {avg_stats.median:.1f} & {avg_stats.p95:.1f} \\\\"
                )
            else:
                lines.append(
                    f"    {model_col} & {INIT_DISPLAY[init]} & "
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
    print(f"Dataset: {dataset.upper()} ({n_samples} samples)")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Init':<15} {'Success%':<10} {'Mean':<8} {'Median':<8} {'P95':<8}")
    print("-" * 70)

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue

            avg_stats = stats[model][init]
            attack_rate_pct = avg_stats.attack_success_rate * 100

            if avg_stats.mean is not None:
                print(
                    f"{model:<15} {init:<15} {attack_rate_pct:<10.0f} "
                    f"{avg_stats.mean:<8.1f} {avg_stats.median:<8.1f} {avg_stats.p95:<8.1f}"
                )
            else:
                print(
                    f"{model:<15} {init:<15} {attack_rate_pct:<10.0f} "
                    f"{'---':<8} {'---':<8} {'---':<8}"
                )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for misclassification analysis.

    Returns:
        Configured ArgumentParser with --input_dir, --out_dir, --exp_name,
        and --max_iter arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze misclassification statistics from PGD attack results. "
            "Reads *_corrects.npy files from the input directory, computes "
            "misclassification rates, and generates heatmaps and LaTeX tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python analyze_misclassification.py \\\n"
            "      --input_dir outputs/arrays/run_all_ex10/ --out_dir outputs\n"
            "  python analyze_misclassification.py \\\n"
            "      --input_dir outputs/arrays/run_all_ex100/ \\\n"
            "      --exp_name custom_experiment --max_iter 200"
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_corrects.npy files from PGD experiments",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/work/outputs",
        help="Base output directory for results (default: /work/outputs)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help=(
            "Experiment name for output subdirectory. "
            "If not specified, inferred from the input directory name."
        ),
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum PGD iteration count (default: 100)",
    )
    return parser


def main() -> None:
    """Entry point for misclassification analysis."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Infer experiment name from input directory if not specified
    exp_name = args.exp_name if args.exp_name else infer_experiment_name(args.input_dir)
    max_iter = args.max_iter

    result_dir = os.path.join(args.out_dir, "misclassification_analysis", exp_name)
    os.makedirs(result_dir, exist_ok=True)

    # Load corrects files (raises FileNotFoundError if directory missing)
    try:
        data_list = load_corrects_files(args.input_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if not data_list:
        print(
            f"[ERROR] No *_corrects.npy files found in: {args.input_dir}\n"
            f"Ensure the directory contains PGD experiment output files "
            f"matching the pattern: {{dataset}}_{{model}}_{{init}}_p{{N}}_corrects.npy",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate datasets found against known configurations
    detected_datasets = set(d.dataset for d in data_list)
    for ds in detected_datasets:
        try:
            resolve_dataset_config(ds)
        except ValueError:
            print(f"[WARN] Unknown dataset '{ds}' detected; skipping config validation")

    # Group data by dataset/model/init
    grouped_data = aggregate_corrects_by_group(data_list)

    # Compute combined statistics (all samples and restarts pooled together)
    averaged_stats = compute_combined_stats_for_all(grouped_data, max_iter=max_iter)

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
            max_iter=max_iter,
            n_samples=n_samples,
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
