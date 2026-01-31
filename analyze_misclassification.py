"""
Misclassification speed analysis for PGD attacks.

Analyzes how many iterations are needed to achieve first misclassification,
which directly measures attack success speed.

Usage:
    python analyze_misclassification.py --input_dir outputs/arrays/run_all/ --out_dir outputs
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

matplotlib.use("Agg")

MODEL_ORDER = ["nat", "nat_and_adv", "adv", "weak_adv"]
INIT_ORDER = ["random", "deepfool", "multi_deepfool"]

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
    """Parse filename to extract dataset, model, init, panel_index.

    Expected patterns:
    - mnist_nat_random_idx0_k100_eps0.3_a0.01_r20_seed0_p1_corrects.npy
    - cifar10_adv_deepfool_idx1_k100_eps*_*_r1_seed0_dfiter50_dfo0.02_dfj0.0_dfproject_clip_p1_corrects.npy
    """
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
        corrects: shape (restarts, iterations+1), bool array where True means correct

    Returns:
        first_wrong: shape (restarts,)
            >= 0: iteration index of first misclassification
            NOT_MISCLASSIFIED (-1): never misclassified (attack failed)
    """
    wrong = ~corrects
    any_wrong = np.any(wrong, axis=1)
    first_wrong_iter = np.argmax(wrong, axis=1).astype(np.int32)
    first_wrong_iter[~any_wrong] = NOT_MISCLASSIFIED

    return first_wrong_iter


def aggregate_misclassification_stats(
    data_list: List[CorrectsData],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate misclassification statistics by model and init.

    Returns:
        stats[model][init] = array of first misclassification iterations
    """
    stats: Dict[str, Dict[str, List[int]]] = {}

    for data in data_list:
        if data.model not in stats:
            stats[data.model] = {}
        if data.init not in stats[data.model]:
            stats[data.model][data.init] = []

        first_wrong = compute_first_misclassification(data.corrects)
        stats[data.model][data.init].extend(first_wrong.tolist())

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for model in stats:
        result[model] = {}
        for init in stats[model]:
            result[model][init] = np.array(stats[model][init])

    return result


def print_misclassification_summary(
    stats: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Print summary table of misclassification statistics."""
    print(f"\n{'=' * 90}")
    print("Misclassification Speed Summary (first misclassification iteration)")
    print(f"{'=' * 90}")

    all_iters = []

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        print(f"\n[{model}]")
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue
            iters = stats[model][init]
            all_iters.extend(iters.tolist())

            n_total = len(iters)
            misclassified = iters[iters >= 0]
            n_misclassified = len(misclassified)
            n_not = int(np.sum(iters == NOT_MISCLASSIFIED))
            attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0

            if n_misclassified > 0:
                print(
                    f"  {init:20s}: "
                    f"attack={attack_rate:5.1f}%, "
                    f"mean={np.mean(misclassified):5.1f}, "
                    f"median={np.median(misclassified):5.1f}, "
                    f"p95={np.percentile(misclassified, 95):5.1f}, "
                    f"max={np.max(misclassified):5.0f}, "
                    f"n={n_total}, "
                    f"failed={n_not}"
                )
            else:
                print(
                    f"  {init:20s}: "
                    f"attack={attack_rate:5.1f}%, "
                    f"n={n_total}, "
                    f"failed={n_not}"
                )

    if all_iters:
        all_iters = np.array(all_iters)
        misclassified = all_iters[all_iters >= 0]
        n_total = len(all_iters)
        n_misclassified = len(misclassified)
        n_not = int(np.sum(all_iters == NOT_MISCLASSIFIED))
        attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0

        print(f"\n{'=' * 90}")
        print("[ALL MODELS COMBINED]")
        if n_misclassified > 0:
            print(
                f"  attack={attack_rate:5.1f}%, "
                f"mean={np.mean(misclassified):5.1f}, "
                f"median={np.median(misclassified):5.1f}, "
                f"p95={np.percentile(misclassified, 95):5.1f}, "
                f"max={np.max(misclassified):5.0f}, "
                f"n={n_total}, "
                f"failed={n_not}"
            )
        else:
            print(
                f"  attack={attack_rate:5.1f}%, "
                f"n={n_total}, "
                f"failed={n_not}"
            )
        print(f"{'=' * 90}")


def generate_markdown_summary(
    stats: Dict[str, Dict[str, np.ndarray]], dataset: str
) -> str:
    """Generate markdown formatted summary."""
    lines = []
    lines.append(f"# Misclassification Speed Analysis Summary ({dataset.upper()})")
    lines.append("")
    lines.append(f"**Dataset**: {dataset.upper()}")
    lines.append("")
    lines.append("**Metric**: First misclassification iteration (attack success speed)")
    lines.append("")
    lines.append("**Data points per init** (single sample):")
    lines.append("- clean: 1 (deterministic, single point)")
    lines.append("- random: 20 (stochastic)")
    lines.append("- deepfool: 1 (deterministic, single point)")
    lines.append("- multi_deepfool: 9 (deterministic, 9 target classes)")
    lines.append("")

    lines.append("## Detailed Results (by model and init)")
    lines.append("")
    lines.append(
        "| Model | Init | Attack Rate | Mean | Median | P95 | Max | N | Failed |"
    )
    lines.append(
        "|-------|------|------------:|-----:|-------:|----:|----:|--:|-------:|"
    )

    all_iters = []
    for model in MODEL_ORDER:
        if model not in stats:
            continue
        for init in INIT_ORDER:
            if init not in stats[model]:
                continue
            iters = stats[model][init]
            all_iters.extend(iters.tolist())

            n_total = len(iters)
            misclassified = iters[iters >= 0]
            n_misclassified = len(misclassified)
            n_not = int(np.sum(iters == NOT_MISCLASSIFIED))
            attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0
            failed_rate = n_not / n_total * 100 if n_total > 0 else 0

            if n_misclassified > 0:
                lines.append(
                    f"| {model} | {init} | "
                    f"{attack_rate:.0f}%({n_misclassified}/{n_total}) | "
                    f"{np.mean(misclassified):.1f} | "
                    f"{np.median(misclassified):.1f} | "
                    f"{np.percentile(misclassified, 95):.1f} | "
                    f"{np.max(misclassified):.0f} | "
                    f"{n_total} | "
                    f"{failed_rate:.0f}%({n_not}) |"
                )
            else:
                lines.append(
                    f"| {model} | {init} | "
                    f"{attack_rate:.0f}%({n_misclassified}/{n_total}) | "
                    f"N/A | N/A | N/A | N/A | "
                    f"{n_total} | "
                    f"{failed_rate:.0f}%({n_not}) |"
                )

    lines.append("")

    lines.append("## Model-level Summary")
    lines.append("")
    lines.append("| Model | Attack Rate | Mean | Median | P95 | Max | N | Failed |")
    lines.append("|-------|------------:|-----:|-------:|----:|----:|--:|-------:|")

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        model_iters = np.array(model_iters)

        n_total = len(model_iters)
        misclassified = model_iters[model_iters >= 0]
        n_misclassified = len(misclassified)
        n_not = int(np.sum(model_iters == NOT_MISCLASSIFIED))
        attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0
        failed_rate = n_not / n_total * 100 if n_total > 0 else 0

        if n_misclassified > 0:
            lines.append(
                f"| {model} | "
                f"{attack_rate:.0f}%({n_misclassified}/{n_total}) | "
                f"{np.mean(misclassified):.1f} | "
                f"{np.median(misclassified):.1f} | "
                f"{np.percentile(misclassified, 95):.1f} | "
                f"{np.max(misclassified):.0f} | "
                f"{n_total} | "
                f"{failed_rate:.0f}%({n_not}) |"
            )
        else:
            lines.append(
                f"| {model} | "
                f"{attack_rate:.0f}%({n_misclassified}/{n_total}) | "
                f"N/A | N/A | N/A | N/A | "
                f"{n_total} | "
                f"{failed_rate:.0f}%({n_not}) |"
            )

    lines.append("")

    if all_iters:
        all_iters = np.array(all_iters)
        misclassified = all_iters[all_iters >= 0]
        n_total = len(all_iters)
        n_misclassified = len(misclassified)
        n_not = int(np.sum(all_iters == NOT_MISCLASSIFIED))
        attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0
        failed_rate = n_not / n_total * 100 if n_total > 0 else 0

        lines.append("## Overall Summary (All Models Combined)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Attack Success Rate | {attack_rate:.0f}%({n_misclassified}/{n_total}) |")
        lines.append(f"| Total samples | {n_total} |")
        lines.append(f"| Misclassified | {n_misclassified} |")
        lines.append(f"| Failed (never misclassified) | {failed_rate:.0f}%({n_not}) |")

        if n_misclassified > 0:
            lines.append(f"| Mean iteration | {np.mean(misclassified):.1f} |")
            lines.append(f"| Median iteration | {np.median(misclassified):.1f} |")
            lines.append(f"| P95 iteration | {np.percentile(misclassified, 95):.1f} |")
            lines.append(f"| Max iteration | {np.max(misclassified):.0f} |")
        lines.append("")

        if n_misclassified > 0:
            p50 = np.percentile(misclassified, 50)
            p90 = np.percentile(misclassified, 90)
            p95 = np.percentile(misclassified, 95)
            lines.append("## Key Findings")
            lines.append("")
            lines.append(
                f"- **50% of successful attacks** achieve misclassification by iteration **{p50:.0f}**"
            )
            lines.append(
                f"- **90% of successful attacks** achieve misclassification by iteration **{p90:.0f}**"
            )
            lines.append(
                f"- **95% of successful attacks** achieve misclassification by iteration **{p95:.0f}**"
            )
            lines.append("")

    return "\n".join(lines)


def save_markdown_summary(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    dataset: str,
) -> None:
    """Save markdown summary to file."""
    md_content = generate_markdown_summary(stats, dataset)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SAVE] Markdown summary: {out_path}")


def plot_misclassification_cdf(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    dataset: str,
    init: str,
) -> None:
    """Plot CDF of misclassification iterations (what % misclassified by iteration N)."""
    all_iters_by_model = {}
    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init_key in stats[model]:
            model_iters.extend(stats[model][init_key].tolist())
        if model_iters:
            all_iters_by_model[model] = np.array(model_iters)

    if not all_iters_by_model:
        return

    models = [m for m in MODEL_ORDER if m in all_iters_by_model]
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, model in enumerate(models):
        iters = all_iters_by_model[model]
        misclassified = iters[iters >= 0]
        n_total = len(iters)
        n_not = int(np.sum(iters == NOT_MISCLASSIFIED))

        attack_rate = len(misclassified) / n_total * 100
        label = f"{model} (n={n_total}, {attack_rate:.0f}%, failed:{n_not})"

        if len(misclassified) == 0:
            ax.plot([], [], color=colors[idx], linewidth=2, label=label)
            continue

        sorted_iters = np.sort(misclassified)
        cdf = np.arange(1, len(sorted_iters) + 1) / n_total

        marker = "o" if n_total <= 5 else None
        markersize = 8 if n_total <= 5 else None
        ax.step(
            sorted_iters,
            cdf,
            where="post",
            label=label,
            color=colors[idx],
            linewidth=2,
            marker=marker,
            markersize=markersize,
        )

    all_iters_flat = np.concatenate(list(all_iters_by_model.values()))
    all_misclassified = all_iters_flat[all_iters_flat >= 0]
    n_total_all = len(all_iters_flat)
    n_not_all = int(np.sum(all_iters_flat == NOT_MISCLASSIFIED))

    if len(all_misclassified) > 0:
        all_sorted = np.sort(all_misclassified)
        cdf_combined = np.arange(1, len(all_sorted) + 1) / n_total_all
        attack_rate_all = len(all_misclassified) / n_total_all * 100
        label_all = f"ALL (n={n_total_all}, {attack_rate_all:.0f}%, failed:{n_not_all})"
        ax.step(
            all_sorted,
            cdf_combined,
            where="post",
            label=label_all,
            color="black",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction misclassified")
    ax.set_title(
        f"Misclassification CDF ({dataset.upper()}, init={init}, single sample)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Misclassification CDF: {out_path}")


def plot_misclassification_histogram(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    dataset: str,
    init: str,
) -> None:
    """Plot histogram of misclassification iterations with side-by-side bars."""
    all_iters_by_model = {}
    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init_key in stats[model]:
            model_iters.extend(stats[model][init_key].tolist())
        if model_iters:
            all_iters_by_model[model] = np.array(model_iters)

    if not all_iters_by_model:
        print("[WARN] No data for histogram")
        return

    models = [m for m in MODEL_ORDER if m in all_iters_by_model]
    n_models = len(models)
    fig, ax = plt.subplots(figsize=(14, 6))

    all_misclassified = []
    for m in models:
        misclassified = all_iters_by_model[m][all_iters_by_model[m] >= 0]
        all_misclassified.extend(misclassified.tolist())

    if all_misclassified:
        max_iter = int(np.max(all_misclassified))
    else:
        max_iter = 100

    bin_edges = list(range(max_iter + 2))
    n_bins = len(bin_edges) - 1
    bar_width = 0.8 / n_models

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for idx, model in enumerate(models):
        iters = all_iters_by_model[model]
        misclassified_iters = iters[iters >= 0]
        n_not_count = int(np.sum(iters == NOT_MISCLASSIFIED))

        counts, _ = np.histogram(misclassified_iters, bins=bin_edges)

        x_positions = np.arange(n_bins) + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(
            x_positions,
            counts,
            width=bar_width,
            color=colors[idx],
            alpha=0.8,
            label=f"{model} (n={len(misclassified_iters)}/{len(iters)})",
        )

        fail_x = n_bins + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(
            fail_x, n_not_count, width=bar_width, color=colors[idx],
            alpha=0.8, hatch="//",
        )

    step = 10
    x_tick_positions = list(range(0, n_bins, step)) + [n_bins]
    x_tick_labels = [str(i) for i in range(0, n_bins, step)] + ["FAIL"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("Iteration (FAIL=never misclassified)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"First Misclassification Distribution "
        f"({dataset.upper()}, init={init}, single sample)"
    )
    ax.legend()
    ax.set_xlim(-0.5, n_bins + 1.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Misclassification histogram: {out_path}")


def plot_single_model_histogram(
    iters: np.ndarray,
    model: str,
    out_path: str,
    dataset: str,
    init: str,
) -> None:
    """Plot histogram of misclassification iterations for a single model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    misclassified_iters = iters[iters >= 0]
    n_not_count = int(np.sum(iters == NOT_MISCLASSIFIED))
    n_total = len(iters)
    n_misclassified = len(misclassified_iters)

    if n_misclassified > 0:
        max_iter = int(np.max(misclassified_iters))
    else:
        max_iter = 100

    bin_edges = list(range(max_iter + 2))
    n_bins = len(bin_edges) - 1

    counts, _ = np.histogram(misclassified_iters, bins=bin_edges)
    ax.bar(range(n_bins), counts, width=0.8, color="steelblue", alpha=0.8)

    ax.bar(n_bins, n_not_count, width=0.8, color="salmon", alpha=0.8, label="FAIL")

    step = 10
    x_tick_positions = list(range(0, n_bins, step)) + [n_bins]
    x_tick_labels = [str(i) for i in range(0, n_bins, step)] + ["FAIL"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("Iteration (FAIL=never misclassified)")
    ax.set_ylabel("Count")
    attack_rate = n_misclassified / n_total * 100 if n_total > 0 else 0
    ax.set_title(
        f"Misclassification Distribution "
        f"({dataset.upper()}, init={init}, model={model}, "
        f"n={n_total}, attack={attack_rate:.1f}%)"
    )
    ax.set_xlim(-0.5, n_bins + 1.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if n_not_count > 0:
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Single model histogram: {out_path}")


def plot_misclassification_cdf_overlay(
    stats_by_init: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
    dataset: str,
) -> None:
    """Plot CDF overlay comparing all inits on one figure."""
    fig, ax = plt.subplots(figsize=(12, 7))

    linestyles = {"clean": "-", "random": "--", "deepfool": "-.", "multi_deepfool": ":"}
    colors_model = {"nat": "C0", "nat_and_adv": "C1", "weak_adv": "C2", "adv": "C3"}

    for init in INIT_ORDER:
        if init not in stats_by_init:
            continue
        stats = stats_by_init[init]

        for model in MODEL_ORDER:
            if model not in stats:
                continue
            model_iters = []
            for init_key in stats[model]:
                model_iters.extend(stats[model][init_key].tolist())
            if not model_iters:
                continue

            iters = np.array(model_iters)
            misclassified = iters[iters >= 0]
            n_total = len(iters)

            if len(misclassified) == 0:
                continue

            sorted_iters = np.sort(misclassified)
            cdf = np.arange(1, len(sorted_iters) + 1) / n_total

            attack_rate = len(misclassified) / n_total * 100
            label = f"{model}/{init} ({attack_rate:.0f}%)"

            ax.step(
                sorted_iters,
                cdf,
                where="post",
                label=label,
                color=colors_model.get(model, "gray"),
                linestyle=linestyles.get(init, "-"),
                linewidth=1.5,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction misclassified")
    ax.set_title(f"Misclassification CDF Overlay ({dataset.upper()}, all inits)")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Misclassification CDF overlay: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze misclassification speed from corrects arrays"
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

    datasets = sorted(set(d.dataset for d in data_list))
    inits = sorted(set(d.init for d in data_list))

    for dataset in datasets:
        for init in inits:
            subdir = os.path.join(result_dir, dataset, init)
            os.makedirs(subdir, exist_ok=True)

    for dataset in datasets:
        dataset_data = [d for d in data_list if d.dataset == dataset]
        if not dataset_data:
            continue

        dataset_stats = aggregate_misclassification_stats(dataset_data)

        print(f"\n{'#' * 40}")
        print(f"# Dataset: {dataset.upper()}")
        print(f"{'#' * 40}")
        print_misclassification_summary(dataset_stats)

        dataset_dir = os.path.join(result_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        save_markdown_summary(
            dataset_stats,
            os.path.join(dataset_dir, "misclassification_report.md"),
            dataset,
        )

    stats_by_dataset_init: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}

    for dataset in datasets:
        stats_by_dataset_init[dataset] = {}
        for init in inits:
            subdir = os.path.join(result_dir, dataset, init)

            subset_data = [
                d for d in data_list if d.dataset == dataset and d.init == init
            ]
            if not subset_data:
                continue

            subset_stats = aggregate_misclassification_stats(subset_data)
            stats_by_dataset_init[dataset][init] = subset_stats

            plot_misclassification_cdf(
                subset_stats,
                os.path.join(subdir, "misclassification_cdf.png"),
                dataset,
                init,
            )

            plot_misclassification_histogram(
                subset_stats,
                os.path.join(subdir, "misclassification_histogram.png"),
                dataset,
                init,
            )

            for model in MODEL_ORDER:
                if model not in subset_stats:
                    continue
                model_iters = []
                for init_key in subset_stats[model]:
                    model_iters.extend(subset_stats[model][init_key].tolist())
                if model_iters:
                    plot_single_model_histogram(
                        np.array(model_iters),
                        model,
                        os.path.join(subdir, f"misclassification_histogram_{model}.png"),
                        dataset,
                        init,
                    )

    for dataset in datasets:
        if dataset in stats_by_dataset_init and stats_by_dataset_init[dataset]:
            dataset_dir = os.path.join(result_dir, dataset)
            plot_misclassification_cdf_overlay(
                stats_by_dataset_init[dataset],
                os.path.join(dataset_dir, "misclassification_cdf_overlay.png"),
                dataset,
            )

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
