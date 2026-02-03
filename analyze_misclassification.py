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
INIT_ORDER = ["clean", "random", "deepfool", "multi_deepfool"]

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

    colors = {"nat": "C0", "nat_and_adv": "C1", "adv": "C2", "weak_adv": "C3"}
    linewidths = {"nat": 3.0, "nat_and_adv": 2.5, "adv": 2.0, "weak_adv": 1.5}

    for model in models:
        iters = all_iters_by_model[model]
        misclassified = iters[iters >= 0]
        n_total = len(iters)
        n_not = int(np.sum(iters == NOT_MISCLASSIFIED))

        attack_rate = len(misclassified) / n_total * 100
        label = f"{model} (n={n_total}, {attack_rate:.0f}%, failed:{n_not})"

        if len(misclassified) == 0:
            ax.plot([], [], color=colors.get(model, "gray"), linewidth=2, label=label)
            continue

        sorted_iters = np.sort(misclassified)
        cdf = np.arange(1, len(sorted_iters) + 1) / n_total

        ax.step(
            sorted_iters,
            cdf,
            where="post",
            label=label,
            color=colors.get(model, "gray"),
            linewidth=linewidths.get(model, 2),
            alpha=0.85,
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

    colors = {"nat": "C0", "nat_and_adv": "C1", "adv": "C2", "weak_adv": "C3"}

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
            color=colors.get(model, "gray"),
            alpha=0.8,
            label=f"{model} (success={len(misclassified_iters)}, fail={n_not_count})",
        )

        fail_x = n_bins + 1 + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(
            fail_x, n_not_count, width=bar_width, color=colors.get(model, "gray"),
            alpha=0.8, hatch="//",
        )

    if max_iter <= 20:
        step = 5
    elif max_iter <= 50:
        step = 10
    else:
        step = 20
    x_tick_positions = list(range(0, n_bins + 1, step))
    if n_bins not in x_tick_positions:
        x_tick_positions.append(n_bins)
    x_tick_positions.append(n_bins + 1)
    x_tick_labels = [str(i) for i in x_tick_positions[:-1]] + ["FAIL"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("First Misclassification Iteration (FAIL = attack failed)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"First Misclassification Distribution ({dataset.upper()}, init={init}, single sample)"
    )
    ax.legend()
    ax.set_xlim(-0.5, n_bins + 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Misclassification histogram: {out_path}")


def get_color_for_init(init: str) -> str:
    """Get color for each init type (4 colors for init comparison).

    Uses highly distinguishable, saturated colors for clear visual separation.
    """
    colors = {
        "clean": "#0000CD",         # medium blue
        "random": "#228B22",        # forest green
        "deepfool": "#DC143C",      # crimson
        "multi_deepfool": "#FF8C00",  # dark orange
    }
    return colors.get(init, "gray")


def get_linestyle_for_init(init: str) -> str:
    """Get line style for each init type."""
    styles = {
        "clean": "-",           # solid
        "random": "--",         # dashed
        "deepfool": "-.",       # dash-dot
        "multi_deepfool": ":",  # dotted
    }
    return styles.get(init, "-")


def get_color_for_init_model(init: str, model: str) -> str:
    """Get distinct color for each init/model combination (legacy, for overlay).

    Uses highly distinguishable, saturated colors for clear visual separation.
    """
    colors = {
        # clean: blue系 (濃い色のみ)
        ("clean", "nat"): "#0000CD",          # medium blue
        ("clean", "nat_and_adv"): "#4169E1",  # royal blue
        ("clean", "adv"): "#1E90FF",          # dodger blue
        ("clean", "weak_adv"): "#00CED1",     # dark turquoise
        # random: green系 (濃い色のみ)
        ("random", "nat"): "#228B22",         # forest green
        ("random", "nat_and_adv"): "#32CD32", # lime green
        ("random", "adv"): "#008B8B",         # dark cyan
        ("random", "weak_adv"): "#2E8B57",    # sea green
        # deepfool: red/magenta系 (濃い色のみ)
        ("deepfool", "nat"): "#DC143C",       # crimson
        ("deepfool", "nat_and_adv"): "#FF1493", # deep pink
        ("deepfool", "adv"): "#C71585",       # medium violet red
        ("deepfool", "weak_adv"): "#DB7093",  # pale violet red
        # multi_deepfool: orange/brown系 (濃い色のみ)
        ("multi_deepfool", "nat"): "#FF4500",     # orange red
        ("multi_deepfool", "nat_and_adv"): "#FF8C00", # dark orange
        ("multi_deepfool", "adv"): "#B8860B",     # dark goldenrod
        ("multi_deepfool", "weak_adv"): "#DAA520", # goldenrod
    }
    return colors.get((init, model), "gray")


def get_linestyle_for_model(model: str, dataset: str = "mnist") -> str:
    """Get line style for each model to add extra distinction."""
    if dataset == "cifar10":
        # CIFAR10: more distinct linestyles for narrow data range
        styles = {
            "nat": "-",
            "nat_and_adv": (0, (10, 3)),  # long dashed (very distinct)
            "adv": "-.",
            "weak_adv": (0, (1, 1)),  # dotted (dense)
        }
    else:
        # MNIST: default styles
        styles = {
            "nat": "-",
            "nat_and_adv": "--",
            "adv": "-.",
            "weak_adv": (0, (3, 1)),
        }
    return styles.get(model, "-")


def plot_misclassification_cdf_overlay(
    stats_by_init: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
    dataset: str,
) -> None:
    """Plot CDF overlay comparing all inits on one figure."""
    fig, ax = plt.subplots(figsize=(16, 14))

    markers = {"nat": "o", "nat_and_adv": "s", "adv": "^", "weak_adv": "D"}
    # Line widths by model (nat thickest, adv thinnest)
    linewidths = {"nat": 3.0, "weak_adv": 2.5, "nat_and_adv": 2.0, "adv": 1.5}
    # Marker sizes by model
    marker_sizes = {"nat": 140, "weak_adv": 120, "nat_and_adv": 100, "adv": 80}
    # Z-order by model (nat on top)
    zorders = {"nat": 14, "weak_adv": 13, "nat_and_adv": 12, "adv": 11}
    handles_labels = []

    # CIFAR10: compute uniform x-offsets for all series (same offset at every x)
    if dataset == "cifar10":
        # Collect valid combinations, separating points (n=1) and lines (n>1)
        point_combinations = []  # clean, deepfool (n=1)
        line_combinations = []   # random, multi_deepfool (n>1)

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
                if len(misclassified) > 0:
                    if n_total == 1:
                        point_combinations.append((init, model))
                    else:
                        line_combinations.append((init, model))

        # Order: points first, then lines (so they group together)
        valid_combinations = point_combinations + line_combinations

        n_valid = len(valid_combinations)
        # Very tight offset (0.01 per item, centered)
        offset_width = 0.01
        if n_valid > 1:
            offsets = {
                combo: (i - (n_valid - 1) / 2) * offset_width
                for i, combo in enumerate(valid_combinations)
            }
        else:
            offsets = {combo: 0.0 for combo in valid_combinations}
    else:
        offsets = {}

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
            n_failed = int(np.sum(iters == NOT_MISCLASSIFIED))

            attack_rate = len(misclassified) / n_total * 100
            label = f"{init}/{model} (n={n_total}, {attack_rate:.0f}%, fail={n_failed})"
            # Same color for same init (both MNIST and CIFAR10)
            color = get_color_for_init(init)
            linestyle = get_linestyle_for_model(model, dataset)
            lw = linewidths.get(model, 2.0)
            ms = marker_sizes.get(model, 100)
            zo = zorders.get(model, 10)

            if len(misclassified) == 0:
                handle = ax.plot([], [], color="lightgray", linewidth=1.5, linestyle="--")[0]
                handles_labels.append((handle, label + " [FAILED]"))
                continue

            sorted_iters = np.sort(misclassified).astype(float)
            cdf = np.arange(1, len(sorted_iters) + 1) / n_total

            # CIFAR10: apply uniform x-offset for this series
            if dataset == "cifar10" and (init, model) in offsets:
                sorted_iters = sorted_iters + offsets[(init, model)]

            if n_total == 1:
                handle = ax.scatter(
                    sorted_iters,
                    cdf,
                    color=color,
                    marker=markers.get(model, "o"),
                    s=ms,
                    alpha=0.9,
                    zorder=zo,
                    edgecolors="black",
                    linewidths=0.5,
                )
            else:
                handle = ax.step(
                    sorted_iters,
                    cdf,
                    where="post",
                    color=color,
                    linewidth=lw,
                    linestyle=linestyle,
                    alpha=0.9,
                    zorder=zo,
                )[0]
            handles_labels.append((handle, label))

    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("Fraction misclassified", fontsize=18)
    ax.set_title(f"Misclassification CDF Overlay ({dataset.upper()}, all inits, single sample)", fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    if handles_labels:
        handles, labels = zip(*handles_labels)
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=14,
            ncol=4,
            frameon=True,
            fancybox=True,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Misclassification CDF overlay: {out_path}")


def plot_misclassification_histogram_overlay(
    stats_by_init: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
    dataset: str,
) -> None:
    """Plot histogram overlay comparing all init/model combinations."""
    fig, ax = plt.subplots(figsize=(16, 8))

    all_data = []
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
            if model_iters:
                all_data.append((init, model, np.array(model_iters)))

    if not all_data:
        print("[WARN] No data for histogram overlay")
        return

    all_misclassified = []
    for _, _, iters in all_data:
        misclassified = iters[iters >= 0]
        all_misclassified.extend(misclassified.tolist())

    if all_misclassified:
        max_iter = int(np.max(all_misclassified))
    else:
        max_iter = 100

    bin_edges = list(range(max_iter + 2))
    n_bins = len(bin_edges) - 1
    n_groups = len(all_data)
    bar_width = 0.85 / n_groups

    for idx, (init, model, iters) in enumerate(all_data):
        misclassified_iters = iters[iters >= 0]
        n_not_count = int(np.sum(iters == NOT_MISCLASSIFIED))
        n_total = len(iters)
        n_success = len(misclassified_iters)
        attack_rate = n_success / n_total * 100 if n_total > 0 else 0

        counts, _ = np.histogram(misclassified_iters, bins=bin_edges)

        if n_success == 0:
            color = "lightgray"
            label_suffix = " [FAILED]"
        else:
            color = get_color_for_init_model(init, model)
            label_suffix = ""

        x_positions = np.arange(n_bins) + idx * bar_width - (n_groups - 1) * bar_width / 2
        ax.bar(
            x_positions,
            counts,
            width=bar_width,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.7,
            label=f"{init}/{model} (n={n_total}, {attack_rate:.0f}%, fail={n_not_count}){label_suffix}",
        )

        fail_x = n_bins + 1.5 + idx * bar_width - (n_groups - 1) * bar_width / 2
        ax.bar(
            fail_x, n_not_count, width=bar_width, color=color,
            alpha=0.9, hatch="//", edgecolor="black", linewidth=0.7,
        )

    if max_iter <= 20:
        step = 5
    elif max_iter <= 50:
        step = 10
    else:
        step = 20
    x_tick_positions = list(range(0, n_bins + 1, step))
    if n_bins not in x_tick_positions:
        x_tick_positions.append(n_bins)
    x_tick_positions.append(n_bins + 1.5)
    x_tick_labels = [str(i) for i in x_tick_positions[:-1]] + ["FAIL"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("First Misclassification Iteration (FAIL = attack failed)")
    ax.set_ylabel("Count")
    ax.set_title(f"First Misclassification Distribution ({dataset.upper()}, all inits, single sample)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_xlim(-0.5, n_bins + 3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Misclassification histogram overlay: {out_path}")


def plot_model_comparison_strip(
    stats_by_model: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: str,
    dataset: str,
) -> None:
    """Plot strip plot comparing misclassification speed for each init type (2x2 subplots).

    Each subplot shows one init type with data points for each model.
    Models that never misclassified show N/A.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    model_colors = {
        "nat": "#1f77b4",
        "nat_and_adv": "#ff7f0e",
        "adv": "#2ca02c",
        "weak_adv": "#d62728",
    }

    for idx, init in enumerate(INIT_ORDER):
        ax = axes_flat[idx]
        all_y_max = 0

        for model_idx, model in enumerate(MODEL_ORDER):
            if model not in stats_by_model:
                continue
            stats = stats_by_model[model]
            if init not in stats:
                continue

            iters = stats[init]
            misclassified = iters[iters >= 0]
            color = model_colors.get(model, "gray")

            if len(misclassified) > 0:
                # Add jitter to x position to avoid overlap
                jitter = np.random.uniform(-0.15, 0.15, size=len(misclassified))
                x_positions = model_idx + jitter

                ax.scatter(
                    x_positions,
                    misclassified,
                    color=color,
                    alpha=0.7,
                    s=40,
                    edgecolors="black",
                    linewidths=0.3,
                    label=f"{model} (n={len(misclassified)})",
                )

                # Add median marker
                median_val = np.median(misclassified)
                ax.scatter(
                    model_idx,
                    median_val,
                    color=color,
                    marker="_",
                    s=300,
                    linewidths=3,
                    zorder=10,
                )

                all_y_max = max(all_y_max, np.max(misclassified))
            else:
                # No misclassification - show N/A
                ax.annotate(
                    "N/A",
                    xy=(model_idx, 0),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="gray",
                )

        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, fontsize=9)
        ax.set_ylabel("First Misclassification Iteration")
        ax.set_title(f"{init}", fontweight="bold", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)

        if all_y_max > 0:
            ax.set_ylim(0, all_y_max * 1.1)
        else:
            ax.set_ylim(0, 10)

    fig.suptitle(
        f"Misclassification Speed by Model ({dataset.upper()})\n"
        "Each dot = one sample, horizontal line = median",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Model comparison strip plot: {out_path}")


def plot_init_comparison_cdf(
    stats: Dict[str, np.ndarray],
    out_path: str,
    dataset: str,
    model: str,
) -> None:
    """Plot CDF comparing inits within a single model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    handles_labels = []

    for init in INIT_ORDER:
        if init not in stats:
            continue

        iters = stats[init]
        misclassified = iters[iters >= 0]
        n_total = len(iters)
        n_failed = int(np.sum(iters == NOT_MISCLASSIFIED))

        attack_rate = len(misclassified) / n_total * 100 if n_total > 0 else 0
        label = f"{init} (n={n_total}, {attack_rate:.0f}%, fail={n_failed})"
        color = get_color_for_init(init)
        linestyle = get_linestyle_for_init(init)

        if len(misclassified) == 0:
            handle = ax.plot([], [], color="lightgray", linewidth=1.5, linestyle="--")[0]
            handles_labels.append((handle, label + " [FAILED]"))
            continue

        sorted_iters = np.sort(misclassified)
        cdf = np.arange(1, len(sorted_iters) + 1) / n_total

        if n_total == 1:
            handle = ax.scatter(
                sorted_iters,
                cdf,
                color=color,
                marker="o",
                s=100,
                alpha=0.9,
                zorder=10,
                edgecolors="black",
                linewidths=0.5,
            )
        else:
            handle = ax.step(
                sorted_iters,
                cdf,
                where="post",
                color=color,
                linewidth=2.5,
                linestyle=linestyle,
                alpha=0.9,
            )[0]
        handles_labels.append((handle, label))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction misclassified")
    ax.set_title(f"Misclassification CDF ({dataset.upper()}, model={model})")

    if handles_labels:
        handles, labels = zip(*handles_labels)
        ax.legend(handles, labels, loc="lower right", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Init comparison CDF: {out_path}")


def plot_init_comparison_histogram(
    stats: Dict[str, np.ndarray],
    out_path: str,
    dataset: str,
    model: str,
) -> None:
    """Plot histogram comparing inits within a single model."""
    fig, ax = plt.subplots(figsize=(14, 6))

    all_data = []
    for init in INIT_ORDER:
        if init not in stats:
            continue
        all_data.append((init, stats[init]))

    if not all_data:
        print(f"[WARN] No data for histogram: {model}")
        plt.close()
        return

    all_misclassified = []
    for _, iters in all_data:
        misclassified = iters[iters >= 0]
        all_misclassified.extend(misclassified.tolist())

    if all_misclassified:
        max_iter = int(np.max(all_misclassified))
    else:
        max_iter = 100

    bin_edges = list(range(max_iter + 2))
    n_bins = len(bin_edges) - 1
    n_groups = len(all_data)
    bar_width = 0.8 / n_groups

    for idx, (init, iters) in enumerate(all_data):
        misclassified_iters = iters[iters >= 0]
        n_not_count = int(np.sum(iters == NOT_MISCLASSIFIED))
        n_total = len(iters)
        n_success = len(misclassified_iters)
        attack_rate = n_success / n_total * 100 if n_total > 0 else 0

        counts, _ = np.histogram(misclassified_iters, bins=bin_edges)

        if n_success == 0:
            color = "lightgray"
        else:
            color = get_color_for_init(init)

        x_positions = np.arange(n_bins) + idx * bar_width - (n_groups - 1) * bar_width / 2
        ax.bar(
            x_positions,
            counts,
            width=bar_width,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label=f"{init} (n={n_total}, {attack_rate:.0f}%, fail={n_not_count})",
        )

        fail_x = n_bins + 1.5 + idx * bar_width - (n_groups - 1) * bar_width / 2
        ax.bar(
            fail_x, n_not_count, width=bar_width, color=color,
            alpha=0.85, hatch="//", edgecolor="black", linewidth=0.5,
        )

    if max_iter <= 20:
        step = 5
    elif max_iter <= 50:
        step = 10
    else:
        step = 20
    x_tick_positions = list(range(0, n_bins + 1, step))
    if n_bins not in x_tick_positions:
        x_tick_positions.append(n_bins)
    x_tick_positions.append(n_bins + 1.5)
    x_tick_labels = [str(i) for i in x_tick_positions[:-1]] + ["FAIL"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("First Misclassification Iteration (FAIL = attack failed)")
    ax.set_ylabel("Count")
    ax.set_title(f"First Misclassification Distribution ({dataset.upper()}, model={model})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.5, n_bins + 3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Init comparison histogram: {out_path}")


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
    models = sorted(set(d.model for d in data_list))

    # Create model directories (new structure: dataset/model/)
    for dataset in datasets:
        for model in models:
            subdir = os.path.join(result_dir, dataset, model)
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

    # stats_by_dataset_model[dataset][model][init] = array of first misclassification iterations
    stats_by_dataset_model: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    # stats_by_dataset_init[dataset][init][model][init] for overlay (legacy structure)
    stats_by_dataset_init: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}

    for dataset in datasets:
        stats_by_dataset_model[dataset] = {}
        stats_by_dataset_init[dataset] = {}

        for model in MODEL_ORDER:
            subset_data = [
                d for d in data_list if d.dataset == dataset and d.model == model
            ]
            if not subset_data:
                continue

            # Aggregate by init within this model
            init_stats: Dict[str, List[int]] = {}
            for data in subset_data:
                if data.init not in init_stats:
                    init_stats[data.init] = []
                first_wrong = compute_first_misclassification(data.corrects)
                init_stats[data.init].extend(first_wrong.tolist())

            model_stats: Dict[str, np.ndarray] = {}
            for init in init_stats:
                model_stats[init] = np.array(init_stats[init])

            stats_by_dataset_model[dataset][model] = model_stats

            # Generate init comparison plots within each model directory
            subdir = os.path.join(result_dir, dataset, model)

            # CDF: skip if all inits are single-point (clean, deepfool)
            has_multi_point = any(
                init in model_stats and len(model_stats[init]) > 1
                for init in ["random", "multi_deepfool"]
            )
            if has_multi_point:
                plot_init_comparison_cdf(
                    model_stats,
                    os.path.join(subdir, "misclassification_cdf.png"),
                    dataset,
                    model,
                )

            plot_init_comparison_histogram(
                model_stats,
                os.path.join(subdir, "misclassification_histogram.png"),
                dataset,
                model,
            )

        # Build legacy structure for overlay (init -> model -> init)
        inits = sorted(set(d.init for d in data_list if d.dataset == dataset))
        for init in inits:
            subset_data = [
                d for d in data_list if d.dataset == dataset and d.init == init
            ]
            if not subset_data:
                continue
            subset_stats = aggregate_misclassification_stats(subset_data)
            stats_by_dataset_init[dataset][init] = subset_stats

    for dataset in datasets:
        dataset_dir = os.path.join(result_dir, dataset)

        # Generate overlay plots
        if dataset in stats_by_dataset_init and stats_by_dataset_init[dataset]:
            plot_misclassification_cdf_overlay(
                stats_by_dataset_init[dataset],
                os.path.join(dataset_dir, f"{dataset}_misclassification_cdf_overlay.png"),
                dataset,
            )
            plot_misclassification_histogram_overlay(
                stats_by_dataset_init[dataset],
                os.path.join(dataset_dir, "misclassification_histogram_overlay.png"),
                dataset,
            )

        # Generate model comparison strip plot
        if dataset in stats_by_dataset_model and stats_by_dataset_model[dataset]:
            plot_model_comparison_strip(
                stats_by_dataset_model[dataset],
                os.path.join(dataset_dir, "model_comparison_strip.png"),
                dataset,
            )

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
