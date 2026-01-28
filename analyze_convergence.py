"""
Convergence analysis for PGD loss curves.

Analyzes how many iterations are needed to reach local maximum loss,
regardless of model type.

Usage:
    python analyze_convergence.py --input_dir outputs/arrays/run_all/
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

# Fixed ordering for consistent plots
MODEL_ORDER = ["nat", "nat_and_adv", "weak_adv", "adv"]  # top to bottom
INIT_ORDER = ["random", "deepfool", "multi_deepfool"]  # left to right

# Markers for unconverged samples
NC_NEVER_REACHED = -1  # Never reached threshold
NC_UNSTABLE = -2       # Reached threshold but not stable in final window


class LossData:
    """Container for loss data from a single experiment."""

    __slots__ = ("filepath", "dataset", "model", "init", "panel_index", "losses")

    def __init__(
        self,
        filepath: str,
        dataset: str,
        model: str,
        init: str,
        panel_index: int,
        losses: np.ndarray,
    ) -> None:
        self.filepath = filepath
        self.dataset = dataset
        self.model = model
        self.init = init
        self.panel_index = panel_index
        self.losses = losses  # shape: (restarts, iterations+1)

    @property
    def num_restarts(self) -> int:
        return self.losses.shape[0]

    @property
    def num_iterations(self) -> int:
        return self.losses.shape[1] - 1  # -1 because iter 0 is initial state


def parse_filename(filepath: str) -> Optional[Tuple[str, str, str, int]]:
    """Parse filename to extract dataset, model, init, panel_index.

    Expected patterns:
    - mnist_nat_random_idx0_p0_losses.npy
    - cifar10_adv_deepfool_idx0-1-2_dfiter50_dfo0.02_p0_losses.npy
    - mnist_nat_and_adv_multi_deepfool_idx0_dfiter50_dfo0.02_p0_losses.npy
    """
    basename = os.path.basename(filepath)

    # Extract panel index
    panel_match = re.search(r"_p(\d+)_losses\.npy$", basename)
    if not panel_match:
        return None
    panel_index = int(panel_match.group(1))

    # Remove panel suffix for further parsing
    prefix = basename[: panel_match.start()]

    # Parse dataset
    if prefix.startswith("mnist_"):
        dataset = "mnist"
        rest = prefix[6:]
    elif prefix.startswith("cifar10_"):
        dataset = "cifar10"
        rest = prefix[8:]
    else:
        return None

    # Parse model and init
    # Models: nat, adv, nat_and_adv, weak_adv
    # Inits: random, deepfool, multi_deepfool
    model_patterns = ["nat_and_adv", "weak_adv", "nat", "adv"]
    init_patterns = ["multi_deepfool", "deepfool", "random"]

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


def load_loss_files(input_dir: str) -> List[LossData]:
    """Load all *_losses.npy files from input directory."""
    data_list = []

    for filename in os.listdir(input_dir):
        if not filename.endswith("_losses.npy"):
            continue

        filepath = os.path.join(input_dir, filename)
        parsed = parse_filename(filepath)

        if parsed is None:
            print(f"[WARN] Could not parse: {filename}")
            continue

        dataset, model, init, panel_index = parsed
        losses = np.load(filepath)

        data_list.append(
            LossData(
                filepath=filepath,
                dataset=dataset,
                model=model,
                init=init,
                panel_index=panel_index,
                losses=losses,
            )
        )

    print(f"[INFO] Loaded {len(data_list)} loss files from {input_dir}")
    return data_list


def compute_convergence_iteration(
    losses: np.ndarray, threshold: float = 0.99, stable_window: int = 10
) -> np.ndarray:
    """Compute iteration where normalized loss convergence ratio reaches threshold.

    Uses max loss as reference: (loss - initial) / (max - initial)
    Requires stability: last `stable_window` iterations must maintain >= threshold.

    Args:
        losses: shape (restarts, iterations+1)
        threshold: fraction of progress to consider "converged" (0 to 1)
        stable_window: number of final iterations that must maintain >= threshold

    Returns:
        convergence_iters: shape (restarts,) - iteration index where converged
            >= 0: converged at that iteration
            NC_NEVER_REACHED (-1): never reached threshold
            NC_UNSTABLE (-2): reached threshold but not stable
    """
    initial = losses[:, :1]  # shape (restarts, 1)
    max_loss = np.max(losses, axis=1, keepdims=True)  # shape (restarts, 1)

    # Compute normalized convergence ratio using max loss
    denom = max_loss - initial
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    rho = (losses - initial) / denom

    # Find first iteration where rho >= threshold
    reached = rho >= threshold  # shape (restarts, iterations+1)

    # argmax on bool array finds first True
    convergence_iters = np.argmax(reached, axis=1)

    # Check stability: last stable_window iterations must be >= threshold
    last_n = rho[:, -stable_window:]
    stable = np.all(last_n >= threshold, axis=1)

    # Classify: converged, never reached, or unstable
    never_reached = ~np.any(reached, axis=1)
    convergence_iters = convergence_iters.astype(np.int32)
    convergence_iters[never_reached] = NC_NEVER_REACHED
    convergence_iters[~never_reached & ~stable] = NC_UNSTABLE

    return convergence_iters


def compute_normalized_losses(losses: np.ndarray) -> np.ndarray:
    """Normalize losses using max loss as reference.

    Normalizes so that:
    - 0 = initial loss (iter 0)
    - 1 = max loss (across all iterations)

    Args:
        losses: shape (restarts, iterations+1)

    Returns:
        normalized: shape (restarts, iterations+1), values in [0, 1]
    """
    initial = losses[:, :1]  # shape (restarts, 1)
    max_loss = np.max(losses, axis=1, keepdims=True)  # shape (restarts, 1)

    denom = max_loss - initial
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    normalized = (losses - initial) / denom
    return normalized


def aggregate_convergence_stats(
    data_list: List[LossData], threshold: float = 0.99, stable_window: int = 10
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate convergence statistics by model and init.

    Returns:
        stats[model][init] = array of convergence iterations
    """
    stats: Dict[str, Dict[str, List[int]]] = {}

    for data in data_list:
        if data.model not in stats:
            stats[data.model] = {}
        if data.init not in stats[data.model]:
            stats[data.model][data.init] = []

        conv_iters = compute_convergence_iteration(data.losses, threshold, stable_window)
        stats[data.model][data.init].extend(conv_iters.tolist())

    # Convert lists to arrays
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for model in stats:
        result[model] = {}
        for init in stats[model]:
            result[model][init] = np.array(stats[model][init])

    return result


def print_convergence_summary(
    stats: Dict[str, Dict[str, np.ndarray]], threshold: float
) -> None:
    """Print summary table of convergence statistics."""
    print(f"\n{'=' * 90}")
    print(f"Convergence Summary (threshold={threshold:.0%} of max loss)")
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
            converged = iters[iters >= 0]
            n_converged = len(converged)
            n_never = int(np.sum(iters == NC_NEVER_REACHED))
            n_unstable = int(np.sum(iters == NC_UNSTABLE))
            conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

            if n_converged > 0:
                print(
                    f"  {init:20s}: "
                    f"conv={conv_rate:5.1f}%, "
                    f"mean={np.mean(converged):5.1f}, "
                    f"median={np.median(converged):5.1f}, "
                    f"p95={np.percentile(converged, 95):5.1f}, "
                    f"max={np.max(converged):5.0f}, "
                    f"n={n_total}, "
                    f"NC(nr/us)={n_never}/{n_unstable}"
                )
            else:
                print(
                    f"  {init:20s}: "
                    f"conv={conv_rate:5.1f}%, "
                    f"n={n_total}, "
                    f"NC(nr/us)={n_never}/{n_unstable}"
                )

    if all_iters:
        all_iters = np.array(all_iters)
        converged = all_iters[all_iters >= 0]
        n_total = len(all_iters)
        n_converged = len(converged)
        n_never = int(np.sum(all_iters == NC_NEVER_REACHED))
        n_unstable = int(np.sum(all_iters == NC_UNSTABLE))
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

        print(f"\n{'=' * 90}")
        print("[ALL MODELS COMBINED]")
        if n_converged > 0:
            print(
                f"  conv={conv_rate:5.1f}%, "
                f"mean={np.mean(converged):5.1f}, "
                f"median={np.median(converged):5.1f}, "
                f"p95={np.percentile(converged, 95):5.1f}, "
                f"max={np.max(converged):5.0f}, "
                f"n={n_total}, "
                f"NC(nr/us)={n_never}/{n_unstable}"
            )
        else:
            print(
                f"  conv={conv_rate:5.1f}%, "
                f"n={n_total}, "
                f"NC(nr/us)={n_never}/{n_unstable}"
            )
        print(f"{'=' * 90}")


def generate_markdown_summary(
    stats: Dict[str, Dict[str, np.ndarray]], threshold: float
) -> str:
    """Generate markdown formatted summary.

    Returns markdown string with:
    - Convergence statistics table
    - Detailed table (model × init breakdown)
    - Summary table (model-level aggregation)
    - Overall summary
    """
    lines = []
    lines.append("# Convergence Analysis Summary")
    lines.append("")
    lines.append(f"**Threshold**: {threshold:.0%} of max loss")
    lines.append("")
    lines.append("**NC Types**:")
    lines.append("- Never Reached (NR): Never reached threshold")
    lines.append("- Unstable (US): Reached threshold but not stable in final window")
    lines.append("")

    # Detailed table: model × init
    lines.append("## Detailed Results (by model and init)")
    lines.append("")
    lines.append(
        "| Model | Init | Conv Rate | Mean | Median | P95 | Max | N | NR | US |"
    )
    lines.append(
        "|-------|------|----------:|-----:|-------:|----:|----:|--:|---:|---:|"
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
            converged = iters[iters >= 0]
            n_converged = len(converged)
            n_never = int(np.sum(iters == NC_NEVER_REACHED))
            n_unstable = int(np.sum(iters == NC_UNSTABLE))
            conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

            if n_converged > 0:
                lines.append(
                    f"| {model} | {init} | "
                    f"{conv_rate:.1f}% | "
                    f"{np.mean(converged):.1f} | "
                    f"{np.median(converged):.1f} | "
                    f"{np.percentile(converged, 95):.1f} | "
                    f"{np.max(converged):.0f} | "
                    f"{n_total} | "
                    f"{n_never} | "
                    f"{n_unstable} |"
                )
            else:
                lines.append(
                    f"| {model} | {init} | "
                    f"{conv_rate:.1f}% | "
                    f"N/A | N/A | N/A | N/A | "
                    f"{n_total} | "
                    f"{n_never} | "
                    f"{n_unstable} |"
                )

    lines.append("")

    # Model-level summary table
    lines.append("## Model-level Summary")
    lines.append("")
    lines.append("| Model | Conv Rate | Mean | Median | P95 | Max | N | NR | US |")
    lines.append("|-------|----------:|-----:|-------:|----:|----:|--:|---:|---:|")

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        model_iters = np.array(model_iters)

        n_total = len(model_iters)
        converged = model_iters[model_iters >= 0]
        n_converged = len(converged)
        n_never = int(np.sum(model_iters == NC_NEVER_REACHED))
        n_unstable = int(np.sum(model_iters == NC_UNSTABLE))
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

        if n_converged > 0:
            lines.append(
                f"| {model} | "
                f"{conv_rate:.1f}% | "
                f"{np.mean(converged):.1f} | "
                f"{np.median(converged):.1f} | "
                f"{np.percentile(converged, 95):.1f} | "
                f"{np.max(converged):.0f} | "
                f"{n_total} | "
                f"{n_never} | "
                f"{n_unstable} |"
            )
        else:
            lines.append(
                f"| {model} | "
                f"{conv_rate:.1f}% | "
                f"N/A | N/A | N/A | N/A | "
                f"{n_total} | "
                f"{n_never} | "
                f"{n_unstable} |"
            )

    lines.append("")

    # Overall summary
    if all_iters:
        all_iters = np.array(all_iters)
        converged = all_iters[all_iters >= 0]
        n_total = len(all_iters)
        n_converged = len(converged)
        n_never = int(np.sum(all_iters == NC_NEVER_REACHED))
        n_unstable = int(np.sum(all_iters == NC_UNSTABLE))
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

        lines.append("## Overall Summary (All Models Combined)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Convergence Rate | {conv_rate:.1f}% |")
        lines.append(f"| Total samples | {n_total} |")
        lines.append(f"| Converged | {n_converged} |")
        lines.append(f"| NC (Never Reached) | {n_never} |")
        lines.append(f"| NC (Unstable) | {n_unstable} |")

        if n_converged > 0:
            lines.append(f"| Mean (converged) | {np.mean(converged):.1f} |")
            lines.append(f"| Median (converged) | {np.median(converged):.1f} |")
            lines.append(f"| P95 (converged) | {np.percentile(converged, 95):.1f} |")
            lines.append(f"| Max (converged) | {np.max(converged):.0f} |")
        lines.append("")

        # Recommendation
        if n_converged > 0:
            p95 = np.percentile(converged, 95)
            p99 = np.percentile(converged, 99)
            lines.append("## Recommendation")
            lines.append("")
            lines.append(
                f"- **95% of converged samples** reach {threshold:.0%} "
                f"of max loss by iteration **{p95:.0f}**"
            )
            lines.append(
                f"- **99% of converged samples** reach {threshold:.0%} "
                f"of max loss by iteration **{p99:.0f}**"
            )
            lines.append("")

    return "\n".join(lines)


def save_markdown_summary(
    stats: Dict[str, Dict[str, np.ndarray]], threshold: float, out_path: str
) -> None:
    """Save markdown summary to file."""
    md_content = generate_markdown_summary(stats, threshold)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SAVE] Markdown summary: {out_path}")


def plot_normalized_loss_curves(
    data_list: List[LossData],
    out_path: str,
    threshold: float,
    dataset: str,
    init: str,
    max_curves_per_group: int = 50,
) -> None:
    """Plot normalized loss curves grouped by model.

    Args:
        data_list: List of LossData objects
        out_path: Output file path
        threshold: Convergence threshold for horizontal line
        dataset: Dataset name (e.g., "mnist", "cifar10")
        init: Initialization method (e.g., "random", "deepfool")
        max_curves_per_group: Maximum curves to plot per subplot
    """
    # Use fixed ordering
    available_models = set(d.model for d in data_list)
    available_inits = set(d.init for d in data_list)
    models = [m for m in MODEL_ORDER if m in available_models]
    inits = [i for i in INIT_ORDER if i in available_inits]

    if not models or not inits:
        print("[WARN] No data for normalized loss curves")
        return

    fig, axes = plt.subplots(
        len(models), len(inits), figsize=(5 * len(inits), 4 * len(models)), squeeze=False
    )
    fig.suptitle(
        f"Normalized Loss Curves ({dataset.upper()}, init={init})",
        fontsize=14, y=1.02
    )

    for i, model in enumerate(models):
        for j, init_key in enumerate(inits):
            ax = axes[i, j]

            subset = [d for d in data_list if d.model == model and d.init == init_key]
            if not subset:
                ax.set_visible(False)
                continue

            # Collect all normalized curves
            all_curves = []
            for data in subset:
                norm = compute_normalized_losses(data.losses)
                all_curves.append(norm)

            if not all_curves:
                continue

            # Concatenate and subsample if too many
            all_curves = np.concatenate(all_curves, axis=0)
            if len(all_curves) > max_curves_per_group:
                indices = np.random.choice(
                    len(all_curves), max_curves_per_group, replace=False
                )
                all_curves = all_curves[indices]

            xs = np.arange(all_curves.shape[1])
            for curve in all_curves:
                ax.plot(xs, curve, alpha=0.3, linewidth=0.5, color="steelblue")

            # Plot mean curve
            mean_curve = np.mean(all_curves, axis=0)
            ax.plot(xs, mean_curve, color="red", linewidth=2, label="mean")

            ax.set_title(f"model={model}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Normalized Loss")
            ax.set_ylim(-0.1, 1.1)
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(
                y=threshold,
                color="orange",
                linestyle="--",
                alpha=0.5,
                label=f"{threshold:.0%}",
            )

            # Add threshold to y-axis ticks
            yticks = list(ax.get_yticks())
            if threshold not in yticks:
                yticks.append(threshold)
                yticks = sorted([y for y in yticks if -0.1 <= y <= 1.1])
                ax.set_yticks(yticks)

            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Normalized loss curves: {out_path}")


def plot_convergence_histogram(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    threshold: float,
    dataset: str,
    init: str,
) -> None:
    """Plot histogram of convergence iterations with side-by-side bars.

    Includes two NC bins at the right end:
    - NR (Never Reached): samples that never reached threshold
    - US (Unstable): samples that reached threshold but not stable
    """
    # Collect all iterations using fixed model order
    all_iters_by_model = {}
    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        if model_iters:
            all_iters_by_model[model] = np.array(model_iters)

    if not all_iters_by_model:
        print("[WARN] No data for histogram")
        return

    models = [m for m in MODEL_ORDER if m in all_iters_by_model]
    n_models = len(models)
    fig, ax = plt.subplots(figsize=(14, 6))

    # Determine bin range (excluding NC values)
    all_converged = []
    for m in models:
        converged = all_iters_by_model[m][all_iters_by_model[m] >= 0]
        all_converged.extend(converged.tolist())

    if all_converged:
        max_iter = int(np.max(all_converged))
    else:
        max_iter = 100

    # Create bins: 0, 1, 2, ..., max_iter, NR, US
    bin_edges = list(range(max_iter + 2))  # 0 to max_iter+1
    n_bins = len(bin_edges) - 1
    bar_width = 0.8 / n_models

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for idx, model in enumerate(models):
        iters = all_iters_by_model[model]
        converged_iters = iters[iters >= 0]
        nc_never_count = int(np.sum(iters == NC_NEVER_REACHED))
        nc_unstable_count = int(np.sum(iters == NC_UNSTABLE))

        # Compute histogram for converged samples
        counts, _ = np.histogram(converged_iters, bins=bin_edges)

        # Plot side-by-side bars for converged
        x_positions = np.arange(n_bins) + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(
            x_positions,
            counts,
            width=bar_width,
            color=colors[idx],
            alpha=0.8,
            label=f"{model} (n={len(iters)})",
        )

        # Plot NR (Never Reached) bin
        nr_x = n_bins + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(nr_x, nc_never_count, width=bar_width, color=colors[idx], alpha=0.8)

        # Plot US (Unstable) bin
        us_x = n_bins + 1 + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(
            us_x, nc_unstable_count, width=bar_width, color=colors[idx],
            alpha=0.8, hatch="//",
        )

    # Set x-axis labels (every 10 iterations + NR + US)
    step = 10
    x_tick_positions = list(range(0, n_bins, step)) + [n_bins, n_bins + 1]
    x_tick_labels = [str(i) for i in range(0, n_bins, step)] + ["NR", "US"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("Iteration (NR=Never Reached, US=Unstable)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Convergence Iteration Distribution "
        f"({dataset.upper()}, init={init}, threshold={threshold:.0%})"
    )
    ax.legend()
    ax.set_xlim(-0.5, n_bins + 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Convergence histogram: {out_path}")


def plot_single_model_histogram(
    iters: np.ndarray,
    model: str,
    out_path: str,
    threshold: float,
    dataset: str,
    init: str,
) -> None:
    """Plot histogram of convergence iterations for a single model.

    Args:
        iters: Array of convergence iterations (may include NC values)
        model: Model name for title
        out_path: Output file path
        threshold: Convergence threshold for title
        dataset: Dataset name (e.g., "mnist", "cifar10")
        init: Initialization method (e.g., "random", "deepfool")
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    converged_iters = iters[iters >= 0]
    nc_never_count = int(np.sum(iters == NC_NEVER_REACHED))
    nc_unstable_count = int(np.sum(iters == NC_UNSTABLE))
    n_total = len(iters)
    n_converged = len(converged_iters)

    if n_converged > 0:
        max_iter = int(np.max(converged_iters))
    else:
        max_iter = 100

    # Create bins
    bin_edges = list(range(max_iter + 2))
    n_bins = len(bin_edges) - 1

    # Plot histogram for converged samples
    counts, _ = np.histogram(converged_iters, bins=bin_edges)
    ax.bar(range(n_bins), counts, width=0.8, color="steelblue", alpha=0.8)

    # Plot NR (Never Reached) bin
    ax.bar(n_bins, nc_never_count, width=0.8, color="salmon", alpha=0.8, label="NR")

    # Plot US (Unstable) bin
    ax.bar(
        n_bins + 1, nc_unstable_count, width=0.8, color="orange",
        alpha=0.8, hatch="//", label="US",
    )

    # Set x-axis labels (every 10 iterations + NR + US)
    step = 10
    x_tick_positions = list(range(0, n_bins, step)) + [n_bins, n_bins + 1]
    x_tick_labels = [str(i) for i in range(0, n_bins, step)] + ["NR", "US"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("Iteration (NR=Never Reached, US=Unstable)")
    ax.set_ylabel("Count")
    convergence_rate = n_converged / n_total * 100 if n_total > 0 else 0
    ax.set_title(
        f"Convergence Distribution "
        f"({dataset.upper()}, init={init}, model={model}, "
        f"threshold={threshold:.0%}, converged={convergence_rate:.1f}%)"
    )
    ax.set_xlim(-0.5, n_bins + 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if nc_never_count > 0 or nc_unstable_count > 0:
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Single model histogram: {out_path}")


def plot_convergence_cdf(
    stats: Dict[str, Dict[str, np.ndarray]],
    out_path: str,
    threshold: float,
    dataset: str,
    init: str,
) -> None:
    """Plot CDF of convergence iterations (what % converged by iteration N).

    Shows NC breakdown (Never Reached / Unstable) in legend.
    """
    all_iters_by_model = {}
    for model in stats:
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        if model_iters:
            all_iters_by_model[model] = np.array(model_iters)

    if not all_iters_by_model:
        return

    models = [m for m in MODEL_ORDER if m in all_iters_by_model]
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, model in enumerate(models):
        iters = all_iters_by_model[model]
        converged = iters[iters >= 0]  # Exclude NC
        n_total = len(iters)
        n_never = int(np.sum(iters == NC_NEVER_REACHED))
        n_unstable = int(np.sum(iters == NC_UNSTABLE))

        if len(converged) == 0:
            continue
        sorted_iters = np.sort(converged)
        # CDF: fraction of total samples converged by iteration N
        cdf = np.arange(1, len(sorted_iters) + 1) / n_total
        conv_rate = len(converged) / n_total * 100
        # Legend shows: model (conv%, NR:n, US:n)
        label = f"{model} ({conv_rate:.0f}%, NR:{n_never}, US:{n_unstable})"
        ax.step(
            sorted_iters,
            cdf,
            where="post",
            label=label,
            color=colors[idx],
            linewidth=2,
        )

    # Combined (excluding NC)
    all_iters_flat = np.concatenate(list(all_iters_by_model.values()))
    all_converged = all_iters_flat[all_iters_flat >= 0]
    n_total_all = len(all_iters_flat)
    n_never_all = int(np.sum(all_iters_flat == NC_NEVER_REACHED))
    n_unstable_all = int(np.sum(all_iters_flat == NC_UNSTABLE))

    if len(all_converged) > 0:
        all_sorted = np.sort(all_converged)
        cdf_combined = np.arange(1, len(all_sorted) + 1) / n_total_all
        conv_rate_all = len(all_converged) / n_total_all * 100
        label_all = f"ALL ({conv_rate_all:.0f}%, NR:{n_never_all}, US:{n_unstable_all})"
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
    ax.set_ylabel("Fraction converged")
    ax.set_title(
        f"Convergence CDF ({dataset.upper()}, init={init}, threshold={threshold:.0%})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Convergence CDF: {out_path}")


def plot_mean_loss_curves_overlay(
    data_list: List[LossData],
    out_path: str,
    threshold: float,
    dataset: str,
    init: str,
) -> None:
    """Plot mean normalized loss curves for all models overlaid.

    Args:
        data_list: List of LossData objects
        out_path: Output file path
        threshold: Convergence threshold for horizontal line
        dataset: Dataset name (e.g., "mnist", "cifar10")
        init: Initialization method (e.g., "random", "deepfool")
    """
    # Use fixed model ordering
    available_models = set(d.model for d in data_list)
    models = [m for m in MODEL_ORDER if m in available_models]

    if not models:
        print("[WARN] No data for mean loss overlay")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, model in enumerate(models):
        subset = [d for d in data_list if d.model == model]
        if not subset:
            continue

        all_norm = []
        for data in subset:
            norm = compute_normalized_losses(data.losses)
            all_norm.append(norm)

        all_norm = np.concatenate(all_norm, axis=0)
        mean_curve = np.mean(all_norm, axis=0)
        std_curve = np.std(all_norm, axis=0)

        xs = np.arange(len(mean_curve))
        ax.plot(xs, mean_curve, color=colors[idx], linewidth=2, label=model)
        ax.fill_between(
            xs,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=colors[idx],
            alpha=0.2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Loss (0=initial, 1=max)")
    ax.set_title(f"Mean Normalized Loss Curves ({dataset.upper()}, init={init})")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(
        y=threshold,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"threshold={threshold:.0%}",
    )
    ax.set_ylim(-0.1, 1.2)

    # Add threshold to y-axis ticks
    yticks = list(ax.get_yticks())
    if threshold not in yticks:
        yticks.append(threshold)
        yticks = sorted([y for y in yticks if -0.1 <= y <= 1.2])
        ax.set_yticks(yticks)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Mean loss curves overlay: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze PGD convergence from loss arrays"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_losses.npy files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/work/outputs",
        help="Base output directory (default: /work/outputs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Convergence threshold as fraction of max loss (default: 0.99)",
    )
    parser.add_argument(
        "--stable_window",
        type=int,
        default=10,
        help="Number of final iterations that must maintain >= threshold (default: 10)",
    )
    args = parser.parse_args()

    # Set output directory: out_dir/convergence_analysis/threshold_{threshold}/
    threshold_str = f"threshold_{args.threshold:.2f}".replace(".", "_")
    result_dir = os.path.join(args.out_dir, "convergence_analysis", threshold_str)
    os.makedirs(result_dir, exist_ok=True)

    # Load data
    data_list = load_loss_files(args.input_dir)
    if not data_list:
        print("[ERROR] No loss files found")
        return

    # Get available datasets and inits
    datasets = sorted(set(d.dataset for d in data_list))
    inits = sorted(set(d.init for d in data_list))

    # Create subdirectories for each dataset/init combination
    for dataset in datasets:
        for init in inits:
            subdir = os.path.join(result_dir, dataset, init)
            os.makedirs(subdir, exist_ok=True)

    # Generate reports and plots for each dataset
    for dataset in datasets:
        # Filter data for this dataset
        dataset_data = [d for d in data_list if d.dataset == dataset]
        if not dataset_data:
            continue

        # Compute stats for this dataset
        dataset_stats = aggregate_convergence_stats(
            dataset_data, args.threshold, args.stable_window
        )

        # Print summary to console
        print(f"\n{'#' * 40}")
        print(f"# Dataset: {dataset.upper()}")
        print(f"{'#' * 40}")
        print_convergence_summary(dataset_stats, args.threshold)

        # Save markdown summary to dataset directory
        dataset_dir = os.path.join(result_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        save_markdown_summary(
            dataset_stats,
            args.threshold,
            os.path.join(dataset_dir, "convergence_report.md"),
        )

    # Generate plots for each dataset/init combination
    for dataset in datasets:
        for init in inits:
            subdir = os.path.join(result_dir, dataset, init)

            # Filter data for this dataset/init
            subset_data = [
                d for d in data_list if d.dataset == dataset and d.init == init
            ]
            if not subset_data:
                continue

            # Compute stats for this subset
            subset_stats = aggregate_convergence_stats(
                subset_data, args.threshold, args.stable_window
            )

            # Plot normalized loss curves
            plot_normalized_loss_curves(
                subset_data,
                os.path.join(subdir, "normalized_loss_curves.png"),
                args.threshold,
                dataset,
                init,
            )

            # Plot combined histogram
            plot_convergence_histogram(
                subset_stats,
                os.path.join(subdir, "convergence_histogram.png"),
                args.threshold,
                dataset,
                init,
            )

            # Plot individual model histograms
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
                        os.path.join(subdir, f"convergence_histogram_{model}.png"),
                        args.threshold,
                        dataset,
                        init,
                    )

            # Plot mean loss overlay
            plot_mean_loss_curves_overlay(
                subset_data,
                os.path.join(subdir, "mean_loss_overlay.png"),
                args.threshold,
                dataset,
                init,
            )

            # Plot convergence CDF
            plot_convergence_cdf(
                subset_stats,
                os.path.join(subdir, "convergence_cdf.png"),
                args.threshold,
                dataset,
                init,
            )

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
