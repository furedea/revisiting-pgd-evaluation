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
import numpy as np

matplotlib.use("Agg")

# Fixed ordering for consistent plots
MODEL_ORDER = ["nat", "nat_and_adv", "weak_adv", "adv"]  # top to bottom
INIT_ORDER = ["random", "deepfool", "multi_deepfool"]  # left to right

# Marker for unconverged samples
NOT_CONVERGED = -1


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
    losses: np.ndarray, threshold: float = 0.99
) -> np.ndarray:
    """Compute iteration where normalized loss progress reaches threshold.

    Uses normalized progress: (loss - initial) / (final - initial)
    This ensures we measure progress from initial state toward final state.

    Args:
        losses: shape (restarts, iterations+1)
        threshold: fraction of progress to consider "converged" (0 to 1)

    Returns:
        convergence_iters: shape (restarts,) - iteration index where converged
            0-99: converged at that iteration
            100: converged at final iteration (if num_iterations=100)
            -1 (NOT_CONVERGED): never reached threshold
    """
    initial = losses[:, :1]  # shape (restarts, 1)
    final = losses[:, -1:]  # shape (restarts, 1)

    # Compute normalized progress
    denom = final - initial
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    progress = (losses - initial) / denom

    # Find first iteration where progress >= threshold
    reached = progress >= threshold  # shape (restarts, iterations+1)

    # argmax on bool array finds first True
    convergence_iters = np.argmax(reached, axis=1)

    # Handle case where never reached threshold
    never_reached = ~np.any(reached, axis=1)
    convergence_iters = convergence_iters.astype(np.int32)
    convergence_iters[never_reached] = NOT_CONVERGED

    return convergence_iters


def compute_normalized_losses(losses: np.ndarray) -> np.ndarray:
    """Normalize losses to [0, 1] range per restart.

    Normalizes so that:
    - 0 = initial loss (iter 0)
    - 1 = final loss (last iter)

    Args:
        losses: shape (restarts, iterations+1)

    Returns:
        normalized: shape (restarts, iterations+1)
    """
    initial = losses[:, :1]  # shape (restarts, 1)
    final = losses[:, -1:]  # shape (restarts, 1)

    denom = final - initial
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    normalized = (losses - initial) / denom
    return normalized


def aggregate_convergence_stats(
    data_list: List[LossData], threshold: float = 0.99
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

        conv_iters = compute_convergence_iteration(data.losses, threshold)
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
    print(f"\n{'=' * 80}")
    print(f"Convergence Summary (threshold={threshold:.0%} of normalized progress)")
    print(f"{'=' * 80}")

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
            converged = iters[iters != NOT_CONVERGED]
            n_converged = len(converged)
            n_unconverged = n_total - n_converged
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
                    f"NC={n_unconverged}"
                )
            else:
                print(
                    f"  {init:20s}: "
                    f"conv={conv_rate:5.1f}%, "
                    f"n={n_total}, "
                    f"NC={n_unconverged}"
                )

    if all_iters:
        all_iters = np.array(all_iters)
        converged = all_iters[all_iters != NOT_CONVERGED]
        n_total = len(all_iters)
        n_converged = len(converged)
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

        print(f"\n{'=' * 80}")
        print("[ALL MODELS COMBINED]")
        if n_converged > 0:
            print(
                f"  conv={conv_rate:5.1f}%, "
                f"mean={np.mean(converged):5.1f}, "
                f"median={np.median(converged):5.1f}, "
                f"p95={np.percentile(converged, 95):5.1f}, "
                f"max={np.max(converged):5.0f}, "
                f"n={n_total}, "
                f"NC={n_total - n_converged}"
            )
        else:
            print(f"  conv={conv_rate:5.1f}%, n={n_total}, NC={n_total - n_converged}")
        print(f"{'=' * 80}")


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
    lines.append(f"**Threshold**: {threshold:.0%} of normalized progress")
    lines.append("")

    # Convergence statistics table
    lines.append("## Convergence Statistics")
    lines.append("")
    lines.append("| Model | Convergence Rate | Mean Iter (converged) | Unconverged |")
    lines.append("|-------|----------------:|----------------------:|------------:|")

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        model_iters = np.array(model_iters)

        n_total = len(model_iters)
        converged = model_iters[model_iters != NOT_CONVERGED]
        n_converged = len(converged)
        n_unconverged = n_total - n_converged
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0
        mean_conv = np.mean(converged) if n_converged > 0 else float("nan")

        lines.append(
            f"| {model} | "
            f"{conv_rate:.1f}% | "
            f"{mean_conv:.1f} | "
            f"{n_unconverged} |"
        )

    lines.append("")

    # Detailed table: model × init
    lines.append("## Detailed Results (by model and init)")
    lines.append("")
    lines.append(
        "| Model | Init | Conv Rate | Mean | Median | P95 | Max | N | Unconverged |"
    )
    lines.append(
        "|-------|------|----------:|-----:|-------:|----:|----:|--:|------------:|"
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
            converged = iters[iters != NOT_CONVERGED]
            n_converged = len(converged)
            n_unconverged = n_total - n_converged
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
                    f"{n_unconverged} |"
                )
            else:
                lines.append(
                    f"| {model} | {init} | "
                    f"{conv_rate:.1f}% | "
                    f"N/A | N/A | N/A | N/A | "
                    f"{n_total} | "
                    f"{n_unconverged} |"
                )

    lines.append("")

    # Model-level summary table
    lines.append("## Model-level Summary")
    lines.append("")
    lines.append("| Model | Conv Rate | Mean | Median | P95 | Max | N |")
    lines.append("|-------|----------:|-----:|-------:|----:|----:|--:|")

    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        model_iters = np.array(model_iters)

        n_total = len(model_iters)
        converged = model_iters[model_iters != NOT_CONVERGED]
        n_converged = len(converged)
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

        if n_converged > 0:
            lines.append(
                f"| {model} | "
                f"{conv_rate:.1f}% | "
                f"{np.mean(converged):.1f} | "
                f"{np.median(converged):.1f} | "
                f"{np.percentile(converged, 95):.1f} | "
                f"{np.max(converged):.0f} | "
                f"{n_total} |"
            )
        else:
            lines.append(
                f"| {model} | "
                f"{conv_rate:.1f}% | "
                f"N/A | N/A | N/A | N/A | "
                f"{n_total} |"
            )

    lines.append("")

    # Overall summary
    if all_iters:
        all_iters = np.array(all_iters)
        converged = all_iters[all_iters != NOT_CONVERGED]
        n_total = len(all_iters)
        n_converged = len(converged)
        conv_rate = n_converged / n_total * 100 if n_total > 0 else 0

        lines.append("## Overall Summary (All Models Combined)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Convergence Rate | {conv_rate:.1f}% |")
        lines.append(f"| Total samples | {n_total} |")
        lines.append(f"| Converged | {n_converged} |")
        lines.append(f"| Unconverged | {n_total - n_converged} |")

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
                f"of progress by iteration **{p95:.0f}**"
            )
            lines.append(
                f"- **99% of converged samples** reach {threshold:.0%} "
                f"of progress by iteration **{p99:.0f}**"
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
    max_curves_per_group: int = 50,
) -> None:
    """Plot normalized loss curves grouped by model.

    Args:
        data_list: List of LossData objects
        out_path: Output file path
        threshold: Convergence threshold for horizontal line
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
    fig.suptitle("Normalized Loss Curves (0=initial, 1=final)", fontsize=14, y=1.02)

    for i, model in enumerate(models):
        for j, init in enumerate(inits):
            ax = axes[i, j]

            subset = [d for d in data_list if d.model == model and d.init == init]
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

            ax.set_title(f"{model} / {init}")
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
    stats: Dict[str, Dict[str, np.ndarray]], out_path: str, threshold: float
) -> None:
    """Plot histogram of convergence iterations with side-by-side bars.

    Includes an "NC" (Not Converged) bin at the right end for samples
    that never reached the threshold.
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
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine bin range (excluding NOT_CONVERGED=-1)
    all_converged = []
    for m in models:
        converged = all_iters_by_model[m][all_iters_by_model[m] != NOT_CONVERGED]
        all_converged.extend(converged.tolist())

    if all_converged:
        max_iter = int(np.max(all_converged))
    else:
        max_iter = 100

    # Create bins: 0, 1, 2, ..., max_iter, NC
    bin_edges = list(range(max_iter + 2))  # 0 to max_iter+1
    n_bins = len(bin_edges) - 1
    bar_width = 0.8 / n_models

    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for idx, model in enumerate(models):
        iters = all_iters_by_model[model]
        converged_iters = iters[iters != NOT_CONVERGED]
        nc_count = np.sum(iters == NOT_CONVERGED)

        # Compute histogram for converged samples
        counts, _ = np.histogram(converged_iters, bins=bin_edges)

        # Plot side-by-side bars
        x_positions = np.arange(n_bins) + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(
            x_positions,
            counts,
            width=bar_width,
            color=colors[idx],
            alpha=0.8,
            label=f"{model} (n={len(iters)})",
        )

        # Plot NC bin at the right
        nc_x = n_bins + idx * bar_width - (n_models - 1) * bar_width / 2
        ax.bar(nc_x, nc_count, width=bar_width, color=colors[idx], alpha=0.8)

    # Set x-axis labels
    x_tick_positions = list(range(n_bins)) + [n_bins]
    x_tick_labels = [str(i) for i in range(n_bins)] + ["NC"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=8)

    ax.set_xlabel("Iteration to reach threshold (NC = Not Converged)")
    ax.set_ylabel("Count")
    ax.set_title(f"Convergence Iteration Distribution (threshold={threshold:.0%})")
    ax.legend()
    ax.set_xlim(-0.5, n_bins + 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Convergence histogram: {out_path}")


def plot_single_model_histogram(
    iters: np.ndarray, model: str, out_path: str, threshold: float
) -> None:
    """Plot histogram of convergence iterations for a single model.

    Args:
        iters: Array of convergence iterations (may include NOT_CONVERGED=-1)
        model: Model name for title
        out_path: Output file path
        threshold: Convergence threshold for title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    converged_iters = iters[iters != NOT_CONVERGED]
    nc_count = int(np.sum(iters == NOT_CONVERGED))
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

    # Plot NC bin
    ax.bar(n_bins, nc_count, width=0.8, color="salmon", alpha=0.8)

    # Set x-axis labels
    x_tick_positions = list(range(n_bins)) + [n_bins]
    x_tick_labels = [str(i) for i in range(n_bins)] + ["NC"]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=8)

    ax.set_xlabel("Iteration to reach threshold (NC = Not Converged)")
    ax.set_ylabel("Count")
    convergence_rate = n_converged / n_total * 100 if n_total > 0 else 0
    ax.set_title(
        f"{model}: Convergence Distribution (threshold={threshold:.0%}, "
        f"converged={convergence_rate:.1f}%)"
    )
    ax.set_xlim(-0.5, n_bins + 1.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Single model histogram: {out_path}")


def plot_convergence_cdf(
    stats: Dict[str, Dict[str, np.ndarray]], out_path: str, threshold: float
) -> None:
    """Plot CDF of convergence iterations (what % converged by iteration N)."""
    all_iters_by_model = {}
    for model in stats:
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        if model_iters:
            all_iters_by_model[model] = np.array(model_iters)

    if not all_iters_by_model:
        return

    models = sorted(all_iters_by_model.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, model in enumerate(models):
        iters = np.sort(all_iters_by_model[model])
        cdf = np.arange(1, len(iters) + 1) / len(iters)
        ax.step(iters, cdf, where="post", label=model, color=colors[idx], linewidth=2)

    # Combined
    all_combined = np.sort(np.concatenate(list(all_iters_by_model.values())))
    cdf_combined = np.arange(1, len(all_combined) + 1) / len(all_combined)
    ax.step(
        all_combined,
        cdf_combined,
        where="post",
        label="ALL",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction converged")
    ax.set_title(f"Convergence CDF (threshold={threshold:.0%} of final loss)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color="red", linestyle=":", alpha=0.7, label="95%")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Convergence CDF: {out_path}")


def plot_mean_loss_curves_overlay(
    data_list: List[LossData], out_path: str, threshold: float
) -> None:
    """Plot mean normalized loss curves for all models overlaid.

    Args:
        data_list: List of LossData objects
        out_path: Output file path
        threshold: Convergence threshold for horizontal line
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
    ax.set_ylabel("Normalized Loss (0=initial, 1=final)")
    ax.set_title("Mean Normalized Loss Curves by Model")
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
        help="Convergence threshold as fraction of normalized progress (default: 0.99)",
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

    # Compute convergence stats
    stats = aggregate_convergence_stats(data_list, args.threshold)

    # Print summary to console
    print_convergence_summary(stats, args.threshold)

    # Save markdown summary to top-level directory
    save_markdown_summary(
        stats, args.threshold, os.path.join(result_dir, "convergence_report.md")
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
            subset_stats = aggregate_convergence_stats(subset_data, args.threshold)

            # Plot normalized loss curves
            plot_normalized_loss_curves(
                subset_data,
                os.path.join(subdir, "normalized_loss_curves.png"),
                args.threshold,
            )

            # Plot combined histogram
            plot_convergence_histogram(
                subset_stats,
                os.path.join(subdir, "convergence_histogram.png"),
                args.threshold,
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
                    )

            # Plot mean loss overlay
            plot_mean_loss_curves_overlay(
                subset_data,
                os.path.join(subdir, "mean_loss_overlay.png"),
                args.threshold,
            )

    # Also generate plots at the top level for all data combined
    plot_normalized_loss_curves(
        data_list,
        os.path.join(result_dir, "normalized_loss_curves.png"),
        args.threshold,
    )
    plot_convergence_histogram(
        stats,
        os.path.join(result_dir, "convergence_histogram.png"),
        args.threshold,
    )
    plot_convergence_cdf(
        stats, os.path.join(result_dir, "convergence_cdf.png"), args.threshold
    )
    plot_mean_loss_curves_overlay(
        data_list, os.path.join(result_dir, "mean_loss_overlay.png"), args.threshold
    )

    # Generate individual model histograms at top level
    for model in MODEL_ORDER:
        if model not in stats:
            continue
        model_iters = []
        for init_key in stats[model]:
            model_iters.extend(stats[model][init_key].tolist())
        if model_iters:
            plot_single_model_histogram(
                np.array(model_iters),
                model,
                os.path.join(result_dir, f"convergence_histogram_{model}.png"),
                args.threshold,
            )

    print(f"\n[DONE] Results saved to {result_dir}")


if __name__ == "__main__":
    main()
