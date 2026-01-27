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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


@dataclass
class LossData:
    """Container for loss data from a single experiment."""

    filepath: str
    dataset: str
    model: str
    init: str
    panel_index: int
    losses: np.ndarray  # shape: (restarts, iterations+1)

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
    """Compute iteration where loss reaches threshold * final_loss.

    Args:
        losses: shape (restarts, iterations+1)
        threshold: fraction of final loss to consider "converged"

    Returns:
        convergence_iters: shape (restarts,) - iteration index where converged
    """
    final_loss = losses[:, -1:]  # shape (restarts, 1)
    target = final_loss * threshold

    # Find first iteration where loss >= target
    reached = losses >= target  # shape (restarts, iterations+1)

    # argmax on bool array finds first True
    convergence_iters = np.argmax(reached, axis=1)

    # Handle case where never reached (shouldn't happen if threshold < 1)
    never_reached = ~np.any(reached, axis=1)
    convergence_iters[never_reached] = losses.shape[1] - 1

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
    print(f"\n{'=' * 70}")
    print(f"Convergence Summary (threshold={threshold:.0%} of final loss)")
    print(f"{'=' * 70}")

    all_iters = []

    for model in sorted(stats.keys()):
        print(f"\n[{model}]")
        for init in sorted(stats[model].keys()):
            iters = stats[model][init]
            all_iters.extend(iters.tolist())

            print(
                f"  {init:20s}: "
                f"mean={np.mean(iters):5.1f}, "
                f"median={np.median(iters):5.1f}, "
                f"p95={np.percentile(iters, 95):5.1f}, "
                f"max={np.max(iters):5.0f}, "
                f"n={len(iters)}"
            )

    if all_iters:
        all_iters = np.array(all_iters)
        print(f"\n{'=' * 70}")
        print("[ALL MODELS COMBINED]")
        print(
            f"  mean={np.mean(all_iters):5.1f}, "
            f"median={np.median(all_iters):5.1f}, "
            f"p95={np.percentile(all_iters, 95):5.1f}, "
            f"max={np.max(all_iters):5.0f}, "
            f"n={len(all_iters)}"
        )
        print(f"{'=' * 70}")


def generate_markdown_summary(
    stats: Dict[str, Dict[str, np.ndarray]], threshold: float
) -> str:
    """Generate markdown formatted summary.

    Returns markdown string with:
    - Detailed table (model × init breakdown)
    - Summary table (model-level aggregation)
    - Overall summary
    """
    lines = []
    lines.append(f"# Convergence Analysis Summary")
    lines.append(f"")
    lines.append(f"**Threshold**: {threshold:.0%} of final loss")
    lines.append(f"")

    # Detailed table: model × init
    lines.append(f"## Detailed Results (by model and init)")
    lines.append(f"")
    lines.append(f"| Model | Init | Mean | Median | P95 | Max | N |")
    lines.append(f"|-------|------|-----:|-------:|----:|----:|--:|")

    all_iters = []
    for model in sorted(stats.keys()):
        for init in sorted(stats[model].keys()):
            iters = stats[model][init]
            all_iters.extend(iters.tolist())
            lines.append(
                f"| {model} | {init} | "
                f"{np.mean(iters):.1f} | "
                f"{np.median(iters):.1f} | "
                f"{np.percentile(iters, 95):.1f} | "
                f"{np.max(iters):.0f} | "
                f"{len(iters)} |"
            )

    lines.append(f"")

    # Model-level summary table
    lines.append(f"## Model-level Summary")
    lines.append(f"")
    lines.append(f"| Model | Mean | Median | P95 | Max | N |")
    lines.append(f"|-------|-----:|-------:|----:|----:|--:|")

    for model in sorted(stats.keys()):
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        model_iters = np.array(model_iters)
        lines.append(
            f"| {model} | "
            f"{np.mean(model_iters):.1f} | "
            f"{np.median(model_iters):.1f} | "
            f"{np.percentile(model_iters, 95):.1f} | "
            f"{np.max(model_iters):.0f} | "
            f"{len(model_iters)} |"
        )

    lines.append(f"")

    # Overall summary
    if all_iters:
        all_iters = np.array(all_iters)
        lines.append(f"## Overall Summary (All Models Combined)")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|------:|")
        lines.append(f"| Mean | {np.mean(all_iters):.1f} |")
        lines.append(f"| Median | {np.median(all_iters):.1f} |")
        lines.append(f"| P95 | {np.percentile(all_iters, 95):.1f} |")
        lines.append(f"| Max | {np.max(all_iters):.0f} |")
        lines.append(f"| Total samples | {len(all_iters)} |")
        lines.append(f"")

        # Recommendation
        p95 = np.percentile(all_iters, 95)
        p99 = np.percentile(all_iters, 99)
        lines.append(f"## Recommendation")
        lines.append(f"")
        lines.append(
            f"- **95% of samples** reach {threshold:.0%} of final loss by iteration **{p95:.0f}**"
        )
        lines.append(
            f"- **99% of samples** reach {threshold:.0%} of final loss by iteration **{p99:.0f}**"
        )
        lines.append(f"")

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
    data_list: List[LossData], out_path: str, max_curves_per_group: int = 50
) -> None:
    """Plot normalized loss curves grouped by model."""
    models = sorted(set(d.model for d in data_list))
    inits = sorted(set(d.init for d in data_list))

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
            ax.axhline(y=0.99, color="orange", linestyle="--", alpha=0.5, label="99%")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Normalized loss curves: {out_path}")


def plot_convergence_histogram(
    stats: Dict[str, Dict[str, np.ndarray]], out_path: str, threshold: float
) -> None:
    """Plot histogram of convergence iterations."""
    # Collect all iterations
    all_iters_by_model = {}
    for model in stats:
        model_iters = []
        for init in stats[model]:
            model_iters.extend(stats[model][init].tolist())
        if model_iters:
            all_iters_by_model[model] = np.array(model_iters)

    if not all_iters_by_model:
        print("[WARN] No data for histogram")
        return

    models = sorted(all_iters_by_model.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine bin range
    all_vals = np.concatenate(list(all_iters_by_model.values()))
    max_iter = int(np.max(all_vals))
    bins = np.arange(0, max_iter + 2) - 0.5

    for model in models:
        ax.hist(
            all_iters_by_model[model],
            bins=bins,
            alpha=0.5,
            label=f"{model} (n={len(all_iters_by_model[model])})",
        )

    ax.set_xlabel("Iteration to reach threshold")
    ax.set_ylabel("Count")
    ax.set_title(f"Convergence Iteration Distribution (threshold={threshold:.0%})")
    ax.legend()
    ax.set_xlim(0, max_iter + 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] Convergence histogram: {out_path}")


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


def plot_mean_loss_curves_overlay(data_list: List[LossData], out_path: str) -> None:
    """Plot mean normalized loss curves for all models overlaid."""
    models = sorted(set(d.model for d in data_list))

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
    ax.axhline(y=0.99, color="red", linestyle=":", alpha=0.7)
    ax.set_ylim(-0.1, 1.2)
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
        default=None,
        help="Output directory for figures (default: input_dir/../convergence_analysis)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Convergence threshold as fraction of final loss (default: 0.99)",
    )
    args = parser.parse_args()

    # Set output directory
    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(args.input_dir.rstrip("/")), "convergence_analysis"
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    data_list = load_loss_files(args.input_dir)
    if not data_list:
        print("[ERROR] No loss files found")
        return

    # Compute convergence stats
    stats = aggregate_convergence_stats(data_list, args.threshold)

    # Print summary to console
    print_convergence_summary(stats, args.threshold)

    # Save markdown summary
    save_markdown_summary(
        stats, args.threshold, os.path.join(args.out_dir, "convergence_summary.md")
    )

    # Generate plots
    plot_normalized_loss_curves(
        data_list, os.path.join(args.out_dir, "normalized_loss_curves.png")
    )
    plot_convergence_histogram(
        stats, os.path.join(args.out_dir, "convergence_histogram.png"), args.threshold
    )
    plot_convergence_cdf(
        stats, os.path.join(args.out_dir, "convergence_cdf.png"), args.threshold
    )
    plot_mean_loss_curves_overlay(
        data_list, os.path.join(args.out_dir, "mean_loss_overlay.png")
    )

    print(f"\n[DONE] Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
