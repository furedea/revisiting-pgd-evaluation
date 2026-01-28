"""Command-line interface argument parsing and formatting."""

import argparse
import os
from typing import Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    ap.add_argument("--model_src_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exp_name", required=True, help="Experiment name for subdirectory")

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=1)
    ap.add_argument("--max_tries", type=int, default=20000)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--total_iter", type=int, default=100)
    ap.add_argument("--num_restarts", type=int, default=20)

    ap.add_argument("--init", choices=["random", "deepfool", "clean"], default="random")
    ap.add_argument("--df_max_iter", type=int, default=50)
    ap.add_argument("--df_overshoot", type=float, default=0.02)
    ap.add_argument("--df_jitter", type=float, default=0.0)
    ap.add_argument("--init_sanity_plot", action="store_true")
    ap.add_argument(
        "--df_project",
        choices=["clip", "scale", "maxloss"],
        default="clip",
        help="How to enforce Linf<=eps for deepfool-init: clip(default) | scale | maxloss",
    )

    return ap


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if int(args.n_examples) < 1 or int(args.n_examples) > 5:
        raise ValueError("--n_examples must be 1..5")
    if str(args.init) == "deepfool" and int(args.df_max_iter) <= 0:
        raise ValueError("--df_max_iter must be > 0")


def get_model_tag(ckpt_dir: str) -> str:
    """Extract model tag from ckpt_dir basename (e.g., 'models/nat' -> 'nat')."""
    return os.path.basename(os.path.normpath(ckpt_dir))


def format_indices_part(indices: Tuple[int, ...]) -> str:
    """Format indices for file naming."""
    return f"idx{indices[0]}" if len(indices) == 1 else f"indices{'-'.join(str(i) for i in indices)}"


def format_base_name(args: argparse.Namespace, indices: Tuple[int, ...]) -> str:
    """Format base name for output files."""
    idx_part = format_indices_part(indices)
    tag = get_model_tag(str(args.ckpt_dir))
    df_part = (
        f"_dfiter{args.df_max_iter}_dfo{args.df_overshoot}_dfj{args.df_jitter}_dfproject_{args.df_project}"
        if args.init == "deepfool"
        else ""
    )

    return (
        f"{args.dataset}_{tag}_{args.init}_{idx_part}_k{args.total_iter}"
        f"_eps{args.epsilon}_a{args.alpha}_r{args.num_restarts}_seed{args.seed}"
        f"{df_part}"
    )


def format_title(args: argparse.Namespace) -> str:
    """Format title for the figure."""
    tag = get_model_tag(str(args.ckpt_dir))
    df_part = f", df_jitter={args.df_jitter}, df_project={args.df_project}" if args.init == "deepfool" else ""
    return f"{args.dataset.upper()} loss curves ({tag}, {args.init}-init{df_part})"
