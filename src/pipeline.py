"""Pipeline orchestration for PGD visualization."""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.cli import format_base_name, format_title, get_model_tag
from src.data_loader import load_test_data
from src.deepfool import build_deepfool_init
from src.dto import ExamplePanel, InitSanityMetrics, ModelOps
from src.logging_config import LOGGER
from src.math_utils import linf_distance
from src.model_loader import (
    create_tf_session,
    instantiate_model,
    load_model_module,
    restore_checkpoint,
)
from src.pgd import choose_show_restart, run_pgd_batch
from src.plot_panel import plot_panels
from src.plot_save import format_panel_metadata, save_panel_outputs


def print_clean_diagnostics(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
) -> None:
    """Print diagnostics for clean (natural) input."""
    pred = sess.run(ops.y_pred_op, feed_dict={ops.x_ph: x_nat, ops.y_ph: y_nat})
    loss_vec = sess.run(ops.per_ex_loss_op, feed_dict={ops.x_ph: x_nat, ops.y_ph: y_nat})

    true_label = int(y_nat.reshape(-1)[0])
    pred_label = int(pred[0])
    loss0 = float(loss_vec[0])

    LOGGER.info(f"[clean] true={true_label} pred={pred_label} loss={loss0:.6g}")


def log_init_sanity(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_nat: np.ndarray,
    y_nat: np.ndarray,
    x_df: Optional[np.ndarray],
    x_init: Optional[np.ndarray],
    eps: float,
) -> Optional[InitSanityMetrics]:
    """Check and log DeepFool init sanity metrics."""
    y0 = int(y_nat.reshape(-1)[0])

    def eval_point(x: np.ndarray) -> Tuple[int, float]:
        pred, loss_vec = sess.run(
            [ops.y_pred_op, ops.per_ex_loss_op],
            feed_dict={ops.x_ph: x, ops.y_ph: y_nat},
        )
        return int(pred[0]), float(loss_vec[0])

    nat_pred, nat_loss = eval_point(x_nat)

    df_pred: Optional[int] = None
    df_loss: Optional[float] = None
    linf_df: Optional[float] = None
    if x_df is not None:
        df_pred, df_loss = eval_point(x_df)
        linf_df = linf_distance(x_df, x_nat)

    init_pred: Optional[int] = None
    init_loss: Optional[float] = None
    linf_init: Optional[float] = None
    if x_init is not None:
        init_pred, init_loss = eval_point(x_init)
        linf_init = linf_distance(x_init, x_nat)

    LOGGER.info(
        "init_sanity true=%d | nat(pred=%d loss=%.6f) | "
        "df(pred=%s loss=%s linf_df=%.6f eps=%.6f) | "
        "init(pred=%s loss=%s linf_init=%s)",
        y0,
        nat_pred,
        nat_loss,
        "NA" if df_pred is None else str(df_pred),
        "NA" if df_loss is None else f"{df_loss:.6f}",
        -1.0 if linf_df is None else float(linf_df),
        float(eps),
        "NA" if init_pred is None else str(init_pred),
        "NA" if init_loss is None else f"{init_loss:.6f}",
        "NA" if linf_init is None else f"{float(linf_init):.6f}",
    )

    if x_df is None and x_init is None:
        return None

    return InitSanityMetrics(
        true_label=y0,
        nat_pred=nat_pred,
        nat_loss=nat_loss,
        df_pred=df_pred,
        linf_df=linf_df,
        df_loss=df_loss,
        init_pred=init_pred,
        linf_init=linf_init,
        init_loss=init_loss,
    )


def find_correct_indices(
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
    start_idx: int,
    max_tries: int,
) -> Tuple[int, ...]:
    """Find k correctly classified test examples starting from start_idx."""
    n = int(len(x_test))
    idx = int(start_idx)
    found_indices = []
    tries = 0

    while len(found_indices) < int(k) and tries < min(int(max_tries), n):
        x = x_test[idx : idx + 1]
        y = y_test[idx : idx + 1]
        pred = sess.run(ops.y_pred_op, feed_dict={ops.x_ph: x, ops.y_ph: y})
        if int(pred[0]) == int(y.reshape(-1)[0]):
            found_indices.append(idx)
        idx = (idx + 1) % n
        tries += 1

    if len(found_indices) < int(k):
        raise RuntimeError(f"Could not find {k} correct examples (found {len(found_indices)}).")

    return tuple(int(i) for i in found_indices)


def run_one_example(
    args: argparse.Namespace,
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    idx: int,
) -> ExamplePanel:
    """Run PGD attack on a single example and return panel data."""
    x_nat = x_test[idx : idx + 1].astype(np.float32)
    y_nat = y_test[idx : idx + 1].astype(np.int64)

    print_clean_diagnostics(sess, ops, x_nat, y_nat)

    x_df = None
    x_init = None
    init_jitter = 0.0

    if args.init == "deepfool":
        x_df, x_init = build_deepfool_init(
            sess=sess,
            ops=ops,
            x_nat=x_nat,
            y_nat=y_nat,
            df_max_iter=int(args.df_max_iter),
            df_overshoot=float(args.df_overshoot),
            df_project=str(getattr(args, "df_project", "clip")),
            eps=float(args.epsilon),
        )
        init_jitter = float(args.df_jitter)

    sanity = log_init_sanity(
        sess=sess,
        ops=ops,
        x_nat=x_nat,
        y_nat=y_nat,
        x_df=x_df,
        x_init=x_init,
        eps=float(args.epsilon),
    )

    pgd = run_pgd_batch(
        sess=sess,
        ops=ops,
        x_nat=x_nat,
        y_nat=y_nat,
        eps=float(args.epsilon),
        alpha=float(args.alpha),
        total_iter=int(args.total_iter),
        num_restarts=int(args.num_restarts),
        seed=int(args.seed),
        init=str(args.init),
        x_init=x_init,
        init_jitter=init_jitter,
    )

    show_restart = choose_show_restart(pgd)
    x_adv_show = pgd.x_adv_final[show_restart : show_restart + 1].astype(np.float32)
    pred_end = int(pgd.preds[show_restart, -1])

    wrong_end = int(np.sum(~pgd.corrects[:, -1]))
    loss_end = pgd.losses[:, -1]
    LOGGER.info(
        f"[pgd] wrong_end={wrong_end}/{int(args.num_restarts)} "
        f"loss_end(med/min/max)={float(np.median(loss_end)):.4g}/"
        f"{float(np.min(loss_end)):.4g}/{float(np.max(loss_end)):.4g}"
    )

    return ExamplePanel(
        x_nat=x_nat,
        y_nat=y_nat,
        x_adv_show=x_adv_show,
        show_restart=int(show_restart),
        pred_end=int(pred_end),
        pgd=pgd,
        sanity=sanity,
    )


def run_all_examples(
    args: argparse.Namespace,
    sess: tf.compat.v1.Session,
    ops: ModelOps,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: Tuple[int, ...],
) -> Tuple[ExamplePanel, ...]:
    """Run PGD attack on all selected examples."""
    panels = []
    for idx in indices:
        panels.append(run_one_example(args, sess, ops, x_test, y_test, int(idx)))
    return tuple(panels)


def save_all_outputs(
    args: argparse.Namespace,
    base: str,
    panels: Tuple[ExamplePanel, ...],
) -> None:
    """Save all panel outputs (arrays, images, metadata) to subdirectories."""
    # Save arrays and images for each panel
    for i, panel in enumerate(panels, start=1):
        save_panel_outputs(
            out_dir=str(args.out_dir),
            base=str(base),
            dataset=str(args.dataset),
            panel_index=int(i),
            panel=panel,
        )

    # Save unified metadata file
    metadata_dir = os.path.join(args.out_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_txt = os.path.join(metadata_dir, f"{base}_meta.txt")

    df_params = ""
    if args.init == "deepfool":
        df_params = (
            f"df_max_iter={args.df_max_iter}\n"
            f"df_overshoot={args.df_overshoot}\n"
            f"df_jitter={args.df_jitter}\n"
            f"df_project={args.df_project}\n"
        )

    content = (
        "[PARAMETERS]\n"
        f"dataset={args.dataset}\n"
        f"epsilon={args.epsilon}\n"
        f"alpha={args.alpha}\n"
        f"total_iter={args.total_iter}\n"
        f"num_restarts={args.num_restarts}\n"
        f"seed={args.seed}\n"
        f"init={args.init}\n"
        f"{df_params}\n"
    )

    for i, panel in enumerate(panels, start=1):
        content += format_panel_metadata(panel, i, args)

    with open(meta_txt, "w", encoding="utf-8") as f:
        f.write(content)

    LOGGER.info(f"[save] metadata={meta_txt}")


def render_figure(
    args: argparse.Namespace,
    base: str,
    title: str,
    panels: Tuple[ExamplePanel, ...],
) -> str:
    """Render and save the figure to figures/ subdirectory."""
    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_png = os.path.join(figures_dir, f"{base}.png")
    plot_panels(
        dataset=str(args.dataset),
        panels=panels,
        out_png=out_png,
        title=str(title),
        init_sanity_plot=bool(args.init_sanity_plot) and (str(args.init) == "deepfool"),
        eps=float(args.epsilon),
    )
    return out_png


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full PGD visualization pipeline."""
    os.makedirs(args.out_dir, exist_ok=True)

    model_src_dir = os.path.abspath(args.model_src_dir)

    tf.compat.v1.reset_default_graph()
    model_module = load_model_module(model_src_dir, str(args.dataset))
    model = instantiate_model(model_module, mode_default="eval")
    ops = ModelOps.from_model(model)

    saver = tf.compat.v1.train.Saver()

    with create_tf_session() as sess:
        restore_checkpoint(sess, saver, str(args.ckpt_dir))
        x_test, y_test = load_test_data(str(args.dataset), model_src_dir)

        indices = find_correct_indices(
            sess=sess,
            ops=ops,
            x_test=x_test,
            y_test=y_test,
            k=int(args.n_examples),
            start_idx=int(args.start_idx),
            max_tries=int(args.max_tries),
        )
        base = format_base_name(args, indices)
        title = format_title(args)
        tag = get_model_tag(str(args.ckpt_dir))

        df_part = (
            f" df_iter={args.df_max_iter} df_overshoot={args.df_overshoot} df_jitter={args.df_jitter}"
            if args.init == "deepfool"
            else ""
        )
        LOGGER.info(
            f"[run] dataset={args.dataset} indices={indices} tag={tag} init={args.init}"
            f" eps={args.epsilon} alpha={args.alpha} total_iter={args.total_iter} restarts={args.num_restarts}"
            f" seed={args.seed}{df_part} out={args.out_dir}"
        )

        panels = run_all_examples(args, sess, ops, x_test, y_test, indices)

    save_all_outputs(args, base, panels)
    out_png = render_figure(args, base, title, panels)

    LOGGER.info(f"[done] figure={out_png}")
