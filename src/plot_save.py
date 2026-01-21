"""Save panel outputs (images, metadata, arrays)."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.dto import ExamplePanel
from src.logging_config import LOGGER


def save_panel_arrays(
    out_dir: str,
    base: str,
    panel_index: int,
    panel: ExamplePanel,
) -> None:
    """Save panel arrays (losses, preds, corrects) to arrays/ subdirectory."""
    arrays_dir = os.path.join(out_dir, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)

    np.save(os.path.join(arrays_dir, f"{base}_p{panel_index}_losses.npy"), panel.pgd.losses)
    np.save(os.path.join(arrays_dir, f"{base}_p{panel_index}_preds.npy"), panel.pgd.preds)
    np.save(
        os.path.join(arrays_dir, f"{base}_p{panel_index}_corrects.npy"),
        panel.pgd.corrects.astype(np.uint8),
    )


def save_panel_images(
    out_dir: str,
    base: str,
    dataset: str,
    panel_index: int,
    panel: ExamplePanel,
) -> None:
    """Save panel images (x_nat, x_adv, delta) to images/ subdirectory."""
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    nat_png = os.path.join(images_dir, f"{base}_p{panel_index}_x_nat.png")
    adv_png = os.path.join(images_dir, f"{base}_p{panel_index}_x_adv.png")
    delta_png = os.path.join(images_dir, f"{base}_p{panel_index}_delta_abs_norm.png")

    nat_img = np.squeeze(panel.x_nat).astype(np.float32)
    adv_img = np.squeeze(panel.x_adv_show).astype(np.float32)

    if dataset == "mnist":
        nat_img = nat_img.reshape(28, 28)
        adv_img = adv_img.reshape(28, 28)
        plt.imsave(nat_png, nat_img, cmap="gray", vmin=0.0, vmax=1.0)
        plt.imsave(adv_png, adv_img, cmap="gray", vmin=0.0, vmax=1.0)
        delta = np.abs(adv_img - nat_img)
        delta_vis = delta / (float(delta.max()) + 1e-12)
        plt.imsave(delta_png, delta_vis, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        nat_img = np.clip(nat_img.reshape(32, 32, 3), 0.0, 1.0)
        adv_img = np.clip(adv_img.reshape(32, 32, 3), 0.0, 1.0)
        plt.imsave(nat_png, nat_img, vmin=0.0, vmax=1.0)
        plt.imsave(adv_png, adv_img, vmin=0.0, vmax=1.0)
        delta = np.abs(adv_img - nat_img)
        delta_vis = delta / (float(delta.max()) + 1e-12)
        plt.imsave(delta_png, delta_vis, vmin=0.0, vmax=1.0)

    LOGGER.info(f"[save] panel={int(panel_index)} nat={nat_png} adv={adv_png}")


def format_panel_metadata(panel: ExamplePanel, panel_index: int, args: argparse.Namespace) -> str:
    """Format metadata for a single panel."""
    losses = panel.pgd.losses
    preds = panel.pgd.preds
    corrects = panel.pgd.corrects
    true_label = int(panel.y_nat.reshape(-1)[0])
    attack_success_rate = float(np.sum(~corrects[:, -1])) / float(corrects.shape[0])
    initial_losses = losses[:, 0]
    final_losses = losses[:, -1]

    nat_pred_line = f"nat_pred={panel.sanity.nat_pred}\n" if panel.sanity is not None else ""
    nat_loss_line = f"nat_loss={panel.sanity.nat_loss:.6f}\n" if panel.sanity is not None else ""

    content = (
        f"[PANEL_{panel_index}]\n"
        f"true_label={true_label}\n"
        f"restart_shown={panel.show_restart}\n"
        f"pred_end={panel.pred_end}\n"
        f"{nat_pred_line}final_preds={','.join(str(int(p)) for p in preds[:, -1])}\n"
        f"attack_success_rate={attack_success_rate:.6f}\n"
        f"{nat_loss_line}initial_loss_min={float(np.min(initial_losses)):.6f}\n"
        f"initial_loss_max={float(np.max(initial_losses)):.6f}\n"
        f"initial_loss_mean={float(np.mean(initial_losses)):.6f}\n"
        f"initial_loss_median={float(np.median(initial_losses)):.6f}\n"
        f"final_loss_min={float(np.min(final_losses)):.6f}\n"
        f"final_loss_max={float(np.max(final_losses)):.6f}\n"
        f"final_loss_mean={float(np.mean(final_losses)):.6f}\n"
        f"final_loss_median={float(np.median(final_losses)):.6f}\n"
    )

    if panel.sanity is not None and args.init == "deepfool":
        df_lines = ""
        if panel.sanity.df_pred is not None:
            df_lines += f"df_pred={panel.sanity.df_pred}\n"
        if panel.sanity.df_loss is not None:
            df_lines += f"df_loss={panel.sanity.df_loss:.6f}\n"
        if panel.sanity.linf_df is not None:
            df_lines += f"linf_df={panel.sanity.linf_df:.6f}\n"
        if panel.sanity.init_pred is not None:
            df_lines += f"init_pred={panel.sanity.init_pred}\n"
        if panel.sanity.init_loss is not None:
            df_lines += f"init_loss={panel.sanity.init_loss:.6f}\n"
        if panel.sanity.linf_init is not None:
            df_lines += f"linf_init={panel.sanity.linf_init:.6f}\n"
        content += df_lines

    return content


def save_panel_outputs(
    out_dir: str,
    base: str,
    dataset: str,
    panel_index: int,
    panel: ExamplePanel,
) -> None:
    """Save panel arrays and images to subdirectories."""
    save_panel_arrays(out_dir, base, panel_index, panel)
    save_panel_images(out_dir, base, dataset, panel_index, panel)
