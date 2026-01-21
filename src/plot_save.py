"""Save panel outputs (images, metadata, arrays)."""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.dto import ExamplePanel
from src.logging_config import LOGGER


def save_panel_outputs(
    out_dir: str,
    base: str,
    dataset: str,
    panel_index: int,
    panel: ExamplePanel,
) -> None:
    """Save panel images, metadata, and arrays to files."""
    os.makedirs(out_dir, exist_ok=True)

    nat_png = os.path.join(out_dir, f"{base}_p{panel_index}_x_nat.png")
    adv_png = os.path.join(out_dir, f"{base}_p{panel_index}_x_adv.png")
    delta_png = os.path.join(out_dir, f"{base}_p{panel_index}_delta_abs_norm.png")
    meta_txt = os.path.join(out_dir, f"{base}_p{panel_index}_meta.txt")

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

    with open(meta_txt, "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset}\n")
        f.write(f"panel={panel_index}\n")
        f.write(f"restart_shown={panel.show_restart}\n")
        f.write(f"pred_end={panel.pred_end}\n")
        f.write(f"x_nat_saved={nat_png}\n")
        f.write(f"x_adv_saved={adv_png}\n")
        f.write(f"delta_saved={delta_png}\n")

    np.save(os.path.join(out_dir, f"{base}_p{panel_index}_losses.npy"), panel.pgd.losses)
    np.save(os.path.join(out_dir, f"{base}_p{panel_index}_preds.npy"), panel.pgd.preds)
    np.save(
        os.path.join(out_dir, f"{base}_p{panel_index}_corrects.npy"),
        panel.pgd.corrects.astype(np.uint8),
    )

    LOGGER.info(f"[save] panel={int(panel_index)} nat={nat_png} adv={adv_png}")
