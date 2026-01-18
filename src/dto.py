"""Data Transfer Objects for PGD visualization (immutable by convention)."""

from typing import Any, Optional

import numpy as np


class ModelOps:
    """Graph ops required for prediction/loss/grad and DeepFool."""

    __slots__ = (
        "x_ph",
        "y_ph",
        "logits",
        "logits_name",
        "y_pred_op",
        "per_ex_loss_op",
        "grad_op",
        "grads_all_op",
    )

    def __init__(
        self,
        x_ph: Any,
        y_ph: Any,
        logits: Any,
        logits_name: str,
        y_pred_op: Any,
        per_ex_loss_op: Any,
        grad_op: Any,
        grads_all_op: Any,
    ) -> None:
        self.x_ph = x_ph
        self.y_ph = y_ph
        self.logits = logits
        self.logits_name = logits_name
        self.y_pred_op = y_pred_op
        self.per_ex_loss_op = per_ex_loss_op
        self.grad_op = grad_op
        self.grads_all_op = grads_all_op

    @classmethod
    def from_model(cls, model: Any) -> "ModelOps":
        # Lazy import: TF 1.15.5 crashes on ARM (AVX required) at import time.
        # Deferring import allows testing non-TF parts on Apple Silicon.
        import tensorflow as tf

        x_ph = model.x_input
        y_ph = model.y_input

        if hasattr(model, "pre_softmax"):
            logits = model.pre_softmax
            logits_name = "pre_softmax"
        elif hasattr(model, "logits"):
            logits = model.logits
            logits_name = "logits"
        else:
            raise AttributeError("Model has neither 'pre_softmax' nor 'logits'.")

        y_pred_op = tf.argmax(logits, axis=1, output_type=tf.int64)

        per_ex_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_ph,
            logits=logits,
        )
        loss_sum = tf.reduce_sum(per_ex_loss_op)
        grad_op = tf.gradients(loss_sum, x_ph)[0]

        num_classes = int(logits.shape[-1])
        grads_k = []
        for k in range(num_classes):
            grads_k.append(tf.gradients(logits[:, k], x_ph)[0])
        grads_all_op = tf.stack(grads_k, axis=1)

        return cls(
            x_ph=x_ph,
            y_ph=y_ph,
            logits=logits,
            logits_name=logits_name,
            y_pred_op=y_pred_op,
            per_ex_loss_op=per_ex_loss_op,
            grad_op=grad_op,
            grads_all_op=grads_all_op,
        )


class InitSanityMetrics:
    """Metrics for DeepFool initialization sanity check."""

    __slots__ = (
        "true_label",
        "nat_pred",
        "nat_loss",
        "df_pred",
        "linf_df",
        "df_loss",
        "init_pred",
        "linf_init",
        "init_loss",
    )

    def __init__(
        self,
        true_label: int,
        nat_pred: int,
        nat_loss: float,
        df_pred: Optional[int],
        linf_df: Optional[float],
        df_loss: Optional[float],
        init_pred: Optional[int],
        linf_init: Optional[float],
        init_loss: Optional[float],
    ) -> None:
        self.true_label = int(true_label)
        self.nat_pred = int(nat_pred)
        self.nat_loss = float(nat_loss)
        self.df_pred = None if df_pred is None else int(df_pred)
        self.linf_df = None if linf_df is None else float(linf_df)
        self.df_loss = None if df_loss is None else float(df_loss)
        self.init_pred = None if init_pred is None else int(init_pred)
        self.linf_init = None if linf_init is None else float(linf_init)
        self.init_loss = None if init_loss is None else float(init_loss)


class PGDBatchResult:
    """Batched PGD result for one example (R restarts)."""

    __slots__ = ("losses", "preds", "corrects", "x_adv_final")

    def __init__(
        self,
        losses: np.ndarray,
        preds: np.ndarray,
        corrects: np.ndarray,
        x_adv_final: np.ndarray,
    ) -> None:
        self.losses = losses
        self.preds = preds
        self.corrects = corrects
        self.x_adv_final = x_adv_final


class ExamplePanel:
    """All info needed to render one panel (one test example)."""

    __slots__ = (
        "x_nat",
        "y_nat",
        "x_adv_show",
        "show_restart",
        "pred_end",
        "pgd",
        "sanity",
    )

    def __init__(
        self,
        x_nat: np.ndarray,
        y_nat: np.ndarray,
        x_adv_show: np.ndarray,
        show_restart: int,
        pred_end: int,
        pgd: PGDBatchResult,
        sanity: Optional[InitSanityMetrics],
    ) -> None:
        self.x_nat = x_nat
        self.y_nat = y_nat
        self.x_adv_show = x_adv_show
        self.show_restart = show_restart
        self.pred_end = pred_end
        self.pgd = pgd
        self.sanity = sanity
