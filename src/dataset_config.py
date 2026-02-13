"""Centralized dataset-specific parameter management."""


class DatasetConfig:
    """Dataset-specific parameters."""

    __slots__ = ("epsilon", "alpha", "model_src_dir")

    def __init__(
        self,
        epsilon: float,
        alpha: float,
        model_src_dir: str,
    ) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.model_src_dir = model_src_dir


# Dataset registry: dataset name -> DatasetConfig
_DATASET_REGISTRY = {
    "mnist": DatasetConfig(
        epsilon=0.3,
        alpha=0.01,
        model_src_dir="model_src/mnist_challenge",
    ),
    "cifar10": DatasetConfig(
        epsilon=8.0 / 255.0,
        alpha=2.0 / 255.0,
        model_src_dir="model_src/cifar10_challenge",
    ),
}


def resolve_dataset_config(dataset: str) -> DatasetConfig:
    """Resolve dataset-specific parameters.

    Args:
        dataset: "mnist" or "cifar10"

    Returns:
        DatasetConfig with epsilon, alpha, model_src_dir

    Raises:
        ValueError: if dataset is unknown
    """
    if dataset not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset!r}. "
            f"Available datasets: {sorted(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[dataset]
