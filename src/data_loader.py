"""Data loading utilities for MNIST and CIFAR-10."""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf


def load_mnist_flattened(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST test data as flattened float32 arrays."""
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    x_test = mnist.test.images.astype(np.float32)
    y_test = mnist.test.labels.astype(np.int64)
    return x_test, y_test


def load_cifar10_float01() -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test data normalized to [0, 1]."""
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.reshape(-1).astype(np.int64)
    return x_test, y_test


def load_test_data(
    dataset: str,
    model_src_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data based on dataset name."""
    if dataset == "mnist":
        data_dir = os.path.join(model_src_dir, "MNIST_data")
        return load_mnist_flattened(data_dir)
    return load_cifar10_float01()
