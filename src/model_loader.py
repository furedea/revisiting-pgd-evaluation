"""Model import, creation, and checkpoint utilities."""

import inspect
import importlib.util
import os
import sys
import types
from typing import Any

import tensorflow as tf

from src.logging_config import LOGGER


def add_sys_path(path: str) -> None:
    """Add path to sys.path if not already present."""
    if path not in sys.path:
        sys.path.insert(0, path)


def load_module_from_path(module_name: str, file_path: str) -> types.ModuleType:
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec: {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_module(model_src_dir: str, dataset: str) -> types.ModuleType:
    """Load the model.py module from the given directory."""
    add_sys_path(model_src_dir)
    model_py = os.path.join(model_src_dir, "model.py")
    if not os.path.exists(model_py):
        raise FileNotFoundError(f"model.py not found: {model_py}")
    return load_module_from_path(f"challenge_model_{dataset}", model_py)


def instantiate_model(model_module: Any, mode_default: str) -> Any:
    """Create model instance, passing mode if constructor accepts it."""
    if not hasattr(model_module, "Model"):
        raise AttributeError("model.py has no Model class.")
    model_cls = model_module.Model

    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.keys())
    return model_cls() if len(params) <= 1 else model_cls(mode_default)


def latest_ckpt(ckpt_dir: str) -> str:
    """Get the latest checkpoint path from a directory."""
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")
    return ckpt


def create_tf_session() -> tf.compat.v1.Session:
    """Create a TensorFlow session with GPU memory growth enabled."""
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=cfg)


def restore_checkpoint(
    sess: tf.compat.v1.Session,
    saver: tf.compat.v1.train.Saver,
    ckpt_dir: str,
) -> str:
    """Restore the latest checkpoint and return its path."""
    ckpt = latest_ckpt(ckpt_dir)
    saver.restore(sess, ckpt)
    LOGGER.info(f"[restore] {ckpt}")
    return ckpt
