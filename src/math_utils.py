"""Math utilities for L-infinity distance and projection."""

import numpy as np


def linf_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute L-infinity distance between two arrays."""
    return float(np.max(np.abs(x1.astype(np.float32) - x2.astype(np.float32))))


def project_linf(x: np.ndarray, x_nat: np.ndarray, eps: float) -> np.ndarray:
    """Project x onto L-infinity ball centered at x_nat with radius eps."""
    return np.clip(x, x_nat - eps, x_nat + eps).astype(np.float32)


def clip_to_unit_interval(x: np.ndarray) -> np.ndarray:
    """Clip array values to [0, 1]."""
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def scale_to_linf_ball(x: np.ndarray, x_nat: np.ndarray, eps: float) -> np.ndarray:
    """Scale delta so that ||x - x_nat||_inf <= eps, preserving direction."""
    delta = x.astype(np.float32) - x_nat.astype(np.float32)
    linf = float(np.max(np.abs(delta)))
    if linf <= float(eps) + 1e-12:
        return x.astype(np.float32)
    s = float(eps) / max(1e-12, linf)
    return (x_nat.astype(np.float32) + s * delta).astype(np.float32)
