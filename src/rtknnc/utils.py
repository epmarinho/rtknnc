"""
utils.py — Utility functions for rtknnc.
"""
import numpy as np


def z_normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise each feature dimension (mirrors z-norm.c).

    Returns
    -------
    X_norm : (N, D) normalised array
    mean   : (D,) per-column mean
    std    : (D,) per-column std  (0-std columns left unchanged)
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std, mean, std


def z_denormalize(X_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse of z_normalize."""
    return X_norm * std + mean


def check_array(X, name: str = "X") -> np.ndarray:
    """Validate and convert to float64 C-contiguous 2-D array."""
    X = np.asarray(X, dtype=np.float64, order="C")
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {X.shape}")
    return X
