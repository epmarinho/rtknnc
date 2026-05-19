import numpy as np
from cannon_knn import z_normalize, z_denormalize

def test_zero_mean():
    X_norm, *_ = z_normalize(np.random.default_rng(1).standard_normal((50, 4)))
    assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-10)

def test_unit_std():
    X_norm, *_ = z_normalize(np.random.default_rng(2).standard_normal((50, 4)))
    assert np.allclose(X_norm.std(axis=0), 1, atol=1e-10)

def test_roundtrip():
    X = np.random.default_rng(3).standard_normal((30, 5))
    X_norm, mean, std = z_normalize(X)
    assert np.allclose(z_denormalize(X_norm, mean, std), X, atol=1e-10)