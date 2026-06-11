from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles, make_swiss_roll
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("csv/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(name, X, y):
    """
    Salva um dataset em dois formatos:
    1. CSV legível
    2. NPZ compacto para carregar depois em Python
    """

    # CSV
    columns = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["label"] = y
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    # NPZ
    np.savez_compressed(
        OUTPUT_DIR / f"{name}.npz",
        X=X,
        y=y
    )

    print(f"Salvo: {OUTPUT_DIR / f'{name}.csv'}")
    print(f"Salvo: {OUTPUT_DIR / f'{name}.npz'}")


# ------------------------------------------------------------
# Blobs com densidades diferentes
# ------------------------------------------------------------

X, y = make_blobs(
    n_samples=2000,
    centers=5,
    cluster_std=[0.35, 0.6, 0.9, 1.2, 1.5],
    random_state=42
)

X = StandardScaler().fit_transform(X)
save_dataset("blobs_var_density", X, y)


# ------------------------------------------------------------
# Moons
# ------------------------------------------------------------

X, y = make_moons(
    n_samples=2000,
    noise=0.06,
    random_state=42
)

X = StandardScaler().fit_transform(X)
save_dataset("moons", X, y)


# ------------------------------------------------------------
# Circles
# ------------------------------------------------------------

X, y = make_circles(
    n_samples=2000,
    noise=0.04,
    factor=0.45,
    random_state=42
)

X = StandardScaler().fit_transform(X)
save_dataset("circles", X, y)


# ------------------------------------------------------------
# Swiss roll
# ------------------------------------------------------------

X, t = make_swiss_roll(
    n_samples=3000,
    noise=0.05,
    random_state=42
)

X = StandardScaler().fit_transform(X)

# t é uma variável contínua. Para testes de clustering,
# podemos discretizar em faixas apenas como "label" auxiliar.
y = pd.qcut(t, q=6, labels=False)

save_dataset("swiss_roll", X, y)
