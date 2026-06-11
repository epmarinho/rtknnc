#!/usr/bin/env python3
import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_blobs


OUTDIR = Path("/tmp/rtknnc_blob_search")
OUTDIR.mkdir(parents=True, exist_ok=True)


def write_legacy_csv(path, X, y):
    df = pd.DataFrame({
        "x0": X[:, 0],
        "x1": X[:, 1],
        "label": y,
    })
    df.to_csv(path, index=False)


def run_rtknnc(csv_path, K=16, K0=4):
    cmd = [
        sys.executable,
        "tests/test_lib.py",
        "--standalone",
        "--input",
        str(csv_path),
        "--K",
        str(K),
        "--K0",
        str(K0),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None, result.stderr

    for line in result.stdout.splitlines():
        if line.startswith("CLCOUNT:"):
            return int(line.split(":")[1].strip()), result.stdout

    return None, result.stdout


def main():
    n_samples_list = [2000]
    seeds = range(0, 200)

    center_sets = [
        [(-6, 0), (-3, 0), (0, 0), (3, 0), (6, 0)],
        [(-4, -2), (-4, 2), (0, 0), (4, -2), (4, 2)],
        [(-5, 0), (-2, 1.5), (0, -1.5), (2, 1.5), (5, 0)],
        [(-5, -2), (-5, 2), (0, 0), (5, -2), (5, 2)],
    ]

    std_sets = [
        [0.35, 0.35, 0.35, 0.35, 0.35],
        [0.45, 0.45, 0.45, 0.45, 0.45],
        [0.60, 0.60, 0.60, 0.60, 0.60],
        [0.35, 0.50, 0.70, 0.50, 0.35],
        [0.30, 0.45, 0.80, 0.45, 0.30],
        [0.25, 0.40, 0.70, 1.00, 1.30],
    ]

    for n_samples, seed, centers, stds in itertools.product(
        n_samples_list,
        seeds,
        center_sets,
        std_sets,
    ):
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=stds,
            random_state=seed,
        )

        csv_path = OUTDIR / f"blobs5_seed{seed}.csv"
        write_legacy_csv(csv_path, X, y)

        clcount, output = run_rtknnc(csv_path, K=16, K0=4)

        print(
            f"seed={seed}, centers={centers}, stds={stds} -> CLCOUNT={clcount}"
        )

        if clcount == 5:
            final_path = Path("csv/synthetic/blobs_var_density.csv")
            final_path.write_text(csv_path.read_text(), encoding="utf-8")

            print()
            print("ACHOU DATASET COM 5 CLUSTERS")
            print(f"seed: {seed}")
            print(f"centers: {centers}")
            print(f"cluster_std: {stds}")
            print(f"salvo em: {final_path}")
            return

    raise SystemExit("Nenhuma combinação testada produziu CLCOUNT=5.")


if __name__ == "__main__":
    main()
