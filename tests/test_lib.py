#!/usr/bin/env python3
import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from rtknnc.layer_io import read_layer_file

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH    = os.path.join(PROJECT_DIR, "src", "rtknnc", "_clib", "librtknnc.so")
CSV_DIR     = os.path.join(PROJECT_DIR, "csv", "synthetic")


def run_pipeline(csv_path, K=16, K0=2, dimension=None,
                 structural_pruning=False, vote_kernel=None,
                 sigmoid_alpha=8.0):
    command = [
        sys.executable, __file__, "--standalone",
        "--input", csv_path, "--K", str(K), "--K0", str(K0),
    ]
    if dimension is not None:
        command.extend(["--dim", str(dimension)])
    if vote_kernel is not None:
        command.extend(["--vote-kernel", vote_kernel])
    if vote_kernel == "sigmoid":
        command.extend(["--sigmoid-alpha", str(sigmoid_alpha)])
    if structural_pruning:
        command.append("--structural-pruning")

    result = subprocess.run(
        command,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline falhou:\n{result.stderr}")
    for line in result.stdout.splitlines():
        if line.startswith("CLCOUNT:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError(f"CLCOUNT nao encontrado:\n{result.stdout}")


def _standalone():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--K",  type=int, default=16)
    parser.add_argument("--K0", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument(
        "--vote-kernel",
        choices=("uniform", "inv_r", "inv_r2", "sigmoid"),
        default=None,
    )
    parser.add_argument("--sigmoid-alpha", type=float, default=8.0)
    parser.add_argument("--structural-pruning", action="store_true")
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    if args.K0 is None:
        args.K0 = args.K
    lib = ctypes.CDLL(LIB_PATH)
    lib.cmdline_parser.restype  = ctypes.c_void_p
    lib.cmdline_parser.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.getdata.restype         = ctypes.c_size_t
    lib.getdata.argtypes        = [ctypes.c_void_p]
    lib.hyperquadtree_build.restype  = None
    lib.hyperquadtree_build.argtypes = [ctypes.c_size_t]
    lib.knn_search.restype  = None
    lib.knn_search.argtypes = [ctypes.c_size_t]
    lib.roundtrip_KNN_cluster_partition.restype  = None
    lib.roundtrip_KNN_cluster_partition.argtypes = [ctypes.c_size_t]
    lib.hyperquadtree_destroy.restype  = None
    lib.hyperquadtree_destroy.argtypes = []

    fake_argv = [b"rtknnc", b"--input", args.input.encode(),
                 b"--K", str(args.K).encode(), b"--K0", str(args.K0).encode()]
    if args.dim is not None:
        fake_argv.extend([b"--dim", str(args.dim).encode()])
    if args.vote_kernel is not None:
        fake_argv.extend([b"--vote-kernel", args.vote_kernel.encode()])
    if args.vote_kernel == "sigmoid":
        fake_argv.extend([b"--sigmoid-alpha", str(args.sigmoid_alpha).encode()])
    if args.structural_pruning:
        fake_argv.append(b"--structural-pruning")
    argc   = len(fake_argv)
    argv_c = (ctypes.c_char_p * argc)(*fake_argv)

    file_ptr = lib.cmdline_parser(argc, argv_c)
    assert file_ptr
    lib.getdata(file_ptr)

    c_n     = ctypes.c_size_t.in_dll(lib, "n")
    c_grade = ctypes.c_size_t.in_dll(lib, "grade")
    assert c_n.value > 0
    assert c_grade.value > 0

    lib.hyperquadtree_build(c_n.value)
    lib.knn_search(c_n.value)
    lib.roundtrip_KNN_cluster_partition(c_n.value)
    lib.hyperquadtree_destroy()

    print(f"CLCOUNT: {ctypes.c_size_t.in_dll(lib, 'clcount').value}")
    sys.exit(0)


if "--standalone" in sys.argv:
    _standalone()


@pytest.fixture(scope="session", autouse=True)
def lib_exists():
    assert os.path.exists(LIB_PATH), f"librtknnc.so nao encontrada em {LIB_PATH}"


@pytest.mark.parametrize("filename,expected", [
    ("blobs_var_density.csv", 5),
    ("circles.csv",           2),
    ("moons.csv",             2),
    ("swiss_roll.csv",        6),
    ("bridge.csv",            3),
])
def test_hierarchical_clustering(filename, expected):
    clusters = run_pipeline(os.path.join(CSV_DIR, filename))
    assert clusters == expected, f"{filename}: esperado {expected}, obtido {clusters}"


@pytest.mark.parametrize("filename,k,expected,min_ari", [
    ("blobs_var_density.csv", 8, 5, 0.96),
    ("bridge.csv",            6, 3, 0.98),
    ("bridge.csv",           10, 3, 0.98),
    ("circles.csv",           7, 2, 0.999),
    ("moons.csv",            10, 2, 0.999),
    ("anisotropic.csv",       8, 3, 0.999),
])
def test_native_structural_pruning(tmp_path, filename, k, expected, min_ari):
    source = os.path.join(CSV_DIR, filename)
    isolated = tmp_path / filename
    shutil.copyfile(source, isolated)

    clusters = run_pipeline(
        str(isolated), K=k, K0=k, dimension=2,
        structural_pruning=True,
    )
    assert clusters == expected

    layer = next(tmp_path.glob(f"{isolated.stem}_K{k}Cs*_layer.csv"))
    rows = read_layer_file(layer)
    table = np.genfromtxt(isolated, delimiter=",", names=True)
    reference = np.asarray(table["label"], dtype=int)
    labels = np.full(len(reference), -1, dtype=int)
    for row in rows:
        labels[row.point_id - 1] = row.cluster_id - 1
    assert adjusted_rand_score(reference, labels) >= min_ari


@pytest.mark.parametrize("filename,k,vote_kernel,expected,min_ari", [
    ("blobs_var_density.csv", 8, "inv_r",   5, 0.96),
    ("blobs_var_density.csv", 8, "inv_r2",  5, 0.96),
    ("blobs_var_density.csv", 8, "sigmoid", 5, 0.96),
    ("bridge.csv",            6, "inv_r",   3, 0.98),
    ("bridge.csv",            6, "inv_r2",  3, 0.98),
    ("bridge.csv",            6, "sigmoid", 3, 0.98),
])
def test_native_structural_pruning_vote_kernels(
    tmp_path, filename, k, vote_kernel, expected, min_ari
):
    source = os.path.join(CSV_DIR, filename)
    isolated = tmp_path / filename
    shutil.copyfile(source, isolated)

    clusters = run_pipeline(
        str(isolated), K=k, K0=k, dimension=2,
        structural_pruning=True, vote_kernel=vote_kernel,
    )
    assert clusters == expected

    layer = next(tmp_path.glob(f"{isolated.stem}_K{k}Cs*_layer.csv"))
    rows = read_layer_file(layer)
    table = np.genfromtxt(isolated, delimiter=",", names=True)
    reference = np.asarray(table["label"], dtype=int)
    labels = np.full(len(reference), -1, dtype=int)
    for row in rows:
        labels[row.point_id - 1] = row.cluster_id - 1
    assert adjusted_rand_score(reference, labels) >= min_ari


def test_grade_equals_2_pow_dimension():
    lib = ctypes.CDLL(LIB_PATH)
    lib.cmdline_parser.restype  = ctypes.c_void_p
    lib.cmdline_parser.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.getdata.restype         = ctypes.c_size_t
    lib.getdata.argtypes        = [ctypes.c_void_p]
    csv_path  = os.path.join(CSV_DIR, "moons.csv")
    fake_argv = [b"rtknnc", b"--input", csv_path.encode(), b"--K", b"5", b"--K0", b"1"]
    argc      = len(fake_argv)
    argv_c    = (ctypes.c_char_p * argc)(*fake_argv)
    lib.getdata(lib.cmdline_parser(argc, argv_c))
    dim   = ctypes.c_size_t.in_dll(lib, "dimension").value
    grade = ctypes.c_size_t.in_dll(lib, "grade").value
    assert grade == (1 << dim), f"grade={grade} mas 2^dim={1<<dim}"

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
