#!/usr/bin/env python3
import ctypes
import os
import subprocess
import sys

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH    = os.path.join(PROJECT_DIR, "src", "cannon_knn", "_clib", "libcannon.so")
CSV_DIR     = os.path.join(PROJECT_DIR, "csv", "synthetic")


def run_pipeline(csv_path, K=16, K0=4):
    result = subprocess.run(
        [sys.executable, __file__, "--standalone",
         "--input", csv_path, "--K", str(K), "--K0", str(K0)],
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

    fake_argv = [b"cannon-knn", b"--input", args.input.encode(),
                 b"--K", str(args.K).encode(), b"--K0", str(args.K0).encode()]
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
    assert os.path.exists(LIB_PATH), f"libcannon.so nao encontrada em {LIB_PATH}"


@pytest.mark.parametrize("filename,expected", [
    ("blobs_var_density.csv", 5),
    ("circles.csv",           2),
    ("moons.csv",             2),
    ("moons-knn.csv",         1),
    ("swiss_roll.csv",        6),
])
def test_hierarchical_clustering(filename, expected):
    clusters = run_pipeline(os.path.join(CSV_DIR, filename))
    assert clusters == expected, f"{filename}: esperado {expected}, obtido {clusters}"


def test_grade_equals_2_pow_dimension():
    lib = ctypes.CDLL(LIB_PATH)
    lib.cmdline_parser.restype  = ctypes.c_void_p
    lib.cmdline_parser.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.getdata.restype         = ctypes.c_size_t
    lib.getdata.argtypes        = [ctypes.c_void_p]
    csv_path  = os.path.join(CSV_DIR, "moons.csv")
    fake_argv = [b"cannon-knn", b"--input", csv_path.encode(), b"--K", b"5", b"--K0", b"1"]
    argc      = len(fake_argv)
    argv_c    = (ctypes.c_char_p * argc)(*fake_argv)
    lib.getdata(lib.cmdline_parser(argc, argv_c))
    dim   = ctypes.c_size_t.in_dll(lib, "dimension").value
    grade = ctypes.c_size_t.in_dll(lib, "grade").value
    assert grade == (1 << dim), f"grade={grade} mas 2^dim={1<<dim}"
