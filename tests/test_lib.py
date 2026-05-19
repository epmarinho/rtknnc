#!/usr/bin/env python3
import ctypes, sys, os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH   = os.path.join(SCRIPT_DIR, "src", "cannon_knn", "_clib", "libcannon.so")

if not os.path.exists(LIB_PATH):
    sys.exit(f"Erro: lib nao encontrada em {LIB_PATH}")

lib = ctypes.CDLL(LIB_PATH)
print(f"Lib carregada: {LIB_PATH}")

lib.cmdline_parser.restype  = ctypes.c_void_p
lib.cmdline_parser.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
lib.getdata.restype  = ctypes.c_size_t
lib.getdata.argtypes = [ctypes.c_void_p]
lib.hyperquadtree_build.restype  = None
lib.hyperquadtree_build.argtypes = [ctypes.c_size_t]
lib.knn_search.restype  = None
lib.knn_search.argtypes = [ctypes.c_size_t]
lib.roundtrip_KNN_cluster_partition.restype  = None
lib.roundtrip_KNN_cluster_partition.argtypes = [ctypes.c_size_t]
lib.hyperquadtree_destroy.restype  = None
lib.hyperquadtree_destroy.argtypes = []

fake_argv = [b"cannon-knn"] + [a.encode() for a in sys.argv[1:]]

if b"--input" not in fake_argv:
    sys.exit("Uso: python test_lib.py --input <arquivo.csv> [--K 3] [--K0 1] [--dim N]")

argc = len(fake_argv)
argv_c = (ctypes.c_char_p * argc)(*fake_argv)

print("[1/5] cmdline_parser...")
file_ptr = lib.cmdline_parser(argc, argv_c)
if not file_ptr:
    sys.exit("Erro: cmdline_parser nao abriu o arquivo.")

print("[2/5] getdata...")
n = lib.getdata(file_ptr)

c_n         = ctypes.c_size_t.in_dll(lib, "n")
c_dimension = ctypes.c_size_t.in_dll(lib, "dimension")
c_K         = ctypes.c_size_t.in_dll(lib, "K")
c_K0        = ctypes.c_size_t.in_dll(lib, "K0")
c_grade     = ctypes.c_size_t.in_dll(lib, "grade")
print(f"      n={c_n.value}  dim={c_dimension.value}  K={c_K.value}  K0={c_K0.value}  grade={c_grade.value}")

print(f"[3/5] hyperquadtree_build...")
lib.hyperquadtree_build(c_n.value)

print(f"[4/5] knn_search...")
lib.knn_search(c_n.value)

print("[5/5] roundtrip_KNN_cluster_partition...")
lib.roundtrip_KNN_cluster_partition(c_n.value)
lib.hyperquadtree_destroy()

try:
    clcount = ctypes.c_size_t.in_dll(lib, "clcount")
    print(f"\nClusters encontrados: {clcount.value}")
except Exception:
    print("\nPipeline executado com sucesso.")
