#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${RTKNNC_OTHERS_PYTHON:-python}"
results_dir="${RTKNNC_OTHERS_RESULTS:-others/results}"

cd "${repository_root}"
"${python_bin}" -m others.benchmark --output-dir "${results_dir}" "$@"
"${python_bin}" -m others.make_comparison --external-results-dir "${results_dir}"
"${python_bin}" -m others.plot_benchmarks --results-dir "${results_dir}"
