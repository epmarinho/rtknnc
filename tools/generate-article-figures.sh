#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash tools/generate-article-figures.sh
  bash tools/generate-article-figures.sh --help

Regenerate the complete figure and benchmark package used by
paper/round_trip_knn_nca_sn_vote_models.tex.

Controlled vote-kernel comparison:
  uniform   g(r) = 1
  inv_r     g(r) = 1/r
  inv_r2    g(r) = 1/r^2  (historical model)
  sigmoid   S_{alpha,R}(r), alpha = 8

The broader benchmark contains all four kernels without pruning.  The moons
figure shows that comparison.  Swiss-roll renderings and the bridge-fraction
experiment remain supporting artifacts generated with the unpruned historical
inv_r2 kernel, but are not manuscript figures.  The two critical native-pruning
figures are separate three-panel mosaics: inv_r, inv_r2, and sigmoid on the
variable-density blobs at K=8 and sparse bridge at K=6.
Archived conceptual PDFs are copied, while the monochrome leadership graph and
corrected rooted forest are rendered from their included Graphviz sources.

The pipeline:
  1. records SHA-256 hashes for the eight input datasets;
  2. runs the mandatory 8-dataset x 15-K C--Python inv_r2 regression;
  3. regenerates and validates the 840-row four-kernel benchmark;
  4. writes Matplotlib and GNUplot PNG/PDF figures and layer files;
  5. runs native baseline/pruned and three-kernel pruning comparisons;
  6. packages the archived figures and renders the two Graphviz reconstructions.

Requirements:
  - run from anywhere inside the repository checkout;
  - an executable ./rtknnc built at the repository root;
  - Python dependencies available in the active environment;
  - GNUplot, Graphviz (dot and neato), and pdftoppm available on PATH.

Recommended invocation from the repository root:
  conda run -n aniso-gnn bash tools/generate-article-figures.sh

Output:
  paper/figures/article_20260826/

Existing benchmark, layer, and figure files in that package are overwritten.
The script does not modify C source files.  The 840-row vote-kernel experiment
remains unpruned; structural pruning is evaluated separately so the two effects
are not confounded.
EOF
}

case "${1:-}" in
    "")
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        printf 'Unknown argument: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$workspace/paper/figures/article_20260826"
input_stage="$(mktemp -d /tmp/rtknnc-paper-inputs.XXXXXX)"
mpl_cache="$(mktemp -d /tmp/rtknnc-paper-mpl.XXXXXX)"

cleanup() {
    rm -rf -- "$input_stage" "$mpl_cache"
}
trap cleanup EXIT

datasets=(
    anisotropic
    blobs
    blobs_var_density
    bridge
    circles
    moons
    swiss_roll
    uniform
)

# The controlled comparison has four kernels.  The historical inverse-square
# kernel is also used alone in supporting Swiss-roll and bridge-fraction
# diagnostics.
vote_models=(uniform inv_r inv_r2 sigmoid)
historical_vote_model=inv_r2

input_manifest="$input_stage/input_sha256.txt"
: > "$input_manifest"

for dataset in "${datasets[@]}"; do
    source_csv="$workspace/csv/synthetic/$dataset.csv"
    if [[ "$dataset" == "blobs" && ! -f "$source_csv" ]]; then
        source_csv="$workspace/csv/benchmarks/blobs.csv"
    fi
    if [[ ! -f "$source_csv" ]]; then
        echo "Missing article dataset: $source_csv" >&2
        exit 1
    fi
    ln -s "$source_csv" "$input_stage/$dataset.csv"
    source_relative="${source_csv#"$workspace/"}"
    source_hash="$(sha256sum "$source_csv" | awk '{print $1}')"
    printf '%s  %s\n' "$source_hash" "$source_relative" >> "$input_manifest"
done

if [[ ! -x "$workspace/rtknnc" ]]; then
    echo "Missing executable: $workspace/rtknnc" >&2
    echo "Build it first with: make rtknnc" >&2
    exit 1
fi

export PYTHONPATH="$workspace/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$mpl_cache"

mkdir -p "$output_dir/data"
cp "$input_manifest" "$output_dir/data/input_sha256.txt"

benchmark_csv="$output_dir/data/vote_model_benchmark_long.csv"

python -m rtknnc.vote_model_runner \
    --workspace "$workspace" \
    --dataset-dir "$input_stage" \
    --output "$benchmark_csv" \
    --force

# Fail before plotting if a future default change silently removes or adds a
# kernel.  The 840-row count is part of the experiment reported in the paper.
python - "$benchmark_csv" "${vote_models[@]}" <<'PY'
import csv
from pathlib import Path
import sys

benchmark = Path(sys.argv[1])
expected_models = tuple(sys.argv[2:])
with benchmark.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

actual_models = tuple(dict.fromkeys(row["vote_model"] for row in rows))
if actual_models != expected_models:
    raise SystemExit(
        f"unexpected vote models in {benchmark}: "
        f"expected {expected_models}, found {actual_models}"
    )
if len(rows) != 840:
    raise SystemExit(f"unexpected benchmark size in {benchmark}: {len(rows)} rows")
PY

scope_manifest="$output_dir/data/figure_scope.txt"
{
    printf 'Compared vote models (%d): %s\n' \
        "${#vote_models[@]}" "${vote_models[*]}"
    printf 'Four-kernel unpruned benchmark; manuscript comparison figure: moons.\n'
    printf 'Unpruned single-kernel supporting artifacts (%s): Swiss roll and bridge-fraction transition; neither is a manuscript figure.\n' \
        "$historical_vote_model"
    printf 'Native pruning mosaics: inv_r, inv_r2, and sigmoid on blobs K8 C5 and bridge K6 C3.\n'
    printf 'Archived conceptual PDFs are copied; the monochrome leadership graph and corrected forest are rendered from Graphviz sources.\n'
} > "$scope_manifest"

python "$workspace/tools/plot-vote-model-clusters.py" \
    --workspace "$workspace" \
    --dataset-dir "$input_stage" \
    --benchmark-csv "$benchmark_csv" \
    --output-dir "$output_dir/experimental" \
    --dpi 240

python "$workspace/tools/render-vote-model-gnuplot.py" \
    --workspace "$workspace" \
    --benchmark-csv "$benchmark_csv" \
    --layer-dir "$output_dir/experimental/layers" \
    --output-dir "$output_dir/experimental/gnuplot"

python "$workspace/tools/generate-native-structural-pruning-figures.py" \
    --workspace "$workspace" \
    --dataset-dir "$input_stage" \
    --executable "$workspace/rtknnc" \
    --output-dir "$output_dir/experimental/structural_pruning" \
    --dpi 240

mkdir -p "$output_dir/historical"
for historical_pdf in \
    figure1_roundtrip_reverse_neighbor \
    fig_2018_knn_graph \
    fig_2018_leadership_graph \
    fig_2018_cluster_tree
do
    cp -p \
        "$workspace/paper/$historical_pdf.pdf" \
        "$output_dir/historical/$historical_pdf.pdf"
    pdftoppm -png -r 180 -singlefile \
        "$output_dir/historical/$historical_pdf.pdf" \
        "$output_dir/historical/$historical_pdf"
done

# Reconstruct the rooted forest from the archived Graphviz leadership graph.
# The new source restores vertex 6, which was present in the leadership graph
# (6 -> 9) but absent from the archived forest rendering.
neato -n2 -Tpdf \
    "$workspace/paper/fig_2018_leadership_graph_black.dot" \
    -o "$output_dir/historical/fig_2018_leadership_graph_black.pdf"
pdftoppm -png -r 180 -singlefile \
    "$output_dir/historical/fig_2018_leadership_graph_black.pdf" \
    "$output_dir/historical/fig_2018_leadership_graph_black"

dot -Tpdf \
    "$workspace/paper/fig_2018_cluster_forest_corrected.dot" \
    -o "$output_dir/historical/fig_2018_cluster_forest_corrected.pdf"
pdftoppm -png -r 180 -singlefile \
    "$output_dir/historical/fig_2018_cluster_forest_corrected.pdf" \
    "$output_dir/historical/fig_2018_cluster_forest_corrected"

printf 'Vote models compared: %s\n' "${vote_models[*]}"
printf 'Single-model diagnostics use: %s\n' "$historical_vote_model"
echo 'Native baseline/pruned evidence: experimental/structural_pruning/'
echo "Article figures written to: $output_dir"
