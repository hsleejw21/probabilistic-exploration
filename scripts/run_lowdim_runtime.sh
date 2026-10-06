#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-results/synthetic/lowdim_mvr_runtime/rerun}"
jobs="${2:-1}"
seed_start="${3:-9100}"
num_seeds="${4:-15}"
python_bin="${PYTHON:-python3}"

if (( jobs < 1 || num_seeds < 1 )); then
  echo "jobs and num_seeds must be positive integers" >&2
  exit 2
fi

# Keep numerical libraries from creating extra worker pools. For the cleanest
# runtime comparison, retain the default jobs=1 and run on an otherwise idle CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python >=3.11 is required"'

"$python_bin" scripts/run_synthetic.py \
  --profile high_dimensional \
  --objectives ackley_shifted_narrow_2d rastrigin_mean_rotated_shifted_2d \
  --acquisitions ucb \
  --beta-modes constant \
  --policies standard decay_uniform_grid_a1_2 decay_mvr_grid_a1_2 \
  --budget 402 \
  --n-initial 2 \
  --no-learn-signal-variance \
  --seed-start "$seed_start" \
  --num-seeds "$num_seeds" \
  --jobs "$jobs" \
  --output-dir "$output_dir"

"$python_bin" scripts/summarize_lowdim_runtime.py \
  --history "$output_dir/history.csv" \
  --output-dir "$output_dir"

"$python_bin" scripts/plot_lowdim_runtime.py \
  --results-dir "$output_dir"
