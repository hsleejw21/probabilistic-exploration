#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current eight-benchmark synthetic sweep. Run one acquisition
# per process so completed trial CSVs can be resumed independently.
acquisition="${1:?usage: $0 ucb|logei|ts [output-root] [seed-start] [num-seeds] [jobs]}"
output_root="${2:-results/reproduction/aistats2027_synthetic}"
seed_start="${3:-0}"
num_seeds="${4:-20}"
jobs="${5:-12}"

case "${acquisition}" in
  ucb) beta_mode=constant ;;
  logei) beta_mode=native ;;
  ts) beta_mode=logarithmic ;;
  *) printf 'Unknown acquisition: %s\n' "${acquisition}" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

objectives=(
  hartmann6_active_6d
  griewank_mean_shifted_10d
  trid_normalized_10d
  rosenbrock_mean_20d
  bent_cigar_shifted_30d
  sphere_mean_shifted_30d
  griewank_mean_shifted_30d
  sum_powers_normalized_30d
)

policies=(
  standard
  decay_uniform_grid_a1_4 decay_uniform_grid_a1_3
  decay_uniform_grid_a1_2 decay_uniform_grid_a2_3
  decay_uniform_grid_a3_4
  decay_greedy_packing_grid_a1_4 decay_greedy_packing_grid_a1_3
  decay_greedy_packing_grid_a1_2 decay_greedy_packing_grid_a2_3
  decay_greedy_packing_grid_a3_4
)

python scripts/run_synthetic.py \
  --profile high_dimensional \
  --objectives "${objectives[@]}" \
  --acquisitions "${acquisition}" \
  --beta-modes "${beta_mode}" \
  --policies "${policies[@]}" \
  --budget 402 --n-initial 2 \
  --no-learn-signal-variance \
  --ts-candidate-schedule sqrt \
  --ts-candidate-min 128 \
  --ts-candidate-max 512 \
  --ts-candidate-growth-scale 64 \
  --seed-start "${seed_start}" --num-seeds "${num_seeds}" \
  --jobs "${jobs}" --output-dir "${output_root}/${acquisition}"
