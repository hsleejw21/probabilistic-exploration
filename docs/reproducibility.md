# Reproducibility

Install the core package with `pip install -e .`. Add `[lunar]`, `[yahpo]`, or
`[test]` for the corresponding optional dependencies.

Every runner writes one file per trial and safely resumes completed trials.
Use a separate output directory whenever the protocol changes. The public
results contain metadata, audits, and aggregate CSV files; large per-iteration
histories and trial files are not committed.

Examples:

```bash
python scripts/run_synthetic.py --help
python scripts/run_sklearn_hpo.py --help
python scripts/run_yahpo_hpo.py --help
```

Finite-grid Greedy Packing can be reproduced with the generic synthetic
runner. For example, the GP-UCB part of the 30D comparison uses:

```bash
python scripts/run_synthetic.py \
  --profile high_dimensional \
  --objectives bent_cigar_shifted_30d hartmann6_sparse_30d rosenbrock_mean_30d \
  --acquisitions ucb --beta-modes constant \
  --policies standard decay_uniform_grid_a1_2 decay_greedy_packing_grid_a1_2 \
  --budget 500 --n-initial 2 --seed-start 5 --num-seeds 10 \
  --noise-variance 0.01 --gp-nugget-variance 0.01 \
  --center-y --no-normalize-y --no-learn-signal-variance \
  --greedy-packing-grid-initial 256 --greedy-packing-grid-growth 16 \
  --jobs 32 --output-dir /tmp/pe-greedy-packing-30d
```

Repeat the command for LogEI and MES-Gumbel with their native modes. GP-TS
uses logarithmic beta and a growing fresh Sobol discretization in the audited
results; the exact stage-specific settings are recorded in
`results/synthetic/greedy_packing/protocol.json`. The dimension study reuses
matched Standard and Uniform trials and adds Greedy Packing on Rastrigin from
2D to 30D. The release also contains a fresh-seed Rastrigin validation and an
exploratory transfer to Ackley and Rosenbrock. Preliminary setting-sweep
outputs are intentionally excluded. Run `python scripts/plot_greedy_packing.py`
after placing the audited aggregates in
`results/synthetic/greedy_packing/`.

YAHPO also requires benchmark data downloaded through `yahpo-gym`; local data
paths are machine-specific and are therefore not included in this repository.
