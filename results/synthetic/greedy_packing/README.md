# Greedy Packing exploration

This directory contains the final audited comparisons of finite-grid Greedy
Packing PE with both Standard Bayesian optimization and Uniform PE. The
original Uniform studies elsewhere in the release remain unchanged; their
exact matching trials are repeated here only as comparators.

Every public comparison uses a higher-is-better convention:

- `Uniform over Standard` = Standard regret minus Uniform PE regret;
- `Greedy over Standard` = Standard regret minus Greedy PE regret;
- `Greedy over Uniform` = Uniform PE regret minus Greedy PE regret.

The included stages are:

- `acquisition_30d/`: three 30D objectives and four acquisition functions;
- `rastrigin_dimensions/`: Rastrigin at 2D, 5D, 10D, 20D, and 30D;
- `rastrigin_validation/`: independently seeded validation at 10D and 30D;
- `cross_benchmark/`: exploratory transfer to Ackley and Rosenbrock across
  2D--30D.

The alpha--grid tuning sweep itself is intentionally omitted. The settings
selected before the independent validation are recorded in
`rastrigin_validation/selected_config.json` and in `protocol.json`.

Raw trial histories are omitted because of size. Each cell retains its audit,
endpoint summary, paired comparisons, and switching summary.
