# High-dimensional YAHPO comparison

This release contains the YAHPO results used by the current manuscript. The
study covers three 14D XGBoost tasks and two 28D joint-tuning tasks, with
GP-UCB, GP-TS, LogEI, and MES-Gumbel. Every comparison uses 30 paired seeds.

Standard BO, Uniform PE, and Greedy Packing PE share the same initial
observations within each task, acquisition, and seed. Both PE methods use
`alpha=1/2`; only the exploration query differs. The 14D and 28D budgets are
120 and 200 total evaluations, respectively, including ten initial points.

The released files are:

- `greedy_packing/default/{14d,28d}/`: protocols, audits, endpoint summaries,
  per-trial summaries, and paired comparisons;
- `figures/fig_appendix_hpo_acquisitions.{pdf,png}`: the aggregate-backed
  appendix figure.

YAHPO returns a loss. Stored candidate-minus-baseline differences are negated
for the figure so that positive values favor the first method named. Large
per-iteration histories are omitted.
