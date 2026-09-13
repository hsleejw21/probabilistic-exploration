# Greedy Packing on YAHPO

These results compare Standard BO, Uniform PE, and finite-grid Greedy Packing
PE under matched YAHPO protocols. All three policies share the same initial
observations within each task, acquisition, and seed. The PE arms use
`alpha=1/2`; only the exploration-point rule differs.

- `default/`: 30 paired seeds (`5--34`) with the default grid
  `|G_t| = 256 + ceil(16 t)`.
- `grid4x_fresh/`: 15 new paired seeds (`35--49`) with the larger grid
  `|G_t| = 1024 + ceil(64 t)`.

The 14D studies use three XGBoost tasks, 120 evaluation steps, and ten initial
points. The 28D studies use two `iaml_super` tasks, 200 evaluation steps, and
ten initial points. Both include GP-UCB, theory-aligned GP-TS, LogEI, and
MES-Gumbel. Signal variance is fixed at one.

YAHPO returns a loss. In `paired_comparisons.csv`, the stored difference is
candidate loss minus baseline loss, so a negative value favors the candidate.
Publication figures convert this to a positive-is-better loss improvement.
Raw evaluation histories are omitted; each stage retains the protocol, audit,
endpoint summary, trial summary, and paired comparisons.
