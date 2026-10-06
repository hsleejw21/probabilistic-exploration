# Experiment index

| Study | Status | Purpose | Public contents |
|---|---|---|---|
| Eight-benchmark synthetic alpha sweep | Complete, 20-seed paper snapshot | Compare Standard and Uniform across three acquisitions; report Uniform and Greedy Packing schedule sensitivity | aggregate trajectories, endpoint CSVs, protocol, figures |
| Low-dimensional MVR runtime | Complete, 15 paired seeds | Compare Standard, Uniform PE, and MVR PE in endpoint regret and computation time | per-seed aggregate, summary, protocol, PDF/PNG figure |
| YAHPO high-dimensional HPO | Complete | Test PE on 14D XGBoost and 28D joint-tuning surrogate tasks | aggregate CSVs, audits, figures, protocol |
| Lunar Lander 12D | Complete | Test controller tuning and held-out terrain generalization | aggregate CSVs, audits, figures, protocol |

The current paper-facing synthetic release is
`synthetic/eight_benchmark_alpha_sweep`. Its selected schedule is chosen and
evaluated on the same 20 paired runs, so the comparison is descriptive rather
than held out. Positive `mean_gain` in its endpoint table means Standard regret
minus Uniform regret and therefore favors Uniform.

Raw trial histories, preliminary screens, grid-size checks, and superseded
snapshots are intentionally omitted. Each included study retains its protocol,
audits where applicable, and the compact aggregates needed for the released
figures and tables.

The paper-facing plotting entry points and their inputs are indexed in
[`scripts/README.md`](../scripts/README.md).
