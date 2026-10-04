# Experiment index

| Study | Status | Purpose | Public contents |
|---|---|---|---|
| Eight-benchmark synthetic alpha sweep | Complete, 20-seed paper snapshot | Compare Standard and Uniform across three acquisitions; report Uniform and Greedy Packing schedule sensitivity | aggregate trajectories, endpoint CSVs, protocol, figures |
| YAHPO high-dimensional HPO | Complete | Test PE on 14D XGBoost and 28D joint-tuning surrogate tasks | aggregate CSVs, audits, figures, protocol |
| Lunar Lander 12D | Complete | Test controller tuning and held-out terrain generalization | aggregate CSVs, audits, figures, protocol |
| 30D acquisition generalization | Earlier completed snapshot | Test PE across UCB, LogEI, MES-Gumbel, and GP-TS | aggregate CSVs, audit, figures |
| Dimension-alpha panels | Earlier completed snapshot | Study the decay rate across dimension and benchmark | figures and aggregate tables |
| Practical alpha selection | Exploratory | Evaluate pilot-based alpha selection on fresh seeds | figures and aggregate tables |
| Synthetic Greedy Packing | Earlier completed snapshot | Compare farthest-point PE with Standard and Uniform across 30D and transfer settings | aggregate CSVs, audits, figures, protocol |
| scikit-learn HPO | Earlier completed snapshot | Small real-data HPO transfer study | metadata, audit, summary |

The current paper-facing synthetic release is
`synthetic/eight_benchmark_alpha_sweep`. Its selected schedule is chosen and
evaluated on the same 20 paired runs, so the comparison is descriptive rather
than held out. Positive `mean_gain` in its endpoint table means Standard regret
minus Uniform regret and therefore favors Uniform.

Raw trial histories and preliminary screens are intentionally omitted. Each
included completed study retains its protocol and aggregate outputs; studies
from earlier research stages remain available for provenance but should not be
confused with the current eight-benchmark protocol.
