# Experiment index

| Study | Status | Purpose | Public contents |
|---|---|---|---|
| 30D acquisition generalization | Complete | Test PE across UCB, LogEI, MES-Gumbel, and theory-aligned GP-TS | aggregate CSV, audit, figures |
| Dimension–alpha panels | Complete | Test whether the useful decay rate changes with dimension and benchmark | figures and report tables |
| Practical alpha selection | Exploratory | Evaluate simple data-driven alpha selection on fresh seeds | figures and report tables |
| Lunar Lander 12D | Complete | Test transfer to simulation-based controller tuning | figures and report tables |
| scikit-learn HPO | Complete | Test transfer to real-data model tuning | metadata, audit, summary |
| YAHPO high-dimensional HPO | Complete | Test transfer to 14D and 28D HPO surrogates | aggregate CSVs, audits, figures, and full protocol |
| Synthetic Greedy Packing | Complete | Compare farthest-point PE with Standard and Uniform PE across 30D acquisitions, Rastrigin dimensions, fresh seeds, and cross-benchmark transfer | aggregate CSVs, audits, figures, and protocol |
| YAHPO Greedy Packing | Complete | Compare Greedy PE with Standard and Uniform PE in 14D and 28D HPO | aggregate CSVs, audits, figures, and protocols |
| Lunar Greedy Packing | Complete | Compare Greedy PE with Standard and Uniform PE on held-out terrains | aggregate CSVs, audit, figure, and protocol |
| Larger-grid confirmation | Complete | Check whether twofold and fourfold finite grids change the Greedy conclusion | fresh-seed synthetic, YAHPO, and Lunar summaries |

Positive differences favor the method named before “over” in each figure. The
Greedy Packing figures show Uniform over Standard, Greedy over Standard, and
Greedy over Uniform together. Raw trial histories and preliminary Greedy
setting sweeps are intentionally omitted; each included completed study retains
its final protocol, aggregate outputs, and audit record.
