# Eight-benchmark synthetic alpha sweep

This directory is the aggregate-data release for the current synthetic section
of the paper.

## Scope

- Objectives: Hartmann 6D, Griewank 10D, Trid 10D, Rosenbrock 20D, Shifted
  Bent Cigar 30D, Sphere 30D, Griewank 30D, and Sum of Different Powers 30D.
- Acquisitions: GP-UCB, LogEI, and GP-TS.
- Policies: Standard, Uniform PE, and Greedy Packing PE.
- Schedule exponents: `1/4`, `1/3`, `1/2`, `2/3`, and `3/4`.
- Runs: 20 paired seeds per objective-acquisition-policy cell.
- Budget: two initial observations followed by 400 BO evaluations.

`selected_uniform_endpoints.csv` contains Standard and selected-Uniform mean,
standard deviation, and paired bootstrap interval. A positive `mean_gain`
means lower regret for Uniform. Selection and evaluation use the same 20 runs.

`alpha_sweep_summary.csv` contains all Uniform and Greedy Packing endpoints.
`trajectory_summary.csv` contains the aggregate curves needed for the main
figure. Raw per-trial histories are excluded because of their size.

Rebuild the figures from the repository root:

```bash
python scripts/plot_aistats2027_synthetic.py
```
