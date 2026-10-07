# Low-dimensional MVR runtime comparison

This directory contains the public aggregate for the appendix comparison of
Standard GP-UCB, Uniform, and maximum-variance exploration (MVR).

## Protocol

- shifted Ackley 2D and rotated, shifted Rastrigin 2D;
- two scrambled-Sobol initial observations followed by 400 BO evaluations;
- GP-UCB with posterior-standard-deviation multiplier 2;
- Uniform and MVR use the same
  `min(1, log(t+1)/(t+1)^alpha)` schedule with `alpha=1/2`;
- 15 paired seeds shared across the three methods;
- MVR maximizes posterior variance with the same 20-start continuous optimizer
  used for acquisition optimization.

`runtime` is the sum of per-round wall-clock measurements for query selection,
objective evaluation, GP fitting, recommendation, and posterior diagnostics.
Job launch and file I/O are excluded. Absolute timings depend on hardware and
server load; paired method comparisons are the intended use.

## Files

- `per_seed.csv`: final inference regret and runtime for every paired run;
- `summary.csv`: mean and sample standard deviation by objective and method;
- `paired_comparisons.csv`: paired mean improvements and bootstrap intervals;
- `protocol.json`: machine-readable experimental settings;
- `figures/`: released PDF and PNG figure.

Rebuild the figure from the aggregate:

```bash
python scripts/plot_lowdim_runtime.py
```

Rerun the experiment, aggregate it, and plot it with:

```bash
bash scripts/run_lowdim_runtime.sh /tmp/lowdim-runtime 1 9100 15
```

The worker count defaults to one to reduce timing interference. Each numerical
library is also restricted to one thread by the launcher. The launcher uses
`python3` from the active environment; set `PYTHON=/path/to/python` to select
an explicit Python 3.11--3.13 interpreter.
