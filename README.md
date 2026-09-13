# Probabilistic Exploration for Efficient Bayesian Optimization

This repository studies a simple wrapper for Bayesian optimization: with a
controlled probability, replace the base acquisition's query by an explicit
exploration query. The central question is whether scheduling that probability
improves the final recommendation efficiently and whether the effect transfers
across acquisitions, dimensions, synthetic functions, and applied tasks.

## Main findings so far

- PE can improve several 30-dimensional acquisition-function settings, but the
  effect is not uniform across every benchmark and acquisition.
- The best decay exponent changes with the function and dimension. A single
  universal \(\alpha\) is therefore not supported by the current evidence.
- On Lunar Lander controller tuning, decay PE clearly improves LogEI and uses
  fewer exploratory queries than fixed `p=0.2`; UCB and MES move in the same
  direction, while GP-TS is weak on this task.
- A small scikit-learn HPO study does not show a reliable overall PE gain.
- In the completed 14D YAHPO study, decay PE is significantly better in 5 of
  24 comparisons and significantly worse in 1. The clearest gain is on the
  `car` task, where alpha 0.5 improves UCB, LogEI, and MES-Gumbel by reducing
  high-loss failures.
- In the completed 28D YAHPO study, alpha 0.5 significantly improves LogEI and
  GP-TS on `sylvine`, with no significant decay-policy loss. Most other cells
  are tied or inconclusive.
- Finite-grid Greedy Packing is strongly problem-dependent. It is effective on
  Rastrigin 10D and 30D and preserves PE gains for several 30D Sparse Hartmann
  acquisitions. Fresh Rastrigin seeds retain an advantage, but transferring
  the same schedule to Ackley and Rosenbrock usually does not. Uniform PE is
  therefore still the more reliable default.

Knowledge Gradient is implemented as an optional acquisition for a planned
follow-up, but no KG result is included in the findings yet.

## Repository map

```text
src/probabilistic_exploration/  reusable BO, PE, GP, and benchmark code
scripts/                        generic experiment and plotting entry points
configs/                        human-readable protocols
results/                        audited aggregate results and figures
paper/                          topic-based LaTeX report and publication figures
docs/                           method and reproduction notes
```

The repository deliberately excludes chronological meeting reports, quota-based
methods, Rover experiments, separately supplied comparison code, and all older
fixed-beta GP-TS results. It contains only experiments about probabilistic
exploration.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python scripts/run_synthetic.py --profile smoke --jobs 2 --output-dir /tmp/pe-smoke
```

See [the method](docs/method.md), [experiment index](results/README.md), and
[reproduction notes](docs/reproducibility.md). Public figures and tables follow
the shared [visual style and file-naming convention](docs/visual_style.md). The
project license has not yet been specified; choose one before making the
repository public.
