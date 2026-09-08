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
  Higher-dimensional YAHPO experiments are still running and are not yet used
  as evidence.

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
[reproduction notes](docs/reproducibility.md). The project license has not yet
been specified; choose one before making the repository public.
