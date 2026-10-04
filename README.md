# Probabilistic Exploration for Efficient Bayesian Optimization

This repository contains the implementation and released experiment artifacts
for probabilistic exploration (PE) in Bayesian optimization. At BO round
\(t\), PE replaces the base acquisition query with an explicit exploration
query with probability

\[
p_t=\min\left\{1,\frac{\log(t+1)}{(t+1)^\alpha}\right\}.
\]

The current paper snapshot evaluates two exploration rules:

- **Uniform:** sample the exploration point uniformly from the search space.
- **Greedy Packing:** select the point farthest from evaluated inputs on a
  nested Sobol candidate grid.

Both rules wrap GP-UCB, LogEI, or GP-TS without changing the acquisition on
the remaining rounds. **Standard** denotes the base acquisition with no PE.

## Current empirical scope

### Synthetic optimization

The main synthetic study uses eight objectives from 6D to 30D, three
acquisitions, 400 BO evaluations after two shared initial observations, and 20
paired runs per setting. It evaluates
\(\alpha\in\{1/4,1/3,1/2,2/3,3/4\}\) for both Uniform and Greedy Packing.

For the Uniform schedule selected by mean final inference regret within this
five-value sweep:

- Uniform has lower mean endpoint regret than Standard in 23 of 24
  objective-acquisition comparisons.
- Pointwise paired bootstrap 95% intervals exclude zero in 21 comparisons.
- \(\alpha=1/4\) is selected in 16 of 24 comparisons, while Hartmann 6D and
  several lower-dimensional settings favor less persistent exploration.

The selected schedule is evaluated on the same runs used for selection. These
numbers describe the observed sweep and are not an out-of-sample tuning claim.
All five Uniform and Greedy Packing schedules are included in the release.

### Applied studies

- **YAHPO Gym:** three 14D XGBoost tasks and two 28D joint-tuning tasks, with
  30 paired runs. Effects depend on task and acquisition; the clearest gains
  occur on the `car` and `sylvine` tasks.
- **Lunar Lander:** a 12-parameter controller with 30 paired runs. Uniform and
  Greedy Packing improve the LogEI controller on 200 shared held-out terrains;
  their direct difference is inconclusive.

## Repository map

```text
src/probabilistic_exploration/  BO, GP, PE, and benchmark implementation
scripts/                        experiment and plotting entry points
configs/                        recorded experiment protocols
results/                        aggregate results, audits, and figures
docs/                           method and reproducibility notes
paper/                          earlier consolidated research-report snapshot
```

The release omits large per-iteration trial collections, preliminary screens,
machine-specific paths, and credentials. Aggregate trajectory data needed to
rebuild the released synthetic figures is included.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python scripts/run_synthetic.py \
  --profile smoke --jobs 2 --output-dir /tmp/pe-smoke
```

Rebuild the current synthetic figures with:

```bash
python scripts/plot_aistats2027_synthetic.py
```

Run one acquisition block of the full eight-benchmark protocol with:

```bash
bash scripts/run_aistats2027_synthetic.sh ucb /tmp/aistats2027 0 20 12
```

See [the method](docs/method.md), [reproduction notes](docs/reproducibility.md),
the exact [AISTATS 2027 protocol](configs/aistats2027.yaml), and the
[experiment index](results/README.md).

The project license has not yet been specified.
