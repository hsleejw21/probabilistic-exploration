# Probabilistic Exploration for Efficient Bayesian Optimization

Code and compact experimental results for exploration-aided Bayesian optimization.
At round t, the algorithm selects an exploration query with probability
`min(1, log(t+1)/(t+1)^alpha)` and otherwise follows the base acquisition rule.
**Standard** uses the base acquisition at every round. **Uniform** samples an
exploration point uniformly; **Greedy Packing** uses a nested Sobol candidate
set and is evaluated in the applications.

## Paper experiments

- **Six synthetic benchmarks:** Trid 10D, Rosenbrock 20D, shifted Bent Cigar,
  shifted Sphere, shifted Griewank, and normalized Sum of Different Powers (30D).
  GP-UCB, LogEI, and GP-TS; two initial observations plus 400 BO evaluations;
  20 seeds; Uniform exponents 1/4, 1/3, 1/2, 2/3, and 3/4.
- **YAHPO:** car (14D, 120 total evaluations) and sylvine (28D, 200 total),
  ten initial observations, 30 seeds, and alpha=1/2. Endpoint comparisons use
  GP-UCB, LogEI, and GP-TS and compare each exploration rule with Standard.
- **Lunar Lander:** 12 controller parameters, 400 total evaluations including
  24 initial points, and 30 seeds. Each evaluation averages 50 terrains fixed
  within a run. The main LogEI comparison includes Uniform and Greedy Packing.
- **Runtime:** shifted Ackley 2D and shifted, rotated Rastrigin 2D; GP-UCB;
  two initial observations plus 400 evaluations; 15 seeds. Uniform and MVR
  share exploration decisions at alpha=1/2. MVR maximizes posterior variance
  only on exploration rounds; both use GP-UCB on other rounds.

With selected exponents, Uniform improves mean synthetic endpoint regret in
17/18 comparisons (16 pointwise bootstrap intervals exclude zero). With
alpha=1/2, it improves 16/18 (11 intervals exclude zero; no conclusive losses).
Selection and evaluation use the same 20 runs; selected results describe the
observed sweep rather than independent validation of a tuned policy.

## Reproduction

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
bash scripts/build_released_figures.sh
# Rerun one synthetic acquisition block:
bash scripts/run_aistats2027_synthetic.sh ucb /tmp/aistats2027 0 20 12
```

The figure command rebuilds the six-panel schedule sweep, selected-exponent
trajectories, alpha=1/2 trajectories, HPO endpoint comparison, and runtime figure.
See [reproduction details](docs/reproducibility.md), [script index](scripts/README.md),
and [protocol](configs/aistats2027.yaml). Add objectives in
`src/probabilistic_exploration/benchmarks.py`, acquisitions in `acquisitions.py`,
and exploration rules in `exploration.py`.

Results contain compact aggregates, not full raw trial histories. Main application
trajectories and Lunar rollout media are not rebuildable from these aggregates;
the current exported application figure is included for reference.
The manuscript, motivating example, and held-out terrain analysis are excluded.
Internal policy identifiers retain their original names for compatibility.

## Licenses and anonymity

See [third-party assets](docs/third_party_assets.md). A license for the original
project code has not yet been selected; no blanket license is granted here.
Manuscript files and local paths are excluded. Git hosting and commit metadata
must be anonymized by the anonymous mirror used for review.
