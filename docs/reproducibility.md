# Reproducibility

## Environment

The current synthetic runs used Python 3.12 with NumPy 1.26.4 or newer,
SciPy 1.17.1, scikit-learn 1.9.0, pandas 3.0.0, and Matplotlib 3.10.8. These
versions are recorded in `pyproject.toml` and `configs/aistats2027.yaml`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

YAHPO requires its released compatibility stack:

```bash
pip install -e '.[yahpo]'
```

The YAHPO extra pins NumPy 1.26.4 because `yahpo-gym==1.0.2` and its
ConfigSpace dependency do not use the same NumPy range as the core package.
Benchmark data downloaded by YAHPO Gym remain outside this repository.

Lunar Lander uses:

```bash
pip install -e '.[lunar]'
```

The controller objective follows the Lunar Lander example distributed with
TuRBO. Each candidate controller is evaluated on 50 terrains fixed within an
optimization run; the held-out analysis uses 200 shared unseen terrains per
final controller.

## Synthetic paper protocol

`scripts/run_aistats2027_synthetic.sh` records the complete current synthetic
matrix: eight objectives, GP-UCB/LogEI/GP-TS, Standard, five Uniform schedules,
and five Greedy Packing schedules. The runner writes one CSV per trial and
skips completed trials when restarted.

Run the acquisitions separately:

```bash
bash scripts/run_aistats2027_synthetic.sh ucb    /tmp/aistats2027 0 20 12
bash scripts/run_aistats2027_synthetic.sh logei  /tmp/aistats2027 0 20 12
bash scripts/run_aistats2027_synthetic.sh ts     /tmp/aistats2027 0 20 12
```

The last three arguments are the seed start, number of seeds, and worker
count. Comparisons are paired within objective, acquisition, and seed. For
shared servers, use one BLAS thread per trial; the launcher sets the common
OpenMP and BLAS variables accordingly.

The synthetic model uses a Matérn-3/2 GP on the unit cube, centered targets,
fixed unit signal variance, observation-noise variance and GP nugget 0.01,
lengthscale bounds `[0.01, 20]`, and refitting every five evaluations.
Acquisition and posterior-mean recommendation optimization use 20 starts.
GP-UCB uses multiplier 2. GP-TS uses covariance multiplier `log(t+1)` and a
fresh Sobol candidate set of size `clip(ceil(64*sqrt(t)), 128, 512)`.

The reported inference regret is evaluated at the current posterior-mean
recommendation. It is not cumulative best-observed regret and therefore need
not be monotone.

## Released aggregates and figures

Large raw trial histories are excluded. The current synthetic result directory
contains:

- `trajectory_summary.csv`: 400-step mean and standard deviation curves;
- `alpha_sweep_summary.csv`: endpoints for all five exponents and both rules;
- `selected_uniform_endpoints.csv`: paired endpoint comparisons for Figure 1;
- `protocol.json`: a machine-readable record of the released snapshot.

Rebuild all three synthetic figures from these aggregates:

```bash
python scripts/plot_aistats2027_synthetic.py
```

YAHPO endpoint comparisons and Lunar held-out comparisons are also rebuilt
from released aggregates by `scripts/plot_application_results.py`. Rebuild the
complete aggregate-backed figure set with:

```bash
bash scripts/build_released_figures.sh
```

Per-iteration YAHPO trajectories require the omitted raw `history.csv` files,
and the Lunar storyboard requires rollout media. Their existing exported
figures are retained as result artifacts, while the compact public package
focuses its one-command rebuild on figures supported by released inputs.

## Low-dimensional runtime comparison

The appendix runtime comparison uses Ackley 2D and Rastrigin 2D with GP-UCB,
two initial observations, 400 BO evaluations, and 15 paired seeds. Standard
uses GP-UCB at every round. Uniform PE and MVR PE share the `alpha=1/2`
exploration decisions; MVR optimizes posterior variance on those rounds.

Rebuild the released figure from its compact per-seed aggregate:

```bash
python scripts/plot_lowdim_runtime.py
```

To rerun the experiment and rebuild both aggregate and figure:

```bash
bash scripts/run_lowdim_runtime.sh /tmp/lowdim-runtime 1 9100 15
```

The runtime is the sum of per-round query selection, objective evaluation, GP
fit, recommendation, and posterior-diagnostic times. Launcher and file-I/O
time are excluded. Use one worker on an otherwise idle CPU for the least noisy
timing comparison; the released snapshot used the common server configuration
recorded in its `protocol.json`, so its absolute seconds are machine dependent.

## Anonymity and release boundaries

The `paper/` directory is ignored and is not part of the release. Public text
and generated files should not contain names, email addresses, institutional
paths, hostnames, credentials, or machine-specific output directories. Git
remote URLs and commit authorship are separate hosting metadata; use a fresh
anonymous remote with clean metadata for double-blind review.

The release contains only the result sets used by the current manuscript.
Earlier pilot studies, grid-size checks, and superseded snapshots are omitted.
