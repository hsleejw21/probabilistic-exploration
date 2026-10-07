# Experiment and figure entry points

All commands below are run from the repository root. Experiment runners write
raw trial files to the requested output directory and resume completed trials.
Plotters read released aggregates unless an alternative directory is supplied.

## Figures reproducible from released aggregates

| Study | Experiment entry point | Figure entry point | Released aggregate |
|---|---|---|---|
| Six synthetic benchmarks | `run_aistats2027_synthetic.sh` | `plot_aistats2027_synthetic.py` | `results/synthetic/six_benchmark_alpha_sweep/` |
| YAHPO endpoint comparisons | `run_yahpo_hpo.py` | `plot_application_results.py` | `results/hpo_yahpo/greedy_packing/default/` |
| Low-dimensional MVR runtime | `run_lowdim_runtime.sh` | `plot_lowdim_runtime.py` | `results/synthetic/lowdim_mvr_runtime/` |

The Uniform schedule sweep and both selected-exponent and alpha=1/2 trajectories are generated
by `plot_aistats2027_synthetic.py`. Rebuild the complete aggregate-backed
figure set with:

```bash
bash scripts/build_released_figures.sh
```

`plot_yahpo.py` generates per-iteration YAHPO trajectories when a rerun
directory containing `history.csv` is supplied. Large histories and Lunar
rollout media are omitted from the compact release, so these trajectory and
storyboard artifacts are not presented as aggregate-only reproductions.

## Extending an experiment

The core implementation is under `src/probabilistic_exploration/`:

- register synthetic objectives in `benchmarks.py`;
- implement acquisitions in `acquisitions.py`;
- add exploration query rules and schedules in `exploration.py`;
- expose reusable protocol defaults in `config.py`;
- keep visual constants or reusable exporters in `plot_style.py`.

Use `run_synthetic.py` for new objective, acquisition, and policy combinations
instead of copying a paper-specific runner. A typical small extension is:

```bash
python scripts/run_synthetic.py \
  --profile smoke \
  --objectives ackley_shifted_narrow_2d \
  --acquisitions ucb \
  --beta-modes constant \
  --policies standard decay_uniform_grid_a1_2 \
  --output-dir /tmp/pe-extension
```

Plotting scripts accept result or output directories through command-line
arguments. New public figures should save both PDF and PNG, use embedded
TrueType fonts, and avoid machine-specific metadata.

## Runtime comparison pipeline

`run_lowdim_runtime.sh` is a thin, portable wrapper around `run_synthetic.py`.
It then calls `summarize_lowdim_runtime.py`, which converts one or more raw
`history.csv` files into compact public aggregates, followed by
`plot_lowdim_runtime.py`. The summarizer accepts repeated `--history` arguments
when method blocks were executed separately:

```bash
python scripts/summarize_lowdim_runtime.py \
  --history /path/to/standard/history.csv \
  --history /path/to/pe/history.csv \
  --output-dir /tmp/lowdim-runtime
```
