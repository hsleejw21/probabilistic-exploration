# Released results

- `synthetic/six_benchmark_alpha_sweep`: six objectives, Uniform sweep,
  selected and common-alpha trajectories and endpoints (20 seeds).
- `synthetic/lowdim_mvr_runtime`: endpoint regret and runtime (15 seeds).
- `hpo_yahpo`: car and sylvine endpoint comparisons (30 seeds).
- `lunar`: optimization-terrain summaries (30 seeds); no held-out analysis.
- `figures/fig_main_applications`: current exported LogEI application figure.

Run `bash scripts/build_released_figures.sh` from the repository root.
Application histories and rollout media are omitted; the main application
figure is an exported artifact, not an aggregate-only reproduction.
