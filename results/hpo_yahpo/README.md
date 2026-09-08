# High-dimensional YAHPO HPO

Status: **14D complete; 28D and 38D in progress**.

The study covers 14D, 28D, and 38D YAHPO surrogate tasks with GP-UCB, LogEI,
theory-aligned GP-TS, and MES-Gumbel. It compares Standard, decay alpha 0.5,
decay alpha 1, and fixed p=0.2 on thirty paired confirmation seeds. All
dimensions use ten shared initial Sobol points; total budgets are 120, 200, and
250. GP signal variance is fixed to one.

## Completed 14D stage

The 14D stage contains three XGBoost tuning tasks (`credit-g`, `car`, and
`blood-transfusion-service-center`), 120 evaluations, and 1,440 audited trials.
The endpoint is best-observed validation log loss, so lower is better. In the
public comparison table, improvement is `Standard loss - PE loss`; positive
therefore always means that PE is better.

The result is useful but not universal:

- Decay PE is significantly better than Standard in 5 of 24 comparisons,
  significantly worse in 1, and inconclusive in the other 18.
- On `car`, decay alpha 0.5 improves LogEI by 0.0527
  [0.0100, 0.1045], MES-Gumbel by 0.0800 [0.0224, 0.1455], and GP-UCB by
  0.0973 [0.0418, 0.1584]. The trajectories show that the gain mainly comes
  from avoiding a small number of high-loss failures rather than winning most
  paired seeds.
- On `blood-transfusion-service-center`, decay alpha 1 improves MES-Gumbel by
  0.0067 [0.0007, 0.0146] and GP-UCB by 0.0041 [0.0008, 0.0076]. Decay alpha
  0.5 makes GP-TS worse by 0.0084 [0.0052, 0.0117].
- No comparison is conclusive on `credit-g`. Across all three tasks, GP-TS
  shows no reliable PE gain.
- Fixed p=0.2 also improves LogEI, MES-Gumbel, and GP-UCB on `car`. The 14D
  result therefore supports problem-dependent exploration, but does not show
  that decay is always better than a fixed exploration probability.

`paired_improvements_14d.csv` contains all paired means, 95% bootstrap
intervals, and win/tie/loss counts. `trial_summary_14d.csv` contains one row per
completed trial. See the compact
[endpoint figure](figures/fig_yahpo_14d_endpoint_improvement.pdf) and the 95%
CI trajectories for [credit-g](figures/fig_yahpo_14d_credit_g_trajectories_ci95.pdf),
[car](figures/fig_yahpo_14d_car_trajectories_ci95.pdf), and
[blood transfusion](figures/fig_yahpo_14d_blood_transfusion_trajectories_ci95.pdf).
Results from 28D and 38D will be added only after their full trial counts pass
the same audit.
