# High-dimensional YAHPO HPO

Status: **14D and 28D complete; 38D in progress**.

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

## Completed 28D stage

The 28D stage contains two `iaml_super` tuning tasks (`OpenML-40981` and
`sylvine`), 200 evaluations, and 960 audited trials. Of the 16 decay-policy
comparisons, 2 are significantly better than Standard, none is significantly
worse, and 14 are inconclusive.

- On `sylvine`, decay alpha 0.5 improves LogEI by 0.0056
  [0.0012, 0.0099] and GP-TS by 0.0103 [0.0029, 0.0174]. GP-TS wins 21 of
  30 paired seeds; LogEI wins 13, loses 4, and ties 13.
- No comparison is conclusive on `OpenML-40981`. UCB and MES-Gumbel usually
  reach the same endpoint across policies, and many LogEI seeds also tie.
- Every fixed p=0.2 comparison is inconclusive. The 28D result therefore adds
  evidence that decay PE can transfer to HPO, but the effect remains dependent
  on the task and acquisition.

`paired_improvements_28d.csv` contains the complete intervals and paired
counts. See the compact
[endpoint figure](figures/fig_yahpo_28d_endpoint_improvement.pdf) and the 95%
CI trajectories for
[OpenML-40981](figures/fig_yahpo_28d_openml_40981_trajectories_ci95.pdf) and
[sylvine](figures/fig_yahpo_28d_sylvine_trajectories_ci95.pdf).
The 38D result will be added only after all trials pass the same audit.
