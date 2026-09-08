# scikit-learn HPO

This transfer study covers six real-data model-tuning tasks, GP-UCB and LogEI,
four policies, twenty confirmation seeds, and 100 evaluations. The audit passes
all 960 trials with fixed unit signal variance and paired initial designs.

No PE policy shows a significant held-out test-loss improvement over Standard
across the 36 task--acquisition--PE comparisons. This is a valid negative
result and prevents overgeneralising the synthetic and Lunar findings.
