# Six-benchmark Uniform comparison

20 seeds per objective/acquisition/setting, two initial points and 400 BO steps.
`trajectory_summary.csv` and `selected_uniform_endpoints.csv` use selected
exponents; `alpha_half_trajectory_summary.csv` and `alpha_half_endpoints.csv`
use alpha=1/2. `alpha_sweep_summary.csv` reports all five Uniform exponents.
Regenerate with `python scripts/plot_aistats2027_synthetic.py`.
Selected exponents are chosen and evaluated using the same runs.
