"""Configuration for the probabilistic-exploration experiments.

``ExperimentConfig`` deliberately defaults to the paper execution path:
scikit-learn GP inference and SciPy multistart L-BFGS-B with 20 random starts.
Values omitted by the paper remain marked as assumptions and are recorded in
every result directory's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .exploration import DIMENSION_ALPHA_POLICIES, FEEDBACK_POLICIES


# [PAPER] Figure-to-method mapping.
FIGURE_SPECS: dict[str, tuple[str, str]] = {
    "fig1_ucb_increasing": ("ucb", "increasing"),
    "fig2_ts_constant": ("ts", "constant"),
    "fig3_ucb_constant": ("ucb", "constant"),
    "fig4_ts_increasing": ("ts", "increasing"),
}

PAPER_ACQUISITIONS = ("ucb", "ts")
POLICIES = ("standard", "fixed_uniform", "decay_uniform")

# Keep ``POLICIES`` as the paper/default set so older launchers that request
# ``--policies all`` remain unchanged. Research launchers may explicitly
# select the additional fixed-probability policies below.
PROBABILITY_SENSITIVITY_POLICIES = (
    "standard",
    "fixed_p010",
    "fixed_p020",
    "fixed_p030",
    "fixed_uniform",
    "decay_uniform",
)
AVAILABLE_POLICIES = tuple(dict.fromkeys(POLICIES + PROBABILITY_SENSITIVITY_POLICIES))
AVAILABLE_POLICIES = tuple(dict.fromkeys(AVAILABLE_POLICIES + FEEDBACK_POLICIES))
AVAILABLE_POLICIES = tuple(
    dict.fromkeys(AVAILABLE_POLICIES + DIMENSION_ALPHA_POLICIES)
)
POLICY_LABELS = {
    "standard": "Standard",
    "fixed_p010": "Fixed $p=0.1$",
    "fixed_p020": "Fixed $p=0.2$",
    "fixed_p030": "Fixed $p=0.3$",
    "fixed_uniform": "Fixed Uniform",
    "fixed_mvr": "Fixed MVR",
    "decay_uniform": "Decay Uniform",
    "decay_mvr": r"Decay MVR ($\alpha=2/3$)",
    "decay_uniform_a1": r"Decay Uniform ($\alpha=1$)",
    "decay_mvr_a1": r"Decay MVR ($\alpha=1$)",
    "decay_uniform_a4_3": r"Decay Uniform ($\alpha=4/3$)",
    "decay_mvr_a4_3": r"Decay MVR ($\alpha=4/3$)",
}

for _rule, _label in (("uniform", "Uniform"), ("mvr", "MVR")):
    for _slug, _alpha in (
        ("1_2", "1/2"),
        ("2_3", "2/3"),
        ("5_6", "5/6"),
        ("1", "1"),
        ("7_6", "7/6"),
        ("4_3", "4/3"),
        ("3_2", "3/2"),
    ):
        POLICY_LABELS[f"decay_{_rule}_grid_a{_slug}"] = (
            rf"Decay {_label} ($\alpha={_alpha}$)"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    # Evaluation protocol.
    budget_low: int = 50                  # [PAPER] d <= 5
    budget_high: int = 100                # [PAPER] d >= 6
    budget_override: int | None = None    # non-paper smoke/research runs only
    n_initial: int = 1                    # [ASSUMPTION] scrambled Sobol design
    budget_includes_initial: bool = True  # [ASSUMPTION] T is total evaluations
    objective_value_scale: float = 1.0    # sensitivity studies; applied before noise

    # Observation model and GP.
    noise_variance: float = 1e-2          # [PAPER] observation-noise variance
    # None preserves the paper convention that GP alpha equals observation
    # noise. A numeric override isolates nugget smoothing from data noise.
    gp_nugget_variance: float | None = None
    gp_backend: str = "sklearn"           # [PAPER]
    gp_input_space: str = "native"        # [ASSUMPTION]
    normalize_y: bool = False             # [ASSUMPTION]
    center_y: bool = False                # subtract mean without rescaling noise
    matern_nu: float = 1.5                # [ASSUMPTION] sklearn Matern default
    initial_lengthscale: float = 1.0      # [ASSUMPTION] sklearn Matern default
    initial_signal_variance: float = 1.0  # fixed by the bare Matern kernel
    learn_signal_variance: bool = False   # [ASSUMPTION]
    lengthscale_bounds: tuple[float, float] = (1e-5, 1e5)
    signal_variance_bounds: tuple[float, float] = (1e-3, 1e3)
    hyperopt_interval: int = 1            # [ASSUMPTION] fit on every GP update
    sklearn_n_restarts_optimizer: int = 0 # GP kernel fit; not acquisition starts
    jitter: float = 1e-8

    # Acquisition and recommendation optimization.
    optimizer_restarts: int = 20          # [PAPER] random L-BFGS-B starts
    recommendation_restarts: int = 20     # [ASSUMPTION]
    recommendation_include_observed: bool = True # numerical argmax safeguard
    optimizer_maxiter: int = 60           # [ASSUMPTION]
    optimizer_start_design: str = "uniform" # [PAPER] random restarts
    ts_candidates: int = 1024             # [ASSUMPTION] discretization X_t
    ts_candidate_design: str = "uniform"  # [ASSUMPTION]
    ts_candidate_schedule: str = "fixed"
    ts_candidate_min: int = 128
    ts_candidate_growth_scale: float = 64.0
    ts_candidate_max: int = 1024
    # Random-feature GP-TS. Kept acquisition-specific so adding the method does
    # not invalidate cached results from the existing discretized GP-TS.
    ts_rff_features: int = 1000            # Do et al. (2024) reference scale
    variance_candidates: int = 512        # diagnostic only

    # MES-G approximation. These settings are acquisition-specific and are
    # tracked separately from the common BO-protocol hash.
    mes_num_max_samples: int = 100         # Wang & Jegelka (2017) experiments
    mes_num_representer_points: int = 10000 # original MES-G reference code
    mes_representer_design: str = "uniform"
    mes_min_std: float = 1e-10

    # One-step Knowledge Gradient on finite Sobol sets. The expectation over a
    # scalar fantasy observation uses Gauss-Hermite quadrature, avoiding a
    # PyTorch/BoTorch dependency while retaining the current exact GP.
    kg_num_candidates: int = 256
    kg_num_representer_points: int = 512
    kg_num_fantasies: int = 32
    kg_design: str = "sobol"
    kg_candidate_batch_size: int = 32
    kg_min_variance: float = 1e-12

    # beta schedules.
    beta_constant: float = 2.0             # [PAPER]
    increasing_beta_scale: float = 1.0     # [ASSUMPTION]
    logarithmic_beta_scale: float = 1.0

    def resolved_gp_nugget_variance(self) -> float:
        if self.gp_nugget_variance is None:
            return float(self.noise_variance)
        return float(self.gp_nugget_variance)

    def paper_horizon(self, dim: int) -> int:
        if self.budget_override is not None:
            return self.budget_override
        return self.budget_low if dim <= 5 else self.budget_high

    def total_evaluations(self, dim: int) -> int:
        horizon = self.paper_horizon(dim)
        return horizon if self.budget_includes_initial else horizon + self.n_initial

    def bo_steps(self, dim: int) -> int:
        return self.total_evaluations(dim) - self.n_initial


def paper_config() -> ExperimentConfig:
    """Return the single source of truth for paper-condition experiments."""
    return ExperimentConfig()


def smoke_config() -> ExperimentConfig:
    """Small end-to-end run using the same sklearn/SciPy code path."""
    return replace(
        paper_config(),
        budget_override=6,
        n_initial=2,
        hyperopt_interval=0,
        optimizer_restarts=2,
        recommendation_restarts=2,
        optimizer_maxiter=10,
        ts_candidates=32,
        ts_rff_features=64,
        variance_candidates=64,
        mes_num_max_samples=8,
        mes_num_representer_points=128,
        kg_num_candidates=16,
        kg_num_representer_points=32,
        kg_num_fantasies=8,
        kg_candidate_batch_size=8,
    )


def research_config() -> ExperimentConfig:
    """Configurable experiments that start from the paper environment."""
    return paper_config()


def high_dimensional_config() -> ExperimentConfig:
    """Numerically calibrated exact-GP profile for controlled 20D--30D runs."""
    return replace(
        research_config(),
        gp_input_space="unit",
        normalize_y=False,
        center_y=True,
        initial_lengthscale=0.5,
        learn_signal_variance=True,
        lengthscale_bounds=(1e-2, 20.0),
        hyperopt_interval=5,
        sklearn_n_restarts_optimizer=2,
        optimizer_start_design="sobol",
        ts_candidates=2048,
        ts_candidate_design="sobol",
        variance_candidates=1024,
        mes_representer_design="sobol",
    )


def sklearn_literal_config(base: ExperimentConfig | None = None) -> ExperimentConfig:
    """Backward-compatible alias for the now-canonical paper GP profile."""
    if base is None:
        return paper_config()
    return replace(
        base,
        n_initial=1,
        gp_backend="sklearn",
        gp_input_space="native",
        normalize_y=False,
        matern_nu=1.5,
        initial_lengthscale=1.0,
        learn_signal_variance=False,
        lengthscale_bounds=(1e-5, 1e5),
        hyperopt_interval=1,
        sklearn_n_restarts_optimizer=0,
        optimizer_restarts=20,
        optimizer_start_design="uniform",
        ts_candidates=1024,
        ts_candidate_design="uniform",
    )
