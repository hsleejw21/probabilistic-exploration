"""Acquisition registry and shared optimization implementations.

To add a continuous acquisition, implement a selector that calls
``maximize_continuous_acquisition`` and register it in ``ACQUISITIONS``.  It
will then share the paper's random-start SciPy L-BFGS-B implementation rather
than introducing a method-specific optimizer by accident.

EI and LogEI use the same best-observed incumbent and the same optimizer.  The
LogEI helper is a NumPy/SciPy port of the stable analytic reformulation in
Ament et al. (2023) and BoTorch's MIT-licensed reference implementation; it is
not ``log(EI + epsilon)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, minimize
from scipy.special import erfcx, log_ndtr, ndtr
from scipy.stats import qmc

from .config import ExperimentConfig
from .gp_surrogate import GaussianProcess, SklearnGaussianProcess, stable_cholesky


Array = np.ndarray
GP = GaussianProcess | SklearnGaussianProcess
PosteriorScore = Callable[[Array, Array], Array]
ContinuousValue = Callable[[Array], Array]


@dataclass(frozen=True)
class AcquisitionContext:
    gp: GP
    dim: int
    beta: float
    best_f: float | None
    rng: np.random.Generator
    config: ExperimentConfig
    bo_iteration: int


Selector = Callable[[AcquisitionContext], Array]


@dataclass(frozen=True)
class AcquisitionDefinition:
    name: str
    selector: Selector
    optimization: str
    description: str
    beta_modes: tuple[str, ...]


def beta_value(
    mode: str,
    t: int,
    config: ExperimentConfig,
    observed_y: Array | None = None,
) -> float:
    if mode == "native":
        return float("nan")
    if mode == "constant":
        return float(config.beta_constant)
    if mode == "zero":
        return 0.0
    if mode == "increasing":
        return float(config.increasing_beta_scale * math.sqrt(math.log(t + 1.0)))
    if mode == "logarithmic":
        return float(config.logarithmic_beta_scale * math.log(t + 1.0))
    if mode == "max_abs_observed":
        values = np.asarray(observed_y, dtype=float).ravel()
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("max_abs_observed beta requires finite observations")
        return float(np.max(np.abs(values)))
    raise ValueError(f"unknown beta mode: {mode}")


def sobol_points(n: int, dim: int, seed: int) -> Array:
    if n <= 0:
        return np.empty((0, dim))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    exponent = int(math.ceil(math.log2(n)))
    return sampler.random_base2(exponent)[:n]


def _random_design(
    n: int, dim: int, rng: np.random.Generator, design: str
) -> Array:
    if design == "sobol":
        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        return sobol_points(n, dim, seed)
    if design == "uniform":
        return rng.random((n, dim))
    raise ValueError(f"unknown random design: {design}")


def maximize_continuous_function(
    dim: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
    value_function: ContinuousValue,
    *,
    restarts: int | None = None,
) -> Array:
    """Maximize a vectorized function with SciPy multistart L-BFGS-B.

    The paper profile supplies 20 independent uniform random starting points.
    Every registered continuous acquisition should use this function.
    """
    number_of_starts = config.optimizer_restarts if restarts is None else restarts
    if number_of_starts < 1:
        raise ValueError("L-BFGS-B requires at least one random start")
    starts = _random_design(
        number_of_starts, dim, rng, config.optimizer_start_design
    )

    def negative_value(flat_x: Array) -> float:
        value = np.asarray(
            value_function(np.asarray(flat_x, dtype=float)[None, :]), dtype=float
        ).ravel()[0]
        return -float(value)

    best_x = starts[0].copy()
    best_value = -negative_value(best_x)
    for start in starts:
        result = minimize(
            negative_value,
            start,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * dim,
            options={"maxiter": config.optimizer_maxiter},
        )
        candidate = start
        if result.success and np.all(np.isfinite(result.x)):
            candidate = np.clip(result.x, 0.0, 1.0)
        value = -negative_value(candidate)
        if value > best_value:
            best_x, best_value = candidate.copy(), value
    return best_x


def maximize_continuous_acquisition(
    gp: GP,
    dim: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
    score: PosteriorScore,
    *,
    restarts: int | None = None,
) -> Array:
    """Maximize a posterior score with the shared multistart optimizer."""
    return maximize_continuous_function(
        dim,
        rng,
        config,
        lambda x: score(*gp.predict(x)),
        restarts=restarts,
    )


def maximize_posterior_variance(
    gp: GP,
    dim: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
    *,
    restarts: int = 20,
) -> Array:
    """Select argmax posterior variance for a PE-MVR round."""
    return maximize_continuous_acquisition(
        gp,
        dim,
        rng,
        config,
        lambda _mean, std: std**2,
        restarts=restarts,
    )


def select_ucb(
    context: AcquisitionContext,
) -> Array:
    return maximize_continuous_acquisition(
        context.gp,
        context.dim,
        context.rng,
        context.config,
        lambda mean, std: mean + context.beta * std,
    )


def select_ts(
    context: AcquisitionContext,
) -> Array:
    """Sample GP(mu, beta*k) on the paper-defined discretization X_t."""
    number_of_candidates = ts_candidate_count(
        context.config, context.bo_iteration
    )
    candidates = _random_design(
        number_of_candidates,
        context.dim,
        context.rng,
        context.config.ts_candidate_design,
    )
    mean, covariance = context.gp.predict(candidates, return_covariance=True)
    factor, _ = stable_cholesky(covariance, context.config.jitter)
    sample = mean + math.sqrt(context.beta) * (
        factor @ context.rng.standard_normal(len(candidates))
    )
    return candidates[int(np.argmax(sample))].copy()


def ts_candidate_count(config: ExperimentConfig, bo_iteration: int) -> int:
    """Return the fixed or progressively denser GP-TS discretization size."""
    if config.ts_candidate_schedule == "fixed":
        return int(config.ts_candidates)
    if config.ts_candidate_schedule == "sqrt":
        proposed = int(
            math.ceil(config.ts_candidate_growth_scale * math.sqrt(bo_iteration))
        )
        return int(
            np.clip(proposed, config.ts_candidate_min, config.ts_candidate_max)
        )
    raise ValueError(
        f"unknown ts_candidate_schedule: {config.ts_candidate_schedule}"
    )


def _sample_matern_frequencies(
    number_of_features: int,
    dim: int,
    lengthscale: float,
    nu: float,
    rng: np.random.Generator,
) -> Array:
    """Sample the Matérn spectral density as a multivariate Student-t."""
    if number_of_features < 1:
        raise ValueError("ts_rff_features must be positive")
    if not np.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("the fitted Matérn lengthscale must be positive")
    degrees_of_freedom = 2.0 * float(nu)
    gaussian = rng.standard_normal((number_of_features, dim))
    chi_square = rng.chisquare(degrees_of_freedom, size=number_of_features)
    return gaussian / (
        float(lengthscale)
        * np.sqrt(chi_square / degrees_of_freedom)[:, None]
    )


def select_ts_rff(context: AcquisitionContext) -> Array:
    """Maximize one continuous Matérn random-feature posterior sample path.

    The exact fitted GP supplies the posterior mean and data-space correction.
    Random Fourier features approximate its prior residual.  As the feature
    count increases this converges to a draw from GP(mu, beta*k_posterior), the
    same target distribution used by ``select_ts``, without restricting the
    maximizer to a finite candidate set.
    """
    gp = context.gp
    training_inputs = np.asarray(gp.training_inputs, dtype=float).reshape(
        -1, context.dim
    )
    if not len(training_inputs):
        raise RuntimeError("random-feature GP-TS requires a fitted GP")

    feature_count = context.config.ts_rff_features
    frequencies = _sample_matern_frequencies(
        feature_count,
        context.dim,
        float(gp.lengthscale),
        context.config.matern_nu,
        context.rng,
    )
    phases = context.rng.uniform(0.0, 2.0 * math.pi, size=feature_count)
    weights = context.rng.standard_normal(feature_count)
    feature_scale = math.sqrt(2.0 * float(gp.signal_variance) / feature_count)

    def prior_sample(unit_x: Array) -> Array:
        kernel_x = gp.kernel_input_coordinates(unit_x)
        features = feature_scale * np.cos(kernel_x @ frequencies.T + phases)
        return features @ weights

    prior_at_training = prior_sample(training_inputs)
    synthetic_noise = context.rng.normal(
        scale=math.sqrt(float(gp.internal_noise_variance)),
        size=len(training_inputs),
    )
    cholesky = gp.training_cholesky
    correction_weights = solve_triangular(
        cholesky.T,
        solve_triangular(
            cholesky, prior_at_training + synthetic_noise, lower=True
        ),
        lower=False,
    )
    output_scale = float(gp.posterior_output_scale)

    def sampled_path(unit_x: Array) -> Array:
        exact_mean, _ = gp.predict(unit_x)
        cross_covariance = gp.fitted_kernel_covariance(unit_x, training_inputs)
        posterior_residual = prior_sample(unit_x) - (
            cross_covariance @ correction_weights
        )
        return exact_mean + math.sqrt(context.beta) * output_scale * posterior_residual

    return maximize_continuous_function(
        context.dim,
        context.rng,
        context.config,
        sampled_path,
    )


def select_random_search(context: AcquisitionContext) -> Array:
    """Select one IID-uniform point for a diagnostic random-search baseline."""
    return context.rng.random(context.dim)


_LOG_2PI_OVER_2 = 0.5 * math.log(2.0 * math.pi)
_LOG_PI_OVER_2_OVER_2 = 0.5 * math.log(math.pi / 2.0)
_NEG_INV_SQRT_TWO = -(2.0**-0.5)


def expected_improvement(mean: Array, std: Array, best_f: float) -> Array:
    """Classic analytic EI for a maximization problem.

    This deliberately evaluates EI in ordinary floating-point space so the
    numerical behavior being studied is not hidden by a log-domain fallback.
    """
    mean_array = np.asarray(mean, dtype=float)
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-12)
    improvement = mean_array - float(best_f)
    scaled = improvement / sigma
    density = np.exp(-0.5 * scaled**2) / math.sqrt(2.0 * math.pi)
    return sigma * (density + scaled * ndtr(scaled))


def _log1mexp(value: Array) -> Array:
    """Return log(1 - exp(value)) for non-positive ``value``."""
    array = np.minimum(np.asarray(value, dtype=float), 0.0)
    threshold = -math.log(2.0)
    result = np.empty_like(array)
    far = array < threshold
    if np.any(far):
        result[far] = np.log1p(-np.exp(array[far]))
    if np.any(~far):
        result[~far] = np.log(-np.expm1(array[~far]))
    return result


def _log_ei_helper(scaled: Array) -> Array:
    """Stably compute log(phi(u) + u*Phi(u)) in float64."""
    u = np.asarray(scaled, dtype=float)
    result = np.empty_like(u)
    upper = u > -1.0
    if np.any(upper):
        upper_u = u[upper]
        density = np.exp(-0.5 * upper_u**2) / math.sqrt(2.0 * math.pi)
        result[upper] = np.log(density + upper_u * ndtr(upper_u))

    lower = ~upper
    if np.any(lower):
        lower_u = u[lower]
        log_phi = -0.5 * lower_u**2 - _LOG_2PI_OVER_2
        extreme = lower_u <= -1e6
        correction = np.empty_like(lower_u)
        if np.any(~extreme):
            moderate = lower_u[~extreme]
            # log(|u| Phi(u) / phi(u)); erfcx avoids Gaussian-tail underflow.
            w = (
                np.log(erfcx(_NEG_INV_SQRT_TWO * moderate) * np.abs(moderate))
                + _LOG_PI_OVER_2_OVER_2
            )
            correction[~extreme] = _log1mexp(w)
        if np.any(extreme):
            correction[extreme] = -2.0 * np.log(np.abs(lower_u[extreme]))
        result[lower] = log_phi + correction
    return result


def log_expected_improvement(mean: Array, std: Array, best_f: float) -> Array:
    """Numerically stable analytic LogEI for a maximization problem."""
    mean_array = np.asarray(mean, dtype=float)
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-12)
    scaled = (mean_array - float(best_f)) / sigma
    return np.log(sigma) + _log_ei_helper(scaled)


def _require_best_f(context: AcquisitionContext) -> float:
    if context.best_f is None or not np.isfinite(context.best_f):
        raise ValueError("EI-family acquisitions require a finite best_f")
    return float(context.best_f)


def select_ei(context: AcquisitionContext) -> Array:
    best_f = _require_best_f(context)
    return maximize_continuous_acquisition(
        context.gp,
        context.dim,
        context.rng,
        context.config,
        lambda mean, std: expected_improvement(mean, std, best_f),
    )


def select_logei(context: AcquisitionContext) -> Array:
    best_f = _require_best_f(context)
    return maximize_continuous_acquisition(
        context.gp,
        context.dim,
        context.rng,
        context.config,
        lambda mean, std: log_expected_improvement(mean, std, best_f),
    )


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _independent_max_log_cdf(value: float, mean: Array, std: Array) -> float:
    """Log CDF of the maximum under the MES-G mean-field approximation."""
    standardized = (float(value) - mean) / std
    return float(np.sum(log_ndtr(standardized)))


def fit_gumbel_maximum(
    mean: Array,
    std: Array,
    *,
    min_std: float = 1e-10,
) -> tuple[float, float]:
    """Fit a Gumbel maximum distribution by quartile matching.

    This follows Wang and Jegelka (2017): the continuous domain has already
    been replaced by representer points, their posterior values are treated as
    independent, and the 0.25/0.50/0.75 maximum-CDF quantiles determine the
    Gumbel location and scale.
    """
    mean_array = np.asarray(mean, dtype=float).ravel()
    std_array = np.maximum(np.asarray(std, dtype=float).ravel(), min_std)
    if mean_array.size == 0 or mean_array.shape != std_array.shape:
        raise ValueError("mean and std must be non-empty vectors of equal shape")
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
        raise ValueError("Gumbel fitting requires finite posterior moments")

    lower = float(np.min(mean_array - 8.0 * std_array))
    upper = float(np.max(mean_array + 8.0 * std_array))

    def quantile(probability: float) -> float:
        target = math.log(probability)
        return float(
            brentq(
                lambda value: _independent_max_log_cdf(
                    value, mean_array, std_array
                )
                - target,
                lower,
                upper,
                xtol=1e-10,
                rtol=1e-12,
                maxiter=200,
            )
        )

    lower_quartile, median, upper_quartile = (
        quantile(0.25),
        quantile(0.50),
        quantile(0.75),
    )
    transformed_lower = -math.log(-math.log(0.25))
    transformed_upper = -math.log(-math.log(0.75))
    scale = (upper_quartile - lower_quartile) / (
        transformed_upper - transformed_lower
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = max((upper_quartile - lower_quartile) / 2.0, min_std)
    location = median - scale * (-math.log(-math.log(0.50)))
    return float(location), float(scale)


def sample_gumbel_maxima(
    gp: GP,
    dim: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> Array:
    """Draw approximate posterior maximum values for MES-G once per BO step."""
    if config.mes_num_max_samples < 1 or config.mes_num_representer_points < 2:
        raise ValueError("MES-G sample and representer counts must be positive")
    representers = _random_design(
        config.mes_num_representer_points,
        dim,
        rng,
        config.mes_representer_design,
    )
    training_inputs = np.asarray(
        getattr(gp, "training_inputs", np.empty((0, dim))), dtype=float
    )
    if training_inputs.size:
        representers = np.vstack((representers, training_inputs.reshape(-1, dim)))
    mean, std = gp.predict(representers)
    location, scale = fit_gumbel_maximum(
        mean,
        std,
        min_std=config.mes_min_std,
    )
    uniforms = np.clip(
        rng.random(config.mes_num_max_samples),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    return location - scale * np.log(-np.log(uniforms))


def max_value_entropy_search(
    mean: Array,
    std: Array,
    maxima: Array,
    *,
    min_std: float = 1e-10,
) -> Array:
    """Original single-query analytic MES objective for maximization.

    The negative Gaussian tail uses an asymptotic expression to avoid the
    cancellation in gamma*phi(gamma)/(2*Phi(gamma)) - log(Phi(gamma)).
    """
    mean_array = np.asarray(mean, dtype=float).reshape(-1, 1)
    std_array = np.maximum(
        np.asarray(std, dtype=float).reshape(-1, 1), float(min_std)
    )
    maximum_array = np.asarray(maxima, dtype=float).reshape(1, -1)
    if maximum_array.size == 0 or not np.all(np.isfinite(maximum_array)):
        raise ValueError("MES requires finite maximum-value samples")
    gamma = (maximum_array - mean_array) / std_array
    values = np.empty_like(gamma)

    regular = gamma >= -10.0
    if np.any(regular):
        regular_gamma = gamma[regular]
        log_cdf = log_ndtr(regular_gamma)
        log_pdf = -0.5 * regular_gamma**2 - _LOG_SQRT_2PI
        inverse_mills = np.exp(log_pdf - log_cdf)
        values[regular] = 0.5 * regular_gamma * inverse_mills - log_cdf

    tail = ~regular
    if np.any(tail):
        magnitude = -gamma[tail]
        inverse_square = 1.0 / magnitude**2
        values[tail] = (
            np.log(magnitude)
            + _LOG_SQRT_2PI
            - 0.5
            + 2.0 * inverse_square
            - 7.5 * inverse_square**2
        )

    return np.mean(np.maximum(values, 0.0), axis=1)


def select_mes_gumbel(context: AcquisitionContext) -> Array:
    maxima = sample_gumbel_maxima(
        context.gp,
        context.dim,
        context.rng,
        context.config,
    )
    return maximize_continuous_acquisition(
        context.gp,
        context.dim,
        context.rng,
        context.config,
        lambda mean, std: max_value_entropy_search(
            mean,
            std,
            maxima,
            min_std=context.config.mes_min_std,
        ),
    )


# The registry is the only list consumed by the CLI and trial runner.  Future
# acquisition functions should be added here with their optimizer made explicit.
ACQUISITIONS: dict[str, AcquisitionDefinition] = {
    "random_search": AcquisitionDefinition(
        name="random_search",
        selector=select_random_search,
        optimization="iid_uniform_sampling",
        description="pure IID-uniform random-search diagnostic baseline",
        beta_modes=("native",),
    ),
    "ucb": AcquisitionDefinition(
        name="ucb",
        selector=select_ucb,
        optimization="scipy_multistart_l_bfgs_b",
        description="posterior mean + beta * posterior standard deviation",
        beta_modes=(
            "constant",
            "zero",
            "increasing",
            "logarithmic",
            "max_abs_observed",
        ),
    ),
    "ts": AcquisitionDefinition(
        name="ts",
        selector=select_ts,
        optimization="posterior_sample_on_discretization",
        description="sample from GP(mu, beta*k) and maximize over X_t",
        beta_modes=("constant", "increasing", "logarithmic"),
    ),
    "ts_rff": AcquisitionDefinition(
        name="ts_rff",
        selector=select_ts_rff,
        optimization="continuous_rff_sample_plus_scipy_multistart_l_bfgs_b",
        description="continuous Matérn random-feature posterior sample path",
        beta_modes=("constant", "increasing"),
    ),
    "ei": AcquisitionDefinition(
        name="ei",
        selector=select_ei,
        optimization="scipy_multistart_l_bfgs_b",
        description="classic analytic expected improvement over best observed y",
        beta_modes=("native",),
    ),
    "logei": AcquisitionDefinition(
        name="logei",
        selector=select_logei,
        optimization="scipy_multistart_l_bfgs_b",
        description="stable analytic log expected improvement over best observed y",
        beta_modes=("native",),
    ),
    "mes_gumbel": AcquisitionDefinition(
        name="mes_gumbel",
        selector=select_mes_gumbel,
        optimization="gumbel_max_sampling_plus_scipy_multistart_l_bfgs_b",
        description="analytic max-value entropy search with Gumbel maximum samples",
        beta_modes=("native",),
    ),
}


def acquisition_names() -> tuple[str, ...]:
    return tuple(ACQUISITIONS)


def acquisition_beta_modes(name: str) -> tuple[str, ...]:
    try:
        return ACQUISITIONS[name].beta_modes
    except KeyError as error:
        raise ValueError(f"unknown acquisition: {name}") from error


def acquisition_metadata() -> dict[str, dict[str, object]]:
    return {
        name: {
            "optimization": definition.optimization,
            "description": definition.description,
            "beta_modes": list(definition.beta_modes),
        }
        for name, definition in ACQUISITIONS.items()
    }


def acquisition_settings(
    name: str, config: ExperimentConfig
) -> dict[str, object]:
    """Return result-hashed settings that affect a named acquisition."""
    if name == "mes_gumbel":
        return {
            "implementation": "wang_jegelka_2017_gumbel_v1",
            "num_max_samples": config.mes_num_max_samples,
            "num_representer_points": config.mes_num_representer_points,
            "representer_design": config.mes_representer_design,
            "min_std": config.mes_min_std,
            "gumbel_fit_quantiles": [0.25, 0.50, 0.75],
            "outer_optimizer": "scipy_multistart_l_bfgs_b",
            "outer_optimizer_restarts": config.optimizer_restarts,
        }
    if name == "ts_rff":
        return {
            "implementation": "decoupled_matern_rff_posterior_path_v1",
            "num_features": config.ts_rff_features,
            "posterior_mean": "exact_sklearn_gp",
            "posterior_correction": "exact_kernel_data_space",
            "outer_optimizer": "scipy_multistart_l_bfgs_b",
            "outer_optimizer_restarts": config.optimizer_restarts,
            "outer_optimizer_start_design": config.optimizer_start_design,
        }
    if name == "ts":
        return {
            "implementation": "joint_gp_sample_on_discretization_v2",
            "candidate_design": config.ts_candidate_design,
            "candidate_schedule": config.ts_candidate_schedule,
            "fixed_candidates": config.ts_candidates,
            "candidate_min": config.ts_candidate_min,
            "candidate_growth_scale": config.ts_candidate_growth_scale,
            "candidate_max": config.ts_candidate_max,
        }
    definition = ACQUISITIONS[name]
    return {
        "implementation": definition.description,
        "optimization": definition.optimization,
        "beta_modes": list(definition.beta_modes),
    }


def maximize_acquisition(
    acquisition: str,
    gp: GP,
    dim: int,
    beta: float,
    best_f: float | None,
    rng: np.random.Generator,
    config: ExperimentConfig,
    bo_iteration: int = 1,
) -> Array:
    try:
        definition = ACQUISITIONS[acquisition]
    except KeyError as error:
        raise ValueError(f"unknown acquisition: {acquisition}") from error
    return definition.selector(
        AcquisitionContext(
            gp=gp,
            dim=dim,
            beta=beta,
            best_f=best_f,
            rng=rng,
            config=config,
            bo_iteration=bo_iteration,
        )
    )


def recommend_posterior_mean(
    gp: GP,
    dim: int,
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> Array:
    continuous = maximize_continuous_acquisition(
        gp,
        dim,
        rng,
        config,
        lambda mean, std: mean,
        restarts=config.recommendation_restarts,
    )
    if not config.recommendation_include_observed:
        return continuous

    # A numerical optimizer for max_x mu(x) should never return a point whose
    # posterior mean is lower than that of an already evaluated point.  This
    # safeguard is especially important in high-dimensional multimodal
    # problems, where a small collection of random L-BFGS-B starts can miss a
    # narrow learned basin and converge to a boundary local optimum instead.
    training_inputs = np.asarray(
        getattr(gp, "training_inputs", np.empty((0, dim))), dtype=float
    ).reshape(-1, dim)
    if not len(training_inputs):
        return continuous
    observed_mean, _ = gp.predict(training_inputs)
    best_observed = training_inputs[int(np.argmax(observed_mean))]
    continuous_mean, _ = gp.predict(continuous[None, :])
    if float(np.max(observed_mean)) > float(continuous_mean[0]):
        return best_observed.copy()
    return continuous
