"""One fully logged probabilistic-exploration Bayesian-optimization trial."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .acquisitions import (
    beta_value,
    maximize_acquisition,
    maximize_posterior_variance,
    recommend_posterior_mean,
    sobol_points,
)
from .benchmarks import BENCHMARKS, Benchmark
from .config import ExperimentConfig
from .exploration import (
    GreedyPackingExplorer,
    decay_exponent,
    exploration_probability,
    exploration_rule,
)
from .gp_surrogate import build_gaussian_process


Array = np.ndarray


@dataclass(frozen=True)
class TrialSpec:
    objective: str
    acquisition: str
    beta_mode: str
    policy: str
    seed: int

    def slug(self) -> str:
        return (
            f"{self.objective}__{self.acquisition}__{self.beta_mode}__"
            f"{self.policy}__seed{self.seed:04d}"
        )


HISTORY_FIELDS = [
    "objective",
    "dimension",
    "acquisition",
    "beta_mode",
    "policy",
    "seed",
    "iteration",
    "bo_iteration",
    "event",
    "exploration_rule",
    "decay_exponent",
    "exploration_probability",
    "query_preselection_posterior_std",
    "cumulative_pe_calls",
    "cumulative_mode_transitions",
    "beta",
    "query_x_unit",
    "query_x_native",
    "latent_query_value",
    "observed_value",
    "recommendation_x_unit",
    "recommendation_x_native",
    "recommendation_value",
    "inference_regret",
    "recommendation_posterior_mean",
    "recommendation_posterior_std",
    "recommendation_nearest_observed_distance",
    "best_queried_latent_regret",
    "max_posterior_std",
    "mean_posterior_std",
    "q90_posterior_std",
    "lengthscale",
    "signal_variance",
    "used_jitter",
    "selection_time_seconds",
    "evaluation_time_seconds",
    "gp_fit_time_seconds",
    "recommendation_time_seconds",
    "diagnostic_time_seconds",
    "step_wall_time_seconds",
    "cumulative_wall_time_seconds",
]


def _json_vector(vector: Array) -> str:
    return json.dumps(
        [float(value) for value in np.asarray(vector).ravel()], separators=(",", ":")
    )


def _stream_seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def expected_rows(spec: TrialSpec, config: ExperimentConfig) -> int:
    return config.total_evaluations(BENCHMARKS[spec.objective].dim)


def should_optimize_hyperparameters(
    n_observed: int, config: ExperimentConfig
) -> bool:
    """Optimize after initialization, then at BO-relative intervals."""
    if config.hyperopt_interval <= 0 or n_observed < config.n_initial:
        return False
    return (
        n_observed == config.n_initial
        or (n_observed - config.n_initial) % config.hyperopt_interval == 0
    )


def run_trial(spec: TrialSpec, config: ExperimentConfig) -> list[dict[str, object]]:
    benchmark: Benchmark = BENCHMARKS[spec.objective]
    objective_scale = float(config.objective_value_scale)
    if not np.isfinite(objective_scale) or objective_scale <= 0.0:
        raise ValueError("objective_value_scale must be finite and positive")
    scaled_f_star = objective_scale * benchmark.f_star
    total_evaluations = config.total_evaluations(benchmark.dim)
    if config.n_initial < 1 or config.n_initial >= total_evaluations:
        raise ValueError("n_initial must be positive and smaller than total evaluations")

    # Separate streams preserve common initial designs and noise draws across
    # policies while preventing different code paths from shifting later noise.
    streams = np.random.SeedSequence(spec.seed).spawn(9)
    initial_seed = _stream_seed(streams[0])
    noise_rng = np.random.default_rng(streams[1])
    decision_rng = np.random.default_rng(streams[2])
    exploration_rng = np.random.default_rng(streams[3])
    acquisition_rng = np.random.default_rng(streams[4])
    recommendation_rng = np.random.default_rng(streams[5])
    diagnostic_seed = _stream_seed(streams[6])
    greedy_packing_seed = _stream_seed(streams[7])

    initial_design = sobol_points(config.n_initial, benchmark.dim, initial_seed)
    diagnostic_design = sobol_points(
        config.variance_candidates, benchmark.dim, diagnostic_seed
    )
    greedy_packing = None
    if exploration_rule(spec.policy) == "greedy_packing":
        greedy_packing = GreedyPackingExplorer(
            dim=benchmark.dim,
            seed=greedy_packing_seed,
            max_iteration=total_evaluations - config.n_initial,
            grid_initial=config.greedy_packing_grid_initial,
            grid_growth=config.greedy_packing_grid_growth,
            distance_batch_size=config.greedy_packing_distance_batch_size,
        )

    gp = build_gaussian_process(
        config,
        np.asarray(benchmark.lower, dtype=float),
        np.asarray(benchmark.upper, dtype=float),
    )
    x_observed: list[Array] = []
    y_observed: list[float] = []
    history: list[dict[str, object]] = []
    cumulative_time = 0.0
    cumulative_pe_calls = 0
    cumulative_mode_transitions = 0
    previous_bo_mode: str | None = None

    def append_observation(
        unit_x: Array,
        *,
        event: str,
        p_explore: float,
        beta: float,
        selection_time: float,
        query_preselection_std: float,
    ) -> None:
        nonlocal cumulative_time, cumulative_pe_calls
        nonlocal cumulative_mode_transitions, previous_bo_mode

        if event != "initial":
            current_mode = "pe" if event.endswith("_exploration") else "acquisition"
            if current_mode == "pe":
                cumulative_pe_calls += 1
            if previous_bo_mode is not None and current_mode != previous_bo_mode:
                cumulative_mode_transitions += 1
            previous_bo_mode = current_mode

        started = time.perf_counter()
        latent = objective_scale * float(
            benchmark.evaluate_unit(unit_x[None, :])[0]
        )
        noisy = latent + float(
            noise_rng.normal(scale=math.sqrt(config.noise_variance))
        )
        evaluation_time = time.perf_counter() - started

        x_observed.append(np.asarray(unit_x, dtype=float).copy())
        y_observed.append(noisy)
        n_observed = len(x_observed)
        optimize_hyperparameters = should_optimize_hyperparameters(
            n_observed, config
        )

        started = time.perf_counter()
        gp.fit(
            np.vstack(x_observed),
            np.asarray(y_observed),
            optimize_hyperparameters=optimize_hyperparameters,
        )
        gp_fit_time = time.perf_counter() - started

        started = time.perf_counter()
        recommendation = recommend_posterior_mean(
            gp, benchmark.dim, recommendation_rng, config
        )
        recommendation_value = objective_scale * float(
            benchmark.evaluate_unit(recommendation[None, :])[0]
        )
        recommendation_mean, recommendation_std = gp.predict(
            recommendation[None, :]
        )
        nearest_observed_distance = float(
            np.min(
                np.linalg.norm(
                    np.vstack(x_observed) - recommendation[None, :], axis=1
                )
            )
        )
        best_queried_latent = max(
            [float(row["latent_query_value"]) for row in history] + [latent]
        )
        recommendation_time = time.perf_counter() - started

        started = time.perf_counter()
        if len(diagnostic_design):
            _, diagnostic_std = gp.predict(diagnostic_design)
            max_std = float(np.max(diagnostic_std))
            mean_std = float(np.mean(diagnostic_std))
            q90_std = float(np.quantile(diagnostic_std, 0.9))
        else:
            max_std = float("nan")
            mean_std = float("nan")
            q90_std = float("nan")
        diagnostic_time = time.perf_counter() - started

        step_time = (
            selection_time
            + evaluation_time
            + gp_fit_time
            + recommendation_time
            + diagnostic_time
        )
        cumulative_time += step_time
        history.append(
            {
                "objective": benchmark.name,
                "dimension": benchmark.dim,
                "acquisition": spec.acquisition,
                "beta_mode": spec.beta_mode,
                "policy": spec.policy,
                "seed": spec.seed,
                "iteration": n_observed,
                "bo_iteration": max(n_observed - config.n_initial, 0),
                "event": event,
                "exploration_rule": exploration_rule(spec.policy),
                "decay_exponent": decay_exponent(spec.policy),
                "exploration_probability": p_explore,
                "query_preselection_posterior_std": query_preselection_std,
                "cumulative_pe_calls": cumulative_pe_calls,
                "cumulative_mode_transitions": cumulative_mode_transitions,
                "beta": beta,
                "query_x_unit": _json_vector(unit_x),
                "query_x_native": _json_vector(benchmark.from_unit(unit_x)[0]),
                "latent_query_value": latent,
                "observed_value": noisy,
                "recommendation_x_unit": _json_vector(recommendation),
                "recommendation_x_native": _json_vector(
                    benchmark.from_unit(recommendation)[0]
                ),
                "recommendation_value": recommendation_value,
                "inference_regret": max(
                    scaled_f_star - recommendation_value, 0.0
                ),
                "recommendation_posterior_mean": float(recommendation_mean[0]),
                "recommendation_posterior_std": float(recommendation_std[0]),
                "recommendation_nearest_observed_distance": nearest_observed_distance,
                "best_queried_latent_regret": max(
                    scaled_f_star - best_queried_latent, 0.0
                ),
                "max_posterior_std": max_std,
                "mean_posterior_std": mean_std,
                "q90_posterior_std": q90_std,
                "lengthscale": gp.lengthscale,
                "signal_variance": gp.signal_variance,
                "used_jitter": gp.used_jitter,
                "selection_time_seconds": selection_time,
                "evaluation_time_seconds": evaluation_time,
                "gp_fit_time_seconds": gp_fit_time,
                "recommendation_time_seconds": recommendation_time,
                "diagnostic_time_seconds": diagnostic_time,
                "step_wall_time_seconds": step_time,
                "cumulative_wall_time_seconds": cumulative_time,
            }
        )

    for unit_x in initial_design:
        append_observation(
            unit_x,
            event="initial",
            p_explore=float("nan"),
            beta=float("nan"),
            selection_time=0.0,
            query_preselection_std=float("nan"),
        )

    while len(x_observed) < total_evaluations:
        bo_t = len(x_observed) - config.n_initial + 1
        p_explore = exploration_probability(spec.policy, benchmark.dim, bo_t)
        beta = beta_value(
            spec.beta_mode, bo_t, config, np.asarray(y_observed, dtype=float)
        )
        started = time.perf_counter()
        if decision_rng.random() < p_explore:
            rule = exploration_rule(spec.policy)
            if rule == "mvr":
                unit_x = maximize_posterior_variance(
                    gp, benchmark.dim, exploration_rng, config, restarts=20
                )
                event = "mvr_exploration"
            elif rule == "greedy_packing":
                assert greedy_packing is not None
                unit_x = greedy_packing.select(bo_t, np.vstack(x_observed))
                event = "greedy_packing_exploration"
            else:
                unit_x = exploration_rng.random(benchmark.dim)
                event = "uniform_exploration"
        else:
            unit_x = maximize_acquisition(
                spec.acquisition,
                gp,
                benchmark.dim,
                beta,
                max(y_observed),
                acquisition_rng,
                config,
                bo_iteration=bo_t,
            )
            event = "acquisition"
        _, query_std = gp.predict(np.asarray(unit_x)[None, :])
        selection_time = time.perf_counter() - started
        append_observation(
            unit_x,
            event=event,
            p_explore=p_explore,
            beta=beta,
            selection_time=selection_time,
            query_preselection_std=float(query_std[0]),
        )

    return history
