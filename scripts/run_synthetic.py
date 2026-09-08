#!/usr/bin/env python3
"""Run the canonical probabilistic-exploration BO replication.

Completed trials are cached individually under ``results/trials``. Re-running
the same configuration skips valid trials, so long studies resume safely after
interruption. Different configurations must use different output directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = EXPERIMENT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import numpy as np
import scipy
import sklearn

from probabilistic_exploration.acquisitions import (
    acquisition_beta_modes,
    acquisition_metadata,
    acquisition_names,
    acquisition_settings,
)
from probabilistic_exploration.benchmarks import BENCHMARKS
from probabilistic_exploration.bo import HISTORY_FIELDS, TrialSpec, expected_rows, run_trial
from probabilistic_exploration.config import (
    AVAILABLE_POLICIES,
    FIGURE_SPECS,
    PAPER_ACQUISITIONS,
    POLICIES,
    ExperimentConfig,
    high_dimensional_config,
    paper_config,
    research_config,
    smoke_config,
)
from probabilistic_exploration.exploration import exploration_probability


def _selection(values: Sequence[str], universe: Iterable[str]) -> list[str]:
    if len(values) == 1 and values[0] == "all":
        return list(universe)
    return list(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["smoke", "paper", "research", "high_dimensional"],
        default="smoke",
        help=(
            "paper locks the stated paper conditions; research starts from the "
            "same environment but permits sensitivity overrides"
        ),
    )
    parser.add_argument("--figure", choices=list(FIGURE_SPECS), default=None)
    parser.add_argument("--objectives", nargs="+", default=["all"])
    parser.add_argument("--acquisitions", nargs="+", default=["all"])
    parser.add_argument("--beta-modes", nargs="+", default=["all"])
    parser.add_argument("--policies", nargs="+", default=["all"])
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None, help="Override both paper horizons")
    parser.add_argument("--budget-low", type=int, default=None)
    parser.add_argument("--budget-high", type=int, default=None)
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument(
        "--objective-value-scale",
        type=float,
        default=None,
        help="Multiply latent objective values by this factor before adding noise",
    )
    parser.add_argument(
        "--budget-excludes-initial",
        action="store_true",
        help="Interpret T as BO steps after initialization instead of total evaluations",
    )
    parser.add_argument("--noise-variance", type=float, default=None)
    parser.add_argument(
        "--gp-nugget-variance",
        type=float,
        default=None,
        help="Override GP alpha without changing simulated observation noise",
    )
    parser.add_argument("--gp-backend", choices=["custom", "sklearn"], default=None)
    parser.add_argument("--gp-input-space", choices=["unit", "native"], default=None)
    parser.add_argument(
        "--normalize-y", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--center-y", action=argparse.BooleanOptionalAction, default=None,
        help="Subtract the observed target mean without changing its scale",
    )
    parser.add_argument("--matern-nu", type=float, choices=[1.5, 2.5], default=None)
    parser.add_argument("--initial-lengthscale", type=float, default=None)
    parser.add_argument(
        "--learn-signal-variance", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--lengthscale-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--signal-variance-bounds", type=float, nargs=2, default=None)
    parser.add_argument("--hyperopt-interval", type=int, default=None)
    parser.add_argument("--sklearn-n-restarts-optimizer", type=int, default=None)
    parser.add_argument("--optimizer-restarts", type=int, default=None)
    parser.add_argument("--recommendation-restarts", type=int, default=None)
    parser.add_argument(
        "--recommendation-include-observed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compare the continuous posterior-mean optimizer against all "
            "evaluated inputs before recording the recommendation"
        ),
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    parser.add_argument(
        "--optimizer-start-design", choices=["sobol", "uniform"], default=None
    )
    parser.add_argument("--ts-candidates", type=int, default=None)
    parser.add_argument(
        "--ts-candidate-design", choices=["sobol", "uniform"], default=None
    )
    parser.add_argument("--ts-rff-features", type=int, default=None)
    parser.add_argument(
        "--ts-candidate-schedule", choices=["fixed", "sqrt"], default=None
    )
    parser.add_argument("--ts-candidate-min", type=int, default=None)
    parser.add_argument("--ts-candidate-growth-scale", type=float, default=None)
    parser.add_argument("--ts-candidate-max", type=int, default=None)
    parser.add_argument("--variance-candidates", type=int, default=None)
    parser.add_argument("--mes-num-max-samples", type=int, default=None)
    parser.add_argument("--mes-num-representer-points", type=int, default=None)
    parser.add_argument(
        "--mes-representer-design", choices=["sobol", "uniform"], default=None
    )
    parser.add_argument("--kg-num-candidates", type=int, default=None)
    parser.add_argument("--kg-num-representer-points", type=int, default=None)
    parser.add_argument("--kg-num-fantasies", type=int, default=None)
    parser.add_argument("--kg-design", choices=["sobol", "uniform"], default=None)
    parser.add_argument("--kg-candidate-batch-size", type=int, default=None)
    parser.add_argument("--kg-min-variance", type=float, default=None)
    parser.add_argument("--beta-constant", type=float, default=None)
    parser.add_argument("--increasing-beta-scale", type=float, default=None)
    parser.add_argument("--logarithmic-beta-scale", type=float, default=None)
    parser.add_argument("--jitter", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute selected trial files; does not delete unrelated cached trials",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> tuple[ExperimentConfig, int]:
    if args.profile == "smoke":
        config = smoke_config()
    elif args.profile == "paper":
        config = paper_config()
    elif args.profile == "research":
        config = research_config()
    else:
        config = high_dimensional_config()
    updates: dict[str, object] = {}
    mapping = {
        "budget": "budget_override",
        "budget_low": "budget_low",
        "budget_high": "budget_high",
        "n_initial": "n_initial",
        "objective_value_scale": "objective_value_scale",
        "noise_variance": "noise_variance",
        "gp_nugget_variance": "gp_nugget_variance",
        "gp_backend": "gp_backend",
        "gp_input_space": "gp_input_space",
        "normalize_y": "normalize_y",
        "center_y": "center_y",
        "matern_nu": "matern_nu",
        "initial_lengthscale": "initial_lengthscale",
        "learn_signal_variance": "learn_signal_variance",
        "lengthscale_bounds": "lengthscale_bounds",
        "signal_variance_bounds": "signal_variance_bounds",
        "hyperopt_interval": "hyperopt_interval",
        "sklearn_n_restarts_optimizer": "sklearn_n_restarts_optimizer",
        "optimizer_restarts": "optimizer_restarts",
        "recommendation_restarts": "recommendation_restarts",
        "recommendation_include_observed": "recommendation_include_observed",
        "optimizer_maxiter": "optimizer_maxiter",
        "optimizer_start_design": "optimizer_start_design",
        "ts_candidates": "ts_candidates",
        "ts_candidate_design": "ts_candidate_design",
        "ts_rff_features": "ts_rff_features",
        "ts_candidate_schedule": "ts_candidate_schedule",
        "ts_candidate_min": "ts_candidate_min",
        "ts_candidate_growth_scale": "ts_candidate_growth_scale",
        "ts_candidate_max": "ts_candidate_max",
        "variance_candidates": "variance_candidates",
        "mes_num_max_samples": "mes_num_max_samples",
        "mes_num_representer_points": "mes_num_representer_points",
        "mes_representer_design": "mes_representer_design",
        "kg_num_candidates": "kg_num_candidates",
        "kg_num_representer_points": "kg_num_representer_points",
        "kg_num_fantasies": "kg_num_fantasies",
        "kg_design": "kg_design",
        "kg_candidate_batch_size": "kg_candidate_batch_size",
        "kg_min_variance": "kg_min_variance",
        "beta_constant": "beta_constant",
        "increasing_beta_scale": "increasing_beta_scale",
        "logarithmic_beta_scale": "logarithmic_beta_scale",
        "jitter": "jitter",
    }
    for argument, field in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            if field in {"lengthscale_bounds", "signal_variance_bounds"}:
                value = tuple(value)
            updates[field] = value
    if args.budget_excludes_initial:
        updates["budget_includes_initial"] = False
    config = replace(config, **updates)
    num_seeds = args.num_seeds
    if num_seeds is None:
        num_seeds = 1 if args.profile == "smoke" else 10
    if num_seeds < 1 or args.jobs < 1:
        raise ValueError("num_seeds and jobs must be positive")
    if not math.isfinite(config.objective_value_scale) or config.objective_value_scale <= 0:
        raise ValueError("objective_value_scale must be finite and positive")
    if config.center_y and config.normalize_y:
        raise ValueError("center_y and normalize_y cannot both be enabled")
    if config.ts_rff_features < 1:
        raise ValueError("ts_rff_features must be positive")
    if (
        config.kg_num_candidates < 1
        or config.kg_num_representer_points < 1
        or config.kg_num_fantasies < 2
        or config.kg_candidate_batch_size < 1
        or config.kg_min_variance <= 0.0
    ):
        raise ValueError("invalid KG discretization or fantasy settings")
    if config.noise_variance < 0.0 or config.resolved_gp_nugget_variance() <= 0.0:
        raise ValueError("noise variance must be non-negative and GP nugget positive")
    if (
        config.ts_candidates < 1
        or config.ts_candidate_min < 1
        or config.ts_candidate_max < config.ts_candidate_min
        or config.ts_candidate_growth_scale <= 0.0
    ):
        raise ValueError("invalid TS candidate schedule")
    for name, bounds in (
        ("lengthscale_bounds", config.lengthscale_bounds),
        ("signal_variance_bounds", config.signal_variance_bounds),
    ):
        if len(bounds) != 2 or bounds[0] <= 0 or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must contain two ordered positive values")
    return config, num_seeds


def validate_paper_protocol(
    config: ExperimentConfig,
    num_seeds: int,
    *,
    python_version: tuple[int, int] | None = None,
) -> None:
    """Reject a paper run if any condition stated in Section 5 is violated."""
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    canonical = paper_config()
    checks = {
        "Python version": (version == (3, 11), f"3.11 required, got {version[0]}.{version[1]}"),
        "GP backend": (config.gp_backend == "sklearn", "scikit-learn required"),
        "noise variance": (math.isclose(config.noise_variance, 1e-2), "lambda=1e-2 required"),
        "low-dimensional horizon": (config.budget_low == 50, "T=50 required"),
        "high-dimensional horizon": (config.budget_high == 100, "T=100 required"),
        "horizon override": (config.budget_override is None, "paper runs cannot override T"),
        "acquisition restarts": (config.optimizer_restarts == 20, "20 starts required"),
        "restart design": (config.optimizer_start_design == "uniform", "random uniform starts required"),
        "fixed beta": (math.isclose(config.beta_constant, 2.0), "beta=2 required"),
        "independent repetitions": (num_seeds == 10, "10 seeds required"),
    }
    errors = [f"{name}: {message}" for name, (valid, message) in checks.items() if not valid]
    canonical_values = asdict(canonical)
    actual_values = asdict(config)
    changed_assumptions = [
        name
        for name in canonical_values
        if actual_values[name] != canonical_values[name]
        and name
        not in {
            "budget_low",
            "budget_high",
            "budget_override",
            "noise_variance",
            "gp_backend",
            "optimizer_restarts",
            "optimizer_start_design",
            "beta_constant",
        }
    ]
    if changed_assumptions:
        errors.append(
            "canonical paper assumptions were overridden: "
            + ", ".join(changed_assumptions)
        )
    if errors:
        raise RuntimeError(
            "Paper protocol validation failed:\n- " + "\n- ".join(errors)
            + "\nUse --profile research for non-paper sensitivity experiments."
        )


def build_specs(
    args: argparse.Namespace, config: ExperimentConfig, num_seeds: int
) -> list[TrialSpec]:
    objectives = _selection(args.objectives, BENCHMARKS)
    acquisition_universe = (
        PAPER_ACQUISITIONS if args.profile == "paper" else acquisition_names()
    )
    if args.figure is not None:
        acquisitions = [FIGURE_SPECS[args.figure][0]]
        acquisition_mode_pairs = [FIGURE_SPECS[args.figure]]
    else:
        acquisitions = _selection(args.acquisitions, acquisition_universe)
        requested_beta_modes = args.beta_modes
        acquisition_mode_pairs: list[tuple[str, str]] = []
        for acquisition in acquisitions:
            supported = acquisition_beta_modes(acquisition)
            modes = (
                list(supported)
                if len(requested_beta_modes) == 1
                and requested_beta_modes[0] == "all"
                else list(requested_beta_modes)
            )
            unsupported = set(modes) - set(supported)
            if unsupported:
                raise ValueError(
                    f"{acquisition} does not support beta mode(s) "
                    f"{sorted(unsupported)}; supported={list(supported)}"
                )
            acquisition_mode_pairs.extend((acquisition, mode) for mode in modes)
    policies = (
        list(POLICIES)
        if len(args.policies) == 1 and args.policies[0] == "all"
        else list(args.policies)
    )

    checks = [
        (objectives, set(BENCHMARKS), "objective"),
        (acquisitions, set(acquisition_universe), "acquisition"),
        (policies, set(AVAILABLE_POLICIES), "policy"),
    ]
    for values, allowed, label in checks:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown {label}: {sorted(unknown)}")
    for objective in objectives:
        dim = BENCHMARKS[objective].dim
        if config.total_evaluations(dim) <= config.n_initial:
            raise ValueError(
                f"{objective}: total evaluations must exceed n_initial={config.n_initial}"
            )

    return [
        TrialSpec(objective, acquisition, beta_mode, policy, seed)
        for objective in objectives
        for acquisition, beta_mode in acquisition_mode_pairs
        for policy in policies
        for seed in range(args.seed_start, args.seed_start + num_seeds)
    ]


def config_hash(config: ExperimentConfig) -> str:
    # Acquisition-specific settings have their own hash. Excluding them here
    # preserves the hashes of the completed UCB/TS and EI/LogEI artifacts.
    common = {
        name: value
        for name, value in asdict(config).items()
        if not name.startswith("mes_")
        and not name.startswith("kg_")
        and not name.startswith("ts_rff_")
        and name
        not in {
            "ts_candidate_schedule",
            "ts_candidate_min",
            "ts_candidate_growth_scale",
            "ts_candidate_max",
        }
        and not (name == "gp_nugget_variance" and value is None)
        and not (name == "logarithmic_beta_scale" and value == 1.0)
        and not (name == "objective_value_scale" and value == 1.0)
        and not (name == "center_y" and value is False)
    }
    encoded = json.dumps(common, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def acquisition_config_hash(
    config: ExperimentConfig, acquisitions: Sequence[str]
) -> str:
    payload = {
        name: acquisition_settings(name, config)
        for name in sorted(set(acquisitions))
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_trial(path: Path, spec: TrialSpec, config: ExperimentConfig) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != HISTORY_FIELDS:
                return None
            rows = list(reader)
    except (OSError, csv.Error):
        return None
    if len(rows) != expected_rows(spec, config):
        return None
    expected = {
        "objective": spec.objective,
        "acquisition": spec.acquisition,
        "beta_mode": spec.beta_mode,
        "policy": spec.policy,
        "seed": str(spec.seed),
    }
    if any(any(row[key] != value for key, value in expected.items()) for row in rows):
        return None
    return rows


def trial_path(trials_dir: Path, spec: TrialSpec) -> Path:
    return trials_dir / f"{spec.slug()}.csv"


def _load_all_valid_trials(trials_dir: Path, config: ExperimentConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(trials_dir.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != HISTORY_FIELDS:
                    continue
                trial_rows = list(reader)
            if not trial_rows:
                continue
            first = trial_rows[0]
            spec = TrialSpec(
                first["objective"],
                first["acquisition"],
                first["beta_mode"],
                first["policy"],
                int(first["seed"]),
            )
            if len(trial_rows) == expected_rows(spec, config):
                rows.extend(trial_rows)
        except (OSError, csv.Error, KeyError, ValueError):
            continue
    rows.sort(
        key=lambda row: (
            row["objective"],
            row["acquisition"],
            row["beta_mode"],
            row["policy"],
            int(row["seed"]),
            int(row["iteration"]),
        )
    )
    return rows


def _write_aggregate_history(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    keys = [
        "objective",
        "dimension",
        "acquisition",
        "beta_mode",
        "policy",
        "iteration",
        "bo_iteration",
    ]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    fields = keys + [
        "n_seeds",
        "mean_inference_regret",
        "median_inference_regret",
        "std_inference_regret",
        "sem_inference_regret",
        "q25_inference_regret",
        "q75_inference_regret",
        "min_inference_regret",
        "max_inference_regret",
        "mean_max_posterior_std",
        "mean_mean_posterior_std",
        "mean_q90_posterior_std",
        "mean_cumulative_pe_calls",
        "mean_cumulative_mode_transitions",
        "mean_cumulative_wall_time_seconds",
        "mean_selection_time_seconds",
        "mean_gp_fit_time_seconds",
        "mean_recommendation_time_seconds",
    ]
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items(), key=lambda item: item[0]):
            regrets = np.asarray([float(row["inference_regret"]) for row in group])
            posterior_std = np.asarray(
                [float(row["max_posterior_std"]) for row in group]
            )
            runtimes = np.asarray(
                [float(row["cumulative_wall_time_seconds"]) for row in group]
            )
            sample_std = float(np.std(regrets, ddof=1)) if len(group) > 1 else 0.0
            output = dict(zip(keys, key))
            output.update(
                n_seeds=len(group),
                mean_inference_regret=float(np.mean(regrets)),
                median_inference_regret=float(np.median(regrets)),
                std_inference_regret=sample_std,
                sem_inference_regret=sample_std / math.sqrt(len(group)),
                q25_inference_regret=float(np.quantile(regrets, 0.25)),
                q75_inference_regret=float(np.quantile(regrets, 0.75)),
                min_inference_regret=float(np.min(regrets)),
                max_inference_regret=float(np.max(regrets)),
                mean_max_posterior_std=float(np.mean(posterior_std)),
                mean_mean_posterior_std=float(np.mean(
                    [float(row["mean_posterior_std"]) for row in group]
                )),
                mean_q90_posterior_std=float(np.mean(
                    [float(row["q90_posterior_std"]) for row in group]
                )),
                mean_cumulative_pe_calls=float(np.mean(
                    [float(row["cumulative_pe_calls"]) for row in group]
                )),
                mean_cumulative_mode_transitions=float(np.mean(
                    [float(row["cumulative_mode_transitions"]) for row in group]
                )),
                mean_cumulative_wall_time_seconds=float(np.mean(runtimes)),
                mean_selection_time_seconds=float(
                    np.mean([float(row["selection_time_seconds"]) for row in group])
                ),
                mean_gp_fit_time_seconds=float(
                    np.mean([float(row["gp_fit_time_seconds"]) for row in group])
                ),
                mean_recommendation_time_seconds=float(
                    np.mean(
                        [float(row["recommendation_time_seconds"]) for row in group]
                    )
                ),
            )
            writer.writerow(output)
    os.replace(temporary, path)


def _write_trial_summary(path: Path, rows: list[dict[str, str]]) -> None:
    """Write one endpoint/switching record per objective-policy-seed trial."""
    trial_keys = ["objective", "dimension", "acquisition", "beta_mode", "policy", "seed"]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in trial_keys), []).append(row)
    fields = trial_keys + [
        "exploration_rule", "decay_exponent", "bo_steps",
        "actual_pe_calls", "expected_pe_calls", "realized_pe_fraction",
        "first_pe_round", "last_pe_round", "mode_transitions",
        "final_inference_regret", "final_best_queried_latent_regret",
        "final_max_posterior_std", "final_mean_posterior_std",
        "final_q90_posterior_std", "final_query_preselection_posterior_std",
    ]
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            ordered = sorted(group, key=lambda row: int(row["iteration"]))
            bo_rows = [row for row in ordered if row["event"] != "initial"]
            pe_rows = [row for row in bo_rows if row["event"].endswith("_exploration")]
            endpoint = ordered[-1]
            dim = int(endpoint["dimension"])
            expected = sum(
                exploration_probability(endpoint["policy"], dim, t)
                for t in range(1, len(bo_rows) + 1)
            )
            actual = len(pe_rows)
            output = dict(zip(trial_keys, key))
            output.update({
                "exploration_rule": endpoint["exploration_rule"],
                "decay_exponent": endpoint["decay_exponent"],
                "bo_steps": len(bo_rows),
                "actual_pe_calls": actual,
                "expected_pe_calls": expected,
                "realized_pe_fraction": actual / len(bo_rows) if bo_rows else float("nan"),
                "first_pe_round": int(pe_rows[0]["bo_iteration"]) if pe_rows else "",
                "last_pe_round": int(pe_rows[-1]["bo_iteration"]) if pe_rows else "",
                "mode_transitions": int(endpoint["cumulative_mode_transitions"]),
                "final_inference_regret": endpoint["inference_regret"],
                "final_best_queried_latent_regret": endpoint["best_queried_latent_regret"],
                "final_max_posterior_std": endpoint["max_posterior_std"],
                "final_mean_posterior_std": endpoint["mean_posterior_std"],
                "final_q90_posterior_std": endpoint["q90_posterior_std"],
                "final_query_preselection_posterior_std": endpoint["query_preselection_posterior_std"],
            })
            writer.writerow(output)
    os.replace(temporary, path)


def _base_metadata(
    config: ExperimentConfig,
    digest: str,
    acquisition_digest: str,
    selected_acquisitions: Sequence[str],
) -> dict[str, object]:
    gp_nugget = config.resolved_gp_nugget_variance()
    if config.center_y:
        noise_scaling = (
            "the observed target mean is subtracted before sklearn fitting and "
            "added back to predictions; target scale remains raw"
        )
    elif config.normalize_y:
        noise_scaling = (
            "sklearn normalize_y=True is enabled; consult the experiment profile "
            "when interpreting alpha on the transformed target scale"
        )
    else:
        noise_scaling = (
            "raw targets are passed to sklearn with normalize_y=False and "
            f"GP alpha={gp_nugget:g}"
        )
    return {
        "paper": "Probabilistic Exploration for Efficient Bayesian Optimization",
        "schema_version": 6,
        "config_hash": digest,
        "acquisition_config_hash": acquisition_digest,
        "experiment_config": asdict(config),
        "replication_notes": {
            "paper_values": "Python 3.11, sklearn GP, scipy L-BFGS-B, 20 random starts, noise, horizons, policies, fixed beta, benchmarks, and 10 repetitions",
            "assumptions": "Matérn nu, initialization, hyperparameter fitting, increasing-beta constant, TS grid, recommendation optimizer, and domains",
            "noise_scaling": noise_scaling,
            "observation_noise_variance": config.noise_variance,
            "gp_nugget_variance": gp_nugget,
            "budget_default": "paper T is interpreted as total evaluations including initialization",
            "objective_scaling": "objective_value_scale multiplies the latent reward and optimum before raw-variance observation noise is added",
            "recommendation": "20-start continuous posterior-mean maximization, safeguarded by the highest-posterior-mean evaluated input when recommendation_include_observed=true",
            "claude_reference": "the original supplied implementation is preserved under archive/claude_reference/",
        },
        "acquisitions": acquisition_metadata(),
        "selected_acquisition_settings": {
            name: acquisition_settings(name, config)
            for name in selected_acquisitions
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "runs": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, num_seeds = resolve_config(args)
    if args.profile == "paper":
        validate_paper_protocol(config, num_seeds)
    specs = build_specs(args, config, num_seeds)
    selected_acquisitions = sorted({spec.acquisition for spec in specs})
    default_names = {
        "smoke": Path("results/smoke/default"),
        "paper": Path("results/baseline_ucb_ts/performance"),
        "research": Path("results/extensions/research"),
        "high_dimensional": Path("results/high_dimensional/calibrated_default"),
    }
    default_name = default_names[args.profile]
    output_dir = args.output_dir or EXPERIMENT_DIR / default_name
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    digest = config_hash(config)
    acquisition_digest = acquisition_config_hash(config, selected_acquisitions)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_hash") != digest:
            raise RuntimeError(
                "Output directory contains a different experiment configuration. "
                "Choose a new --output-dir; configurations are never mixed."
            )
        recorded_acquisition_digest = metadata.get("acquisition_config_hash")
        if (
            recorded_acquisition_digest is not None
            and recorded_acquisition_digest != acquisition_digest
        ):
            raise RuntimeError(
                "Output directory contains a different acquisition configuration. "
                "Choose a new --output-dir; configurations are never mixed."
            )
    else:
        metadata = _base_metadata(
            config,
            digest,
            acquisition_digest,
            selected_acquisitions,
        )

    invocation = {
        "started_unix_time": time.time(),
        "profile": args.profile,
        "figure": args.figure,
        "selected_trials": len(specs),
        "seed_start": args.seed_start,
        "num_seeds": num_seeds,
        "jobs": args.jobs,
        "python_executable": sys.executable,
        "paper_protocol_validated": args.profile == "paper",
        "overwrite": args.overwrite,
        "status": "running",
    }
    metadata.setdefault("runs", []).append(invocation)
    _atomic_json(metadata_path, metadata)

    pending: list[TrialSpec] = []
    cached = 0
    for spec in specs:
        path = trial_path(trials_dir, spec)
        if not args.overwrite and read_trial(path, spec, config) is not None:
            cached += 1
        else:
            pending.append(spec)

    started = time.perf_counter()
    if args.jobs == 1:
        for index, spec in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] running {spec}", flush=True)
            rows = run_trial(spec, config)
            _atomic_csv(trial_path(trials_dir, spec), rows)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_trial, spec, config): spec for spec in pending}
            for index, future in enumerate(as_completed(futures), 1):
                spec = futures[future]
                rows = future.result()
                _atomic_csv(trial_path(trials_dir, spec), rows)
                print(f"[{index}/{len(pending)}] completed {spec}", flush=True)

    all_rows = _load_all_valid_trials(trials_dir, config)
    _write_aggregate_history(output_dir / "history.csv", all_rows)
    _write_summary(output_dir / "summary.csv", all_rows)
    _write_trial_summary(output_dir / "trial_summary.csv", all_rows)

    invocation["status"] = "completed"
    invocation["finished_unix_time"] = time.time()
    invocation["wall_time_seconds"] = time.perf_counter() - started
    invocation["cached_trials"] = cached
    invocation["computed_trials"] = len(pending)
    invocation["available_trial_files"] = len(list(trials_dir.glob("*.csv")))
    _atomic_json(metadata_path, metadata)
    print(
        f"Ready: {len(pending)} computed, {cached} cached, "
        f"{len(all_rows)} aggregate rows in {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
