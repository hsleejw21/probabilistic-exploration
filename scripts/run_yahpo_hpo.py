#!/usr/bin/env python3
"""Run paired PE comparisons on YAHPO real-data HPO surrogates."""

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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = EXPERIMENT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import numpy as np
import scipy
import sklearn

from probabilistic_exploration.acquisitions import (
    beta_value,
    maximize_acquisition,
    maximize_posterior_variance,
    sobol_points,
)
from probabilistic_exploration.bo import should_optimize_hyperparameters
from probabilistic_exploration.config import AVAILABLE_POLICIES, POLICIES, high_dimensional_config
from probabilistic_exploration.exploration import decay_exponent, exploration_probability, exploration_rule
from probabilistic_exploration.gp_surrogate import build_gaussian_process
from probabilistic_exploration.yahpo_benchmarks import YAHPO_TASKS, empirical_target_range, objective_loss


ACQUISITIONS = ("ucb", "ts", "logei")
FIELDS = [
    "task", "scenario", "instance", "target", "dimension", "acquisition",
    "beta_mode", "policy", "seed", "iteration", "bo_iteration", "event",
    "exploration_rule", "decay_exponent", "exploration_probability",
    "query_preselection_posterior_std", "cumulative_pe_calls",
    "cumulative_mode_transitions", "query_x_unit", "parameters", "loss",
    "best_observed_loss", "best_observed_iteration", "empirical_reference_min",
    "empirical_reference_max", "lengthscale", "signal_variance", "used_jitter",
    "max_posterior_std", "mean_posterior_std", "q90_posterior_std",
    "selection_time_seconds", "objective_time_seconds", "gp_fit_time_seconds",
    "cumulative_wall_time_seconds",
]


@dataclass(frozen=True)
class TrialSpec:
    task: str
    acquisition: str
    beta_mode: str
    policy: str
    seed: int

    def slug(self) -> str:
        return (
            f"{self.task}__{self.acquisition}__{self.beta_mode}__"
            f"{self.policy}__seed{self.seed:04d}"
        )


def application_config(dimension: int, budget: int):
    return replace(
        high_dimensional_config(),
        budget_override=budget,
        n_initial=2 * dimension,
        noise_variance=0.0,
        gp_nugget_variance=1e-4,
        initial_signal_variance=1.0,
        learn_signal_variance=False,
        variance_candidates=1024,
        ts_candidate_design="sobol",
        ts_candidate_schedule="sqrt",
        ts_candidate_min=128,
        ts_candidate_growth_scale=64.0,
        ts_candidate_max=512,
        logarithmic_beta_scale=1.0,
    )


def _stream_seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_trial(spec: TrialSpec, budget: int) -> list[dict[str, object]]:
    task = YAHPO_TASKS[spec.task]
    config = application_config(task.dim, budget)
    if config.n_initial >= budget:
        raise ValueError(f"budget {budget} must exceed 2d={config.n_initial}")

    streams = np.random.SeedSequence(spec.seed).spawn(5)
    initial_design = sobol_points(
        config.n_initial, task.dim, _stream_seed(streams[0])
    )
    decision_rng = np.random.default_rng(streams[1])
    exploration_rng = np.random.default_rng(streams[2])
    acquisition_rng = np.random.default_rng(streams[3])
    diagnostic_design = sobol_points(
        config.variance_candidates, task.dim, _stream_seed(streams[4])
    )
    reference_min, reference_max = empirical_target_range(task)

    gp = build_gaussian_process(
        config, np.zeros(task.dim), np.ones(task.dim)
    )
    x_observed: list[np.ndarray] = []
    rewards: list[float] = []
    losses: list[float] = []
    rows: list[dict[str, object]] = []
    cumulative_time = 0.0
    cumulative_pe_calls = 0
    cumulative_mode_transitions = 0
    previous_bo_mode: str | None = None

    def observe(
        unit_x: np.ndarray,
        event: str,
        probability: float,
        selection_time: float,
        query_preselection_std: float,
    ):
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
        loss, parameters = objective_loss(task, unit_x)
        objective_time = time.perf_counter() - started
        x_observed.append(np.asarray(unit_x, dtype=float).copy())
        losses.append(loss)
        rewards.append(-loss)

        n_observed = len(x_observed)
        started = time.perf_counter()
        gp.fit(
            np.vstack(x_observed),
            np.asarray(rewards),
            optimize_hyperparameters=should_optimize_hyperparameters(
                n_observed, config
            ),
        )
        gp_time = time.perf_counter() - started
        started = time.perf_counter()
        _, diagnostic_std = gp.predict(diagnostic_design)
        diagnostic_time = time.perf_counter() - started
        cumulative_time += selection_time + objective_time + gp_time + diagnostic_time
        best_index = int(np.argmin(losses))
        rows.append({
            "task": task.name,
            "scenario": task.scenario,
            "instance": task.instance,
            "target": task.target,
            "dimension": task.dim,
            "acquisition": spec.acquisition,
            "beta_mode": spec.beta_mode,
            "policy": spec.policy,
            "seed": spec.seed,
            "iteration": n_observed,
            "bo_iteration": max(n_observed - config.n_initial, 0),
            "event": event,
            "exploration_rule": exploration_rule(spec.policy),
            "decay_exponent": decay_exponent(spec.policy),
            "exploration_probability": probability,
            "query_preselection_posterior_std": query_preselection_std,
            "cumulative_pe_calls": cumulative_pe_calls,
            "cumulative_mode_transitions": cumulative_mode_transitions,
            "query_x_unit": json.dumps(np.asarray(unit_x).tolist()),
            "parameters": json.dumps(parameters, sort_keys=True),
            "loss": loss,
            "best_observed_loss": losses[best_index],
            "best_observed_iteration": best_index + 1,
            "empirical_reference_min": reference_min,
            "empirical_reference_max": reference_max,
            "lengthscale": gp.lengthscale,
            "signal_variance": gp.signal_variance,
            "used_jitter": gp.used_jitter,
            "max_posterior_std": float(np.max(diagnostic_std)),
            "mean_posterior_std": float(np.mean(diagnostic_std)),
            "q90_posterior_std": float(np.quantile(diagnostic_std, 0.9)),
            "selection_time_seconds": selection_time,
            "objective_time_seconds": objective_time,
            "gp_fit_time_seconds": gp_time,
            "cumulative_wall_time_seconds": cumulative_time,
        })

    for point in initial_design:
        observe(point, "initial", float("nan"), 0.0, float("nan"))

    while len(x_observed) < budget:
        bo_iteration = len(x_observed) - config.n_initial + 1
        probability = exploration_probability(spec.policy, task.dim, bo_iteration)
        beta = beta_value(spec.beta_mode, bo_iteration, config)
        started = time.perf_counter()
        if decision_rng.random() < probability:
            if exploration_rule(spec.policy) == "mvr":
                point = maximize_posterior_variance(
                    gp, task.dim, exploration_rng, config, restarts=20
                )
                event = "mvr_exploration"
            else:
                point = exploration_rng.random(task.dim)
                event = "uniform_exploration"
        else:
            point = maximize_acquisition(
                spec.acquisition,
                gp,
                task.dim,
                beta,
                max(rewards),
                acquisition_rng,
                config,
                bo_iteration=bo_iteration,
            )
            event = "acquisition"
        _, query_std = gp.predict(np.asarray(point)[None, :])
        observe(
            point, event, probability, time.perf_counter() - started,
            float(query_std[0]),
        )
    return rows


def _atomic_csv(path: Path, rows: list[dict[str, object]], fields=FIELDS):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_valid(path: Path, spec: TrialSpec, budget: int):
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if reader.fieldnames != FIELDS or len(rows) != budget:
            return None
        expected = (
            spec.task, spec.acquisition, spec.beta_mode, spec.policy, str(spec.seed)
        )
        if any(
            (
                row["task"], row["acquisition"], row["beta_mode"],
                row["policy"], row["seed"],
            )
            != expected for row in rows
        ):
            return None
        return rows
    except (OSError, csv.Error, KeyError):
        return None


def _write_aggregates(output_dir: Path, budget: int):
    all_rows: list[dict[str, str]] = []
    for path in sorted((output_dir / "trials").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) == budget:
            all_rows.extend(rows)
    all_rows.sort(key=lambda row: (
        row["task"], row["acquisition"], row["beta_mode"], row["policy"],
        int(row["seed"]), int(row["iteration"]),
    ))
    _atomic_csv(output_dir / "history.csv", all_rows)

    keys = (
        "task", "scenario", "instance", "target", "dimension", "acquisition",
        "beta_mode", "policy", "iteration", "bo_iteration",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in all_rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary_fields = list(keys) + [
        "n_seeds", "mean_best_observed_loss", "median_best_observed_loss",
        "std_best_observed_loss", "sem_best_observed_loss",
        "q25_best_observed_loss", "q75_best_observed_loss",
        "mean_cumulative_wall_time_seconds",
        "mean_max_posterior_std", "mean_mean_posterior_std",
        "mean_q90_posterior_std", "mean_cumulative_pe_calls",
        "mean_cumulative_mode_transitions",
    ]
    summary_rows = []
    for key, group in sorted(groups.items()):
        values = np.asarray([float(row["best_observed_loss"]) for row in group])
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out = dict(zip(keys, key))
        out.update(
            n_seeds=len(group),
            mean_best_observed_loss=float(np.mean(values)),
            median_best_observed_loss=float(np.median(values)),
            std_best_observed_loss=std,
            sem_best_observed_loss=std / math.sqrt(len(values)),
            q25_best_observed_loss=float(np.quantile(values, 0.25)),
            q75_best_observed_loss=float(np.quantile(values, 0.75)),
            mean_cumulative_wall_time_seconds=float(np.mean([
                float(row["cumulative_wall_time_seconds"]) for row in group
            ])),
            mean_max_posterior_std=float(np.mean([
                float(row["max_posterior_std"]) for row in group
            ])),
            mean_mean_posterior_std=float(np.mean([
                float(row["mean_posterior_std"]) for row in group
            ])),
            mean_q90_posterior_std=float(np.mean([
                float(row["q90_posterior_std"]) for row in group
            ])),
            mean_cumulative_pe_calls=float(np.mean([
                float(row["cumulative_pe_calls"]) for row in group
            ])),
            mean_cumulative_mode_transitions=float(np.mean([
                float(row["cumulative_mode_transitions"]) for row in group
            ])),
        )
        summary_rows.append(out)
    _atomic_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    _write_trial_summary(output_dir, all_rows)


def _write_trial_summary(output_dir: Path, rows: list[dict[str, str]]) -> None:
    keys = ["task", "dimension", "acquisition", "beta_mode", "policy", "seed"]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    fields = keys + [
        "exploration_rule", "decay_exponent", "bo_steps", "actual_pe_calls",
        "expected_pe_calls", "realized_pe_fraction", "first_pe_round",
        "last_pe_round", "mode_transitions", "final_best_observed_loss",
        "final_max_posterior_std", "final_mean_posterior_std",
        "final_q90_posterior_std", "final_query_preselection_posterior_std",
    ]
    outputs = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["iteration"]))
        bo_rows = [row for row in ordered if row["event"] != "initial"]
        pe_rows = [row for row in bo_rows if row["event"].endswith("_exploration")]
        endpoint = ordered[-1]
        expected = sum(
            exploration_probability(endpoint["policy"], int(endpoint["dimension"]), t)
            for t in range(1, len(bo_rows) + 1)
        )
        out = dict(zip(keys, key))
        out.update({
            "exploration_rule": endpoint["exploration_rule"],
            "decay_exponent": endpoint["decay_exponent"],
            "bo_steps": len(bo_rows),
            "actual_pe_calls": len(pe_rows),
            "expected_pe_calls": expected,
            "realized_pe_fraction": len(pe_rows) / len(bo_rows) if bo_rows else float("nan"),
            "first_pe_round": int(pe_rows[0]["bo_iteration"]) if pe_rows else "",
            "last_pe_round": int(pe_rows[-1]["bo_iteration"]) if pe_rows else "",
            "mode_transitions": int(endpoint["cumulative_mode_transitions"]),
            "final_best_observed_loss": endpoint["best_observed_loss"],
            "final_max_posterior_std": endpoint["max_posterior_std"],
            "final_mean_posterior_std": endpoint["mean_posterior_std"],
            "final_q90_posterior_std": endpoint["q90_posterior_std"],
            "final_query_preselection_posterior_std": endpoint["query_preselection_posterior_std"],
        })
        outputs.append(out)
    _atomic_csv(output_dir / "trial_summary.csv", outputs, fields)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--acquisitions", nargs="+", default=list(ACQUISITIONS))
    parser.add_argument(
        "--ucb-beta-modes", nargs="+", choices=["constant", "zero"],
        default=["constant"],
    )
    parser.add_argument("--policies", nargs="+", default=["all"])
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = list(YAHPO_TASKS) if args.tasks == ["all"] else args.tasks
    policies = list(POLICIES) if args.policies == ["all"] else args.policies
    unknown = (
        set(tasks) - set(YAHPO_TASKS),
        set(args.acquisitions) - set(ACQUISITIONS),
        set(policies) - set(AVAILABLE_POLICIES),
    )
    if any(unknown):
        raise ValueError(f"unknown tasks/acquisitions/policies: {unknown}")

    acquisition_modes = []
    for acquisition in args.acquisitions:
        if acquisition == "ucb":
            modes = args.ucb_beta_modes
        elif acquisition == "ts":
            modes = ["logarithmic"]
        else:
            modes = ["native"]
        acquisition_modes.extend((acquisition, mode) for mode in modes)
    specs = [
        TrialSpec(task, acquisition, beta_mode, policy, seed)
        for task in tasks
        for acquisition, beta_mode in acquisition_modes
        for policy in policies
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
    ]
    payload = {
        "schema_version": 2,
        "benchmark_family": "YAHPO Gym real-data HPO surrogate",
        "tasks": {name: asdict(YAHPO_TASKS[name]) for name in tasks},
        "acquisitions": args.acquisitions,
        "ucb_beta_modes": args.ucb_beta_modes,
        "policies": policies,
        "num_seeds": args.num_seeds,
        "seed_start": args.seed_start,
        "budget": args.budget,
        "initial_design": "2d scrambled Sobol, included in budget",
        "objective": "minimize YAHPO-predicted validation log loss",
        "gp_observation_noise": 0.0,
        "gp_nugget_variance": 1e-4,
        "signal_variance": "fixed at 1; lengthscales learned",
        "gp_ts_protocol": (
            "beta_t=log(t+1); fresh scrambled Sobol candidates; "
            "clip(ceil(64*sqrt(t)),128,512)"
        ),
        "yahpo_data_version": "1.0.2",
        "gp_config_by_dimension": {
            str(dim): asdict(application_config(dim, args.budget))
            for dim in sorted({YAHPO_TASKS[name].dim for name in tasks})
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "sources": [
            "https://github.com/slds-lmu/yahpo_gym",
            "https://github.com/slds-lmu/yahpo_data/releases/tag/v1.0.2",
            "https://proceedings.mlr.press/v188/pfisterer22a.html",
        ],
    }
    protocol = {k: v for k, v in payload.items() if k not in {"num_seeds", "seed_start"}}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    payload["config_hash"] = digest
    output_dir = args.output_dir
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != digest:
            raise RuntimeError("output directory contains a different configuration")
    else:
        _atomic_json(metadata_path, payload)

    pending, cached = [], 0
    for spec in specs:
        path = trials_dir / f"{spec.slug()}.csv"
        if _read_valid(path, spec, args.budget) is None:
            pending.append(spec)
        else:
            cached += 1
    completed = 0
    if args.jobs == 1:
        for spec in pending:
            _atomic_csv(trials_dir / f"{spec.slug()}.csv", run_trial(spec, args.budget))
            completed += 1
            print(f"[{completed}/{len(pending)}] completed {spec}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_trial, spec, args.budget): spec for spec in pending}
            for future in as_completed(futures):
                spec = futures[future]
                _atomic_csv(trials_dir / f"{spec.slug()}.csv", future.result())
                completed += 1
                print(f"[{completed}/{len(pending)}] completed {spec}", flush=True)
    _write_aggregates(output_dir, args.budget)
    print(f"Ready: {completed} computed, {cached} cached in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
