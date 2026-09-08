#!/usr/bin/env python3
"""Run paired PE comparisons on Bayesmark-derived real-data HPO tasks."""

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

from probabilistic_exploration.acquisitions import beta_value, maximize_acquisition, sobol_points
from probabilistic_exploration.bo import should_optimize_hyperparameters
from probabilistic_exploration.config import AVAILABLE_POLICIES, POLICIES, high_dimensional_config
from probabilistic_exploration.exploration import exploration_probability
from probabilistic_exploration.gp_surrogate import build_gaussian_process
from probabilistic_exploration.real_world_benchmarks import (
    BAYESMARK_TASKS,
    heldout_test_loss,
    validation_loss,
)


ACQUISITIONS = ("ucb", "logei", "ts", "mes_gumbel")
FIELDS = [
    "task", "dataset", "model", "dimension", "acquisition", "beta_mode",
    "policy", "seed", "iteration", "bo_iteration", "event",
    "exploration_probability", "query_x_unit", "parameters",
    "validation_loss", "best_validation_loss", "best_validation_iteration",
    "final_test_loss", "lengthscale", "signal_variance", "used_jitter",
    "selection_time_seconds", "objective_time_seconds", "gp_fit_time_seconds",
    "cumulative_wall_time_seconds",
]


@dataclass(frozen=True)
class RealTrialSpec:
    task: str
    acquisition: str
    policy: str
    seed: int

    @property
    def beta_mode(self) -> str:
        if self.acquisition == "ts":
            return "logarithmic"
        return "constant" if self.acquisition == "ucb" else "native"

    def slug(self) -> str:
        return (
            f"{self.task}__{self.acquisition}__{self.beta_mode}__"
            f"{self.policy}__seed{self.seed:04d}"
        )


def real_world_config(dimension: int, budget: int):
    return replace(
        high_dimensional_config(),
        budget_override=budget,
        n_initial=2 * dimension,
        noise_variance=1e-4,
        initial_signal_variance=1.0,
        learn_signal_variance=False,
        ts_candidate_design="sobol",
        ts_candidate_schedule="sqrt",
        ts_candidate_min=128,
        ts_candidate_growth_scale=64.0,
        ts_candidate_max=512,
        logarithmic_beta_scale=1.0,
        variance_candidates=0,
    )


def _stream_seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_real_trial(spec: RealTrialSpec, budget: int) -> list[dict[str, object]]:
    task = BAYESMARK_TASKS[spec.task]
    config = real_world_config(task.dim, budget)
    if config.n_initial >= budget:
        raise ValueError(f"budget {budget} must exceed 2d={config.n_initial}")

    streams = np.random.SeedSequence(spec.seed).spawn(5)
    initial_seed = _stream_seed(streams[0])
    decision_rng = np.random.default_rng(streams[1])
    exploration_rng = np.random.default_rng(streams[2])
    acquisition_rng = np.random.default_rng(streams[3])
    initial_design = sobol_points(config.n_initial, task.dim, initial_seed)

    gp = build_gaussian_process(
        config,
        np.zeros(task.dim, dtype=float),
        np.ones(task.dim, dtype=float),
    )
    x_observed: list[np.ndarray] = []
    rewards: list[float] = []
    losses: list[float] = []
    rows: list[dict[str, object]] = []
    cumulative_time = 0.0

    def observe(unit_x: np.ndarray, event: str, p_explore: float, selection_time: float) -> None:
        nonlocal cumulative_time
        started = time.perf_counter()
        loss, parameters = validation_loss(task, unit_x, spec.seed)
        objective_time = time.perf_counter() - started
        x_observed.append(np.asarray(unit_x, dtype=float).copy())
        losses.append(float(loss))
        rewards.append(-float(loss))

        n_observed = len(x_observed)
        started = time.perf_counter()
        gp.fit(
            np.vstack(x_observed),
            np.asarray(rewards),
            optimize_hyperparameters=should_optimize_hyperparameters(
                n_observed, config
            ),
        )
        gp_fit_time = time.perf_counter() - started
        step_time = selection_time + objective_time + gp_fit_time
        cumulative_time += step_time
        best_index = int(np.argmin(losses))
        rows.append(
            {
                "task": task.name,
                "dataset": task.dataset,
                "model": task.model,
                "dimension": task.dim,
                "acquisition": spec.acquisition,
                "beta_mode": spec.beta_mode,
                "policy": spec.policy,
                "seed": spec.seed,
                "iteration": n_observed,
                "bo_iteration": max(n_observed - config.n_initial, 0),
                "event": event,
                "exploration_probability": p_explore,
                "query_x_unit": json.dumps(np.asarray(unit_x).tolist()),
                "parameters": json.dumps(parameters, sort_keys=True),
                "validation_loss": loss,
                "best_validation_loss": losses[best_index],
                "best_validation_iteration": best_index + 1,
                "final_test_loss": "",
                "lengthscale": gp.lengthscale,
                "signal_variance": gp.signal_variance,
                "used_jitter": gp.used_jitter,
                "selection_time_seconds": selection_time,
                "objective_time_seconds": objective_time,
                "gp_fit_time_seconds": gp_fit_time,
                "cumulative_wall_time_seconds": cumulative_time,
            }
        )

    for point in initial_design:
        observe(point, "initial", float("nan"), 0.0)

    while len(x_observed) < budget:
        bo_t = len(x_observed) - config.n_initial + 1
        p_explore = exploration_probability(spec.policy, task.dim, bo_t)
        beta = beta_value(spec.beta_mode, bo_t, config)
        started = time.perf_counter()
        if decision_rng.random() < p_explore:
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
                bo_iteration=bo_t,
            )
            event = "acquisition"
        observe(point, event, p_explore, time.perf_counter() - started)

    best_index = int(np.argmin(losses))
    started = time.perf_counter()
    test_loss = heldout_test_loss(task, x_observed[best_index], spec.seed)
    rows[-1]["final_test_loss"] = test_loss
    rows[-1]["cumulative_wall_time_seconds"] = (
        float(rows[-1]["cumulative_wall_time_seconds"])
        + time.perf_counter()
        - started
    )
    return rows


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_valid(path: Path, spec: RealTrialSpec, budget: int):
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if reader.fieldnames != FIELDS or len(rows) != budget:
            return None
        expected = (spec.task, spec.acquisition, spec.policy, str(spec.seed))
        if any(
            (row["task"], row["acquisition"], row["policy"], row["seed"])
            != expected
            for row in rows
        ):
            return None
        return rows
    except (OSError, csv.Error, KeyError):
        return None


def _write_aggregates(output_dir: Path, budget: int) -> None:
    all_rows: list[dict[str, str]] = []
    for path in sorted((output_dir / "trials").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) == budget:
            all_rows.extend(rows)
    all_rows.sort(
        key=lambda row: (
            row["task"], row["acquisition"], row["policy"],
            int(row["seed"]), int(row["iteration"]),
        )
    )
    _atomic_csv(output_dir / "history.csv", all_rows)

    keys = ("task", "dataset", "model", "dimension", "acquisition", "policy", "iteration", "bo_iteration")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in all_rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary_fields = list(keys) + [
        "n_seeds", "mean_best_validation_loss", "std_best_validation_loss",
        "sem_best_validation_loss", "mean_cumulative_wall_time_seconds",
    ]
    temporary = output_dir / f".summary.csv.tmp.{os.getpid()}"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            values = np.asarray([float(row["best_validation_loss"]) for row in group])
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out = dict(zip(keys, key))
            out.update(
                n_seeds=len(group),
                mean_best_validation_loss=float(np.mean(values)),
                std_best_validation_loss=std,
                sem_best_validation_loss=std / math.sqrt(len(values)),
                mean_cumulative_wall_time_seconds=float(np.mean([
                    float(row["cumulative_wall_time_seconds"]) for row in group
                ])),
            )
            writer.writerow(out)
    os.replace(temporary, output_dir / "summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--acquisitions", nargs="+", default=list(ACQUISITIONS))
    parser.add_argument("--policies", nargs="+", default=["all"])
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = list(BAYESMARK_TASKS) if args.tasks == ["all"] else args.tasks
    policies = list(POLICIES) if args.policies == ["all"] else args.policies
    unknown_tasks = set(tasks) - set(BAYESMARK_TASKS)
    unknown_acquisitions = set(args.acquisitions) - set(ACQUISITIONS)
    unknown_policies = set(policies) - set(AVAILABLE_POLICIES)
    if unknown_tasks or unknown_acquisitions or unknown_policies:
        raise ValueError(
            f"unknown selections: tasks={sorted(unknown_tasks)}, "
            f"acquisitions={sorted(unknown_acquisitions)}, policies={sorted(unknown_policies)}"
        )
    specs = [
        RealTrialSpec(task, acquisition, policy, seed)
        for task in tasks
        for acquisition in args.acquisitions
        for policy in policies
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
    ]
    payload = {
        "schema_version": 1,
        "benchmark_family": "Bayesmark-derived sklearn classification",
        "tasks": {name: asdict(BAYESMARK_TASKS[name]) for name in tasks},
        "acquisitions": args.acquisitions,
        "policies": policies,
        "num_seeds": args.num_seeds,
        "seed_start": args.seed_start,
        "budget": args.budget,
        "initial_design": "2d scrambled Sobol, included in budget",
        "validation": "Bayesmark 80/20 split and five-fold accuracy on training portion",
        "final_test": "one fit of the best-observed validation configuration",
        "gp_config_by_dimension": {
            str(dim): asdict(real_world_config(dim, args.budget))
            for dim in sorted({BAYESMARK_TASKS[name].dim for name in tasks})
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    protocol_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"num_seeds", "seed_start"}
    }
    digest = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
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

    pending = []
    cached = 0
    for spec in specs:
        path = trials_dir / f"{spec.slug()}.csv"
        if _read_valid(path, spec, args.budget) is None:
            pending.append(spec)
        else:
            cached += 1

    completed = 0
    if args.jobs == 1:
        for spec in pending:
            rows = run_real_trial(spec, args.budget)
            _atomic_csv(trials_dir / f"{spec.slug()}.csv", rows)
            completed += 1
            print(f"[{completed}/{len(pending)}] completed {spec}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_real_trial, spec, args.budget): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                rows = future.result()
                _atomic_csv(trials_dir / f"{spec.slug()}.csv", rows)
                completed += 1
                print(f"[{completed}/{len(pending)}] completed {spec}", flush=True)

    _write_aggregates(output_dir, args.budget)
    print(f"Ready: {completed} computed, {cached} cached in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
