#!/usr/bin/env python3
"""Run paired PE experiments on the Lunar Lander controller objective."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = EXPERIMENT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probabilistic_exploration.acquisitions import acquisition_beta_modes, acquisition_names
from probabilistic_exploration.benchmarks import BENCHMARKS
from probabilistic_exploration.bo import TrialSpec, run_trial
from probabilistic_exploration.config import AVAILABLE_POLICIES, high_dimensional_config
from probabilistic_exploration.lunar_benchmark import (
    LunarSettings,
    build_lunar_benchmark,
    build_lunar_placeholder,
    close_lunar_benchmark,
    lunar_episode_seeds,
)
from run_synthetic import (
    _atomic_csv,
    _atomic_json,
    _base_metadata,
    _load_all_valid_trials,
    _write_aggregate_history,
    _write_summary,
    _write_trial_summary,
    acquisition_config_hash,
    config_hash,
    read_trial,
    trial_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", nargs="+", default=["ucb"])
    parser.add_argument("--beta-mode", default="constant")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "standard",
            "decay_uniform_grid_a1_2",
            "decay_uniform",
            "decay_uniform_a1",
            "fixed_p020",
        ],
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument(
        "--noise-variance",
        type=float,
        default=0.0,
        help="observation-noise variance; the released protocol uses zero",
    )
    parser.add_argument("--lunar-rollouts", type=int, default=50)
    parser.add_argument("--ts-candidates", type=int, default=None)
    parser.add_argument(
        "--ts-candidate-design", choices=["sobol", "uniform"], default=None
    )
    parser.add_argument(
        "--ts-candidate-schedule", choices=["fixed", "sqrt"], default=None
    )
    parser.add_argument("--ts-candidate-min", type=int, default=None)
    parser.add_argument("--ts-candidate-growth-scale", type=float, default=None)
    parser.add_argument("--ts-candidate-max", type=int, default=None)
    parser.add_argument("--logarithmic-beta-scale", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _build_config(args: argparse.Namespace):
    default_budget, default_initial, nugget = 400, 24, 1e-2
    budget = args.budget if args.budget is not None else default_budget
    n_initial = args.n_initial if args.n_initial is not None else default_initial
    if args.smoke:
        budget = args.budget if args.budget is not None else 6
        n_initial = args.n_initial if args.n_initial is not None else 2
    if n_initial < 1 or n_initial >= budget:
        raise ValueError("n_initial must be positive and smaller than budget")
    updates = dict(
        budget_override=budget,
        n_initial=n_initial,
        noise_variance=args.noise_variance,
        gp_nugget_variance=nugget,
        initial_signal_variance=1.0,
        learn_signal_variance=False,
    )
    for argument, field in (
        (args.ts_candidates, "ts_candidates"),
        (args.ts_candidate_design, "ts_candidate_design"),
        (args.ts_candidate_schedule, "ts_candidate_schedule"),
        (args.ts_candidate_min, "ts_candidate_min"),
        (args.ts_candidate_growth_scale, "ts_candidate_growth_scale"),
        (args.ts_candidate_max, "ts_candidate_max"),
        (args.logarithmic_beta_scale, "logarithmic_beta_scale"),
    ):
        if argument is not None:
            updates[field] = argument
    if args.smoke:
        updates.update(
            hyperopt_interval=0,
            optimizer_restarts=2,
            recommendation_restarts=2,
            optimizer_maxiter=10,
            variance_candidates=32,
            ts_candidates=32,
            mes_num_max_samples=8,
            mes_num_representer_points=128,
            kg_num_candidates=16,
            kg_num_representer_points=32,
            kg_num_fantasies=8,
            kg_candidate_batch_size=8,
        )
    return replace(high_dimensional_config(), **updates)


def _build_specs(args: argparse.Namespace) -> list[TrialSpec]:
    if args.num_seeds < 1 or args.jobs < 1:
        raise ValueError("num_seeds and jobs must be positive")
    unknown_acquisitions = set(args.acquisitions) - set(acquisition_names())
    unknown_policies = set(args.policies) - set(AVAILABLE_POLICIES)
    if unknown_acquisitions:
        raise ValueError(f"unknown acquisitions: {sorted(unknown_acquisitions)}")
    if unknown_policies:
        raise ValueError(f"unknown policies: {sorted(unknown_policies)}")
    for acquisition in args.acquisitions:
        if args.beta_mode not in acquisition_beta_modes(acquisition):
            raise ValueError(
                f"{acquisition} does not support beta mode {args.beta_mode}"
            )
    return [
        TrialSpec("lunar_lander_12d", acquisition, args.beta_mode, policy, seed)
        for acquisition in args.acquisitions
        for policy in args.policies
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
    ]


def _run_lunar_trial(
    spec: TrialSpec,
    config,
    settings: LunarSettings,
):
    benchmark, _ = build_lunar_benchmark(settings, trial_seed=spec.seed)
    BENCHMARKS[spec.objective] = benchmark
    try:
        return run_trial(spec, config)
    finally:
        close_lunar_benchmark(benchmark)


def _write_lunar_audit(
    output_dir: Path,
    specs: list[TrialSpec],
    config,
) -> None:
    failures: list[str] = []
    loaded: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for spec in specs:
        rows = read_trial(trial_path(output_dir / "trials", spec), spec, config)
        if rows is None:
            failures.append(f"missing_or_invalid_trial={spec.slug()}")
            continue
        if any(abs(float(row["signal_variance"]) - 1.0) > 1e-12 for row in rows):
            failures.append(f"signal_variance_not_one={spec.slug()}")
        loaded[(spec.acquisition, spec.policy, spec.seed)] = rows

    acquisitions = sorted({spec.acquisition for spec in specs})
    seeds = sorted({spec.seed for spec in specs})
    for acquisition in acquisitions:
        for seed in seeds:
            group = [
                rows
                for (acq, _policy, trial_seed), rows in loaded.items()
                if acq == acquisition and trial_seed == seed
            ]
            if len(group) < 2:
                continue
            prefixes = [
                [
                    (
                        row["query_x_unit"],
                        row["latent_query_value"],
                        row["observed_value"],
                    )
                    for row in rows
                    if row["event"] == "initial"
                ]
                for rows in group
            ]
            if any(prefix != prefixes[0] for prefix in prefixes[1:]):
                failures.append(
                    f"unpaired_initial={acquisition}:seed{seed:04d}"
                )

    lines = [
        "PASS" if not failures else "FAIL",
        f"expected_trials={len(specs)}",
        f"valid_trials={len(loaded)}",
        f"fixed_signal_variance={not any('signal_variance' in x for x in failures)}",
        f"paired_initial_designs={not any('unpaired_initial' in x for x in failures)}",
    ]
    lines.extend(failures)
    (output_dir / "audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(
            "Lunar benchmark audit failed; inspect "
            f"{output_dir / 'audit.txt'}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    specs = _build_specs(args)
    settings = LunarSettings(rollouts=args.lunar_rollouts)

    # Parent-side registration supplies dimensions for cache validation and
    # aggregation.  Workers replace it with their paired trial-seed objective.
    placeholder, task_metadata = build_lunar_placeholder(settings, trial_seed=0)
    BENCHMARKS[settings.task] = placeholder

    default_dir = EXPERIMENT_DIR / "results" / "lunar"
    output_dir = args.output_dir or default_dir
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    selected_acquisitions = sorted(set(args.acquisitions))
    common_digest = config_hash(config)
    acquisition_digest = acquisition_config_hash(config, selected_acquisitions)
    lunar_payload = {
        "settings": settings.metadata(),
        "source": "TuRBO Lunar Lander controller example",
        "task_metadata_seed0": task_metadata,
    }
    lunar_digest = __import__("hashlib").sha256(
        json.dumps(lunar_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = (common_digest, acquisition_digest, lunar_digest)
        actual = (
            metadata.get("config_hash"),
            metadata.get("acquisition_config_hash"),
            metadata.get("lunar_config_hash"),
        )
        if actual != expected:
            raise RuntimeError(
                "Output directory contains a different Lunar protocol; "
                "choose a new --output-dir"
            )
    else:
        metadata = _base_metadata(
            config, common_digest, acquisition_digest, selected_acquisitions
        )
        metadata.update(
            lunar_config_hash=lunar_digest,
            lunar_benchmark=lunar_payload,
            inference_regret_note=(
                "Lunar uses the scaled solved-score target 200, not a known global optimum"
            ),
        )

    invocation = {
        "started_unix_time": time.time(),
        "task": settings.task,
        "selected_trials": len(specs),
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "jobs": args.jobs,
        "smoke": args.smoke,
        "overwrite": args.overwrite,
        "python_executable": sys.executable,
        "status": "running",
    }
    invocation["lunar_episode_seeds"] = {
        str(seed): list(lunar_episode_seeds(seed, args.lunar_rollouts))
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
    }
    metadata.setdefault("runs", []).append(invocation)
    _atomic_json(metadata_path, metadata)

    pending = []
    cached = 0
    for spec in specs:
        path = trial_path(trials_dir, spec)
        if not args.overwrite and read_trial(path, spec, config) is not None:
            cached += 1
        else:
            pending.append(spec)

    started = time.perf_counter()
    try:
        if args.jobs == 1:
            for index, spec in enumerate(pending, 1):
                print(f"[{index}/{len(pending)}] running {spec}", flush=True)
                rows = _run_lunar_trial(spec, config, settings)
                _atomic_csv(trial_path(trials_dir, spec), rows)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(_run_lunar_trial, spec, config, settings): spec
                    for spec in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    spec = futures[future]
                    rows = future.result()
                    _atomic_csv(trial_path(trials_dir, spec), rows)
                    print(
                        f"[{index}/{len(pending)}] completed {spec}", flush=True
                    )

        all_rows = _load_all_valid_trials(trials_dir, config)
        _write_aggregate_history(output_dir / "history.csv", all_rows)
        _write_summary(output_dir / "summary.csv", all_rows)
        _write_trial_summary(output_dir / "trial_summary.csv", all_rows)
        _write_lunar_audit(output_dir, specs, config)
        invocation["status"] = "completed"
        invocation["computed_trials"] = len(pending)
        invocation["cached_trials"] = cached
        invocation["available_trial_files"] = len(list(trials_dir.glob("*.csv")))
    except Exception:
        invocation["status"] = "failed"
        raise
    finally:
        invocation["finished_unix_time"] = time.time()
        invocation["wall_time_seconds"] = time.perf_counter() - started
        _atomic_json(metadata_path, metadata)
        close_lunar_benchmark(placeholder)

    print(
        f"Ready: {len(pending)} computed, {cached} cached in {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
