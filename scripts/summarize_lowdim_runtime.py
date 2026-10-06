#!/usr/bin/env python3
"""Create the public low-dimensional runtime aggregates from raw histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_TO_METHOD = {
    "standard": "Standard",
    "decay_uniform_grid_a1_2": "Uniform",
    "decay_mvr_grid_a1_2": "MVR",
}
REQUIRED_COLUMNS = {
    "objective",
    "policy",
    "seed",
    "iteration",
    "inference_regret",
    "cumulative_wall_time_seconds",
}


def load_final_rows(paths: list[Path]) -> pd.DataFrame:
    """Load histories and retain the final observation of every paired run."""
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")

    frame = frame.loc[frame["policy"].isin(POLICY_TO_METHOD)].copy()
    if frame.empty:
        raise ValueError("No Standard, Uniform PE, or MVR PE rows were found")
    frame["method"] = frame["policy"].map(POLICY_TO_METHOD)
    keys = ["objective", "method", "seed"]
    final = (
        frame.sort_values(keys + ["iteration"])
        .groupby(keys, as_index=False, sort=True)
        .tail(1)
    )
    final = final.rename(
        columns={
            "inference_regret": "final_regret",
            "cumulative_wall_time_seconds": "runtime",
        }
    )
    final = final[["objective", "method", "seed", "final_regret", "runtime"]]

    counts = final.groupby(["objective", "seed"])["method"].nunique()
    incomplete = counts[counts != len(POLICY_TO_METHOD)]
    if len(incomplete):
        raise ValueError(
            "Every objective/seed pair must contain all three methods; "
            f"incomplete pairs: {list(incomplete.index)}"
        )
    return final.sort_values(["objective", "method", "seed"])


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample standard deviation for regret and runtime."""
    labels = {
        "ackley_shifted_narrow_2d": "Ackley 2D",
        "rastrigin_mean_rotated_shifted_2d": "Rastrigin 2D",
    }
    summary = (
        frame.groupby(["objective", "method"], as_index=False)
        .agg(
            n=("seed", "size"),
            regret_mean=("final_regret", "mean"),
            regret_sd=("final_regret", "std"),
            runtime_mean_s=("runtime", "mean"),
            runtime_sd_s=("runtime", "std"),
        )
    )
    summary["objective"] = summary["objective"].map(labels).fillna(
        summary["objective"]
    )
    return summary


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute paired improvements; positive values favor the candidate."""
    rng = np.random.default_rng(20261006)
    rows: list[dict[str, object]] = []
    comparisons = (("Standard", "Uniform"), ("Standard", "MVR"), ("Uniform", "MVR"))
    for objective, part in frame.groupby("objective", sort=True):
        for metric in ("final_regret", "runtime"):
            wide = part.pivot(index="seed", columns="method", values=metric)
            for baseline, candidate in comparisons:
                difference = (
                    wide[baseline].to_numpy(float) - wide[candidate].to_numpy(float)
                )
                indices = rng.integers(
                    0, len(difference), size=(20_000, len(difference))
                )
                bootstrap = difference[indices].mean(axis=1)
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                rows.append(
                    {
                        "objective": objective,
                        "metric": metric,
                        "baseline_method": baseline,
                        "candidate_method": candidate,
                        "n": len(difference),
                        "mean_improvement": float(difference.mean()),
                        "bootstrap_ci95_low": float(low),
                        "bootstrap_ci95_high": float(high),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        action="append",
        required=True,
        help="A history.csv file; repeat for separately executed method blocks.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    final = load_final_rows(args.history)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_dir / "per_seed.csv", index=False)
    summarize(final).to_csv(args.output_dir / "summary.csv", index=False)
    paired_comparisons(final).to_csv(
        args.output_dir / "paired_comparisons.csv", index=False
    )


if __name__ == "__main__":
    main()
