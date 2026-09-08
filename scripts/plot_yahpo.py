#!/usr/bin/env python3
"""Plot YAHPO best-observed loss and write a final-step comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "standard": "#4d4d4d",
    "fixed_uniform": "#1f77b4",
    "decay_uniform": "#2ca02c",
    "decay_uniform_grid_a1_2": "#86b6ef",
    "decay_uniform_a1": "#104281",
    "fixed_p020": "#eb6834",
}
LABELS = {
    "standard": "Standard",
    "fixed_uniform": "Fixed Uniform",
    "decay_uniform": "Decay Uniform",
    "decay_uniform_grid_a1_2": "Decay alpha=1/2",
    "decay_uniform_a1": "Decay alpha=1",
    "fixed_p020": "Fixed p=0.2",
}
POLICY_ORDER = (
    "standard", "decay_uniform_grid_a1_2", "decay_uniform_a1",
    "fixed_p020", "fixed_uniform", "decay_uniform",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--band", choices=("std", "iqr"), default="std")
    args = parser.parse_args()

    history = pd.read_csv(args.input_dir / "history.csv")
    figure_dir = args.input_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    for task, task_frame in history.groupby("task"):
        acquisitions = sorted(task_frame["acquisition"].unique())
        fig, axes = plt.subplots(1, len(acquisitions), figsize=(5.2 * len(acquisitions), 4.0), squeeze=False)
        for axis, acquisition in zip(axes[0], acquisitions):
            frame = task_frame[task_frame["acquisition"] == acquisition]
            observed = set(frame["policy"].unique())
            policies = [policy for policy in POLICY_ORDER if policy in observed]
            policies.extend(sorted(observed - set(policies)))
            for policy in policies:
                selected = frame[frame["policy"] == policy]
                pivot = selected.pivot(index="iteration", columns="seed", values="best_observed_loss")
                center = pivot.mean(axis=1)
                if args.band == "std":
                    lower = center - pivot.std(axis=1, ddof=1).fillna(0.0)
                    upper = center + pivot.std(axis=1, ddof=1).fillna(0.0)
                    band_label = "mean ± seed SD"
                else:
                    lower = pivot.quantile(0.25, axis=1)
                    upper = pivot.quantile(0.75, axis=1)
                    band_label = "seed IQR"
                axis.plot(center.index, center, color=COLORS[policy], label=LABELS[policy])
                axis.fill_between(center.index, lower, upper, color=COLORS[policy], alpha=0.18)
            axis.set_title(acquisition.upper().replace("_GUMBEL", "-G"))
            axis.set_xlabel("Number of evaluations")
            axis.set_ylabel("Best-observed validation log loss")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, title=band_label)
        fig.suptitle(f"{task}: YAHPO HPO")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_{args.band}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{task}_{args.band}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    final_iteration = history.groupby(["task", "acquisition", "policy", "seed"])["iteration"].transform("max")
    final = history[history["iteration"] == final_iteration]
    table = final.groupby(["task", "acquisition", "policy"])["best_observed_loss"].agg(
        n_seeds="count", mean="mean", median="median", std="std", minimum="min", maximum="max"
    ).reset_index()
    table.to_csv(args.input_dir / "final_comparison.csv", index=False)

    paired_rows = []
    rng = np.random.default_rng(20260914)
    for (task, acquisition), frame in final.groupby(["task", "acquisition"]):
        standard = frame[frame["policy"] == "standard"].set_index("seed")["best_observed_loss"]
        for policy in sorted(set(frame["policy"]) - {"standard"}):
            candidate = frame[frame["policy"] == policy].set_index("seed")["best_observed_loss"]
            seeds = standard.index.intersection(candidate.index)
            difference = candidate.loc[seeds].to_numpy() - standard.loc[seeds].to_numpy()
            if len(difference) > 1:
                samples = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
            else:
                low = high = float(difference[0])
            paired_rows.append({
                "task": task,
                "acquisition": acquisition,
                "candidate_policy": policy,
                "baseline_policy": "standard",
                "n_seeds": len(difference),
                "mean_difference_candidate_minus_standard": float(np.mean(difference)),
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "candidate_wins": int(np.sum(difference < 0)),
                "candidate_losses": int(np.sum(difference > 0)),
                "ties": int(np.sum(difference == 0)),
            })
    pd.DataFrame(paired_rows).to_csv(args.input_dir / "paired_comparisons.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
