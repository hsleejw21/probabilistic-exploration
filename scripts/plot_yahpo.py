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
TASK_LABELS = {
    "rbv2_xgboost_31": "credit-g",
    "rbv2_xgboost_40975": "car",
    "rbv2_xgboost_1464": "blood transfusion",
}
POLICY_ORDER = (
    "standard", "decay_uniform_grid_a1_2", "decay_uniform_a1",
    "fixed_p020", "fixed_uniform", "decay_uniform",
)


def plot_endpoint_improvements(paired: pd.DataFrame, figure_dir: Path) -> None:
    """Plot paired endpoint improvements using the repository-wide sign convention."""
    preferred_tasks = (
        "rbv2_xgboost_31",
        "rbv2_xgboost_40975",
        "rbv2_xgboost_1464",
    )
    observed_tasks = set(paired["task"])
    tasks = [task for task in preferred_tasks if task in observed_tasks]
    tasks.extend(sorted(observed_tasks - set(tasks)))
    acquisitions = ("logei", "mes_gumbel", "ts", "ucb")
    policies = (
        "decay_uniform_grid_a1_2",
        "decay_uniform_a1",
        "fixed_p020",
    )
    offsets = (-0.20, 0.0, 0.20)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.0), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        frame = paired[paired["task"] == task]
        for policy, offset in zip(policies, offsets):
            selected = frame.set_index(["acquisition", "candidate_policy"])
            rows = [selected.loc[(acquisition, policy)] for acquisition in acquisitions]
            means = np.asarray(
                [row["mean_improvement_standard_minus_candidate"] for row in rows]
            )
            lows = np.asarray([row["bootstrap_ci95_low"] for row in rows])
            highs = np.asarray([row["bootstrap_ci95_high"] for row in rows])
            x = np.arange(len(acquisitions), dtype=float) + offset
            significant = (lows > 0) | (highs < 0)
            axis.errorbar(
                x,
                means,
                yerr=np.vstack((means - lows, highs - means)),
                fmt="o",
                color=COLORS[policy],
                label=LABELS[policy],
                capsize=2.5,
                linewidth=1.2,
                markersize=5,
            )
            axis.scatter(
                x[significant], means[significant], color=COLORS[policy],
                edgecolor="black", linewidth=0.8, s=38, zorder=4,
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xticks(range(len(acquisitions)))
        axis.set_xticklabels(("LogEI", "MES-G", "GP-TS", "GP-UCB"))
        axis.tick_params(axis="x", labelsize=9)
        axis.set_title(TASK_LABELS.get(task, task))
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("Endpoint improvement over Standard\n(Standard loss - PE loss)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(figure_dir / "yahpo_endpoint_improvement.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "yahpo_endpoint_improvement.pdf", bbox_inches="tight")
    plt.close(fig)


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
            # YAHPO reports a loss.  Standard - candidate is therefore a
            # higher-is-better improvement, consistent with every other public result.
            difference = standard.loc[seeds].to_numpy() - candidate.loc[seeds].to_numpy()
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
                "mean_improvement_standard_minus_candidate": float(np.mean(difference)),
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "candidate_wins": int(np.sum(difference > 0)),
                "candidate_losses": int(np.sum(difference < 0)),
                "ties": int(np.sum(difference == 0)),
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.input_dir / "paired_comparisons.csv", index=False)
    plot_endpoint_improvements(paired, figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
