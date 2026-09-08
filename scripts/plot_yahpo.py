#!/usr/bin/env python3
"""Plot YAHPO best-observed loss and write a final-step comparison table."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from probabilistic_exploration.plot_style import (
    ACQUISITION_LABELS,
    ACQUISITION_ORDER,
    LINE_WIDTH,
    MARKER_SIZE,
    POLICY_COLORS,
    POLICY_LABELS,
    STANDARD_GRAY,
    WIDTH_FULL,
    apply_publication_style,
    save_figure,
    style_axis,
)

TASK_LABELS = {
    "rbv2_xgboost_31": "credit-g",
    "rbv2_xgboost_40975": "car",
    "rbv2_xgboost_1464": "blood transfusion",
}
POLICY_ORDER = (
    "standard", "decay_uniform_grid_a1_2", "decay_uniform_a1",
    "fixed_p020", "fixed_uniform", "decay_uniform",
)
def plot_endpoint_improvements(
    paired: pd.DataFrame, figure_dir: Path, dimension: int
) -> None:
    """Plot paired endpoint improvements using the repository-wide sign convention."""
    preferred_tasks = (
        "rbv2_xgboost_31",
        "rbv2_xgboost_40975",
        "rbv2_xgboost_1464",
    )
    observed_tasks = set(paired["task"])
    tasks = [task for task in preferred_tasks if task in observed_tasks]
    tasks.extend(sorted(observed_tasks - set(tasks)))
    acquisitions = tuple(
        acquisition
        for acquisition in ACQUISITION_ORDER
        if acquisition in set(paired["acquisition"])
    )
    policies = (
        "decay_uniform_grid_a1_2",
        "decay_uniform_a1",
        "fixed_p020",
    )
    offsets = (-0.20, 0.0, 0.20)
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(WIDTH_FULL, 4.15), squeeze=False, sharey=True
    )
    for axis, task in zip(axes[0], tasks):
        frame = paired[paired["task"] == task]
        selected = frame.set_index(["acquisition", "candidate_policy"])
        for policy, offset in zip(policies, offsets):
            rows = [selected.loc[(acquisition, policy)] for acquisition in acquisitions]
            means = np.asarray(
                [row["mean_improvement_standard_minus_candidate"] for row in rows]
            )
            lows = np.asarray([row["bootstrap_ci95_low"] for row in rows])
            highs = np.asarray([row["bootstrap_ci95_high"] for row in rows])
            y = np.arange(len(acquisitions), dtype=float) + offset
            significant = (lows > 0) | (highs < 0)
            color = POLICY_COLORS[policy]
            for position, mean, low, high, is_significant in zip(
                y, means, lows, highs, significant
            ):
                axis.plot(
                    [low, high], [position, position], color=color,
                    linewidth=LINE_WIDTH, zorder=2,
                )
                axis.plot(
                    mean,
                    position,
                    marker="o",
                    markerfacecolor=color if is_significant else "white",
                    markeredgecolor=color,
                    markeredgewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    zorder=3,
                )
        axis.axvline(0.0, color=STANDARD_GRAY, linestyle="--", linewidth=1.1)
        axis.set_yticks(range(len(acquisitions)))
        axis.set_yticklabels([ACQUISITION_LABELS[name] for name in acquisitions])
        axis.set_title(f"{TASK_LABELS.get(task, task)} {dimension}D")
        axis.set_xlabel("PE improvement over Standard")
        style_axis(axis, grid_axis="x")
    axes[0][0].invert_yaxis()
    handles = [
        Line2D(
            [0], [0], marker="o", color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy], linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )
        for policy in policies
    ]
    handles.append(
        Line2D(
            [0], [0], marker="o", color=STANDARD_GRAY, markerfacecolor="white",
            markeredgewidth=LINE_WIDTH, label="hollow: 95% CI contains 0",
            linewidth=0, markersize=MARKER_SIZE,
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.text(
        0.5,
        0.02,
        "Thirty paired seeds; intervals are paired bootstrap 95% CIs. "
        "Positive values favour PE; panels use independent scales.",
        ha="center",
        color=STANDARD_GRAY,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.84))
    save_figure(fig, figure_dir, f"fig_yahpo_{dimension}d_endpoint_improvement")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--band", choices=("ci95", "std", "iqr"), default="ci95")
    args = parser.parse_args()
    apply_publication_style()

    history = pd.read_csv(args.input_dir / "history.csv")
    figure_dir = args.input_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    for task, task_frame in history.groupby("task"):
        observed_acquisitions = set(task_frame["acquisition"])
        acquisitions = [
            acquisition
            for acquisition in ACQUISITION_ORDER
            if acquisition in observed_acquisitions
        ]
        acquisitions.extend(sorted(observed_acquisitions - set(acquisitions)))
        fig, axes = plt.subplots(
            1,
            len(acquisitions),
            figsize=(WIDTH_FULL, 4.15),
            squeeze=False,
            sharey=True,
        )
        for axis, acquisition in zip(axes[0], acquisitions):
            frame = task_frame[task_frame["acquisition"] == acquisition]
            observed = set(frame["policy"].unique())
            policies = [policy for policy in POLICY_ORDER if policy in observed]
            policies.extend(sorted(observed - set(policies)))
            for policy in policies:
                selected = frame[frame["policy"] == policy]
                pivot = selected.pivot(index="iteration", columns="seed", values="best_observed_loss")
                center = pivot.mean(axis=1)
                if args.band == "ci95":
                    width = 1.96 * pivot.std(axis=1, ddof=1).fillna(0.0) / math.sqrt(
                        pivot.shape[1]
                    )
                    lower = center - width
                    upper = center + width
                    band_label = "95% CI"
                elif args.band == "std":
                    lower = center - pivot.std(axis=1, ddof=1).fillna(0.0)
                    upper = center + pivot.std(axis=1, ddof=1).fillna(0.0)
                    band_label = "seed SD"
                else:
                    lower = pivot.quantile(0.25, axis=1)
                    upper = pivot.quantile(0.75, axis=1)
                    band_label = "seed IQR"
                linestyle = "--" if policy == "standard" else "-"
                linewidth = 2.1 if policy == "standard" else LINE_WIDTH
                axis.plot(
                    center.index,
                    center,
                    color=POLICY_COLORS[policy],
                    label=POLICY_LABELS[policy],
                    linestyle=linestyle,
                    linewidth=linewidth,
                )
                axis.fill_between(
                    center.index,
                    lower,
                    upper,
                    color=POLICY_COLORS[policy],
                    alpha=0.14,
                    linewidth=0,
                )
            axis.set_title(ACQUISITION_LABELS.get(acquisition, acquisition))
            axis.set_xlabel("Evaluation step")
            style_axis(axis, grid_axis="both")
        axes[0][0].set_ylabel("Best-observed validation log loss")
        handles, labels = axes[0][0].get_legend_handles_labels()
        dimension = int(task_frame["dimension"].iloc[0])
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
        fig.suptitle(
            f"{TASK_LABELS.get(task, task)} {dimension}D: YAHPO XGBoost tuning",
            y=0.88,
        )
        fig.text(
            0.5,
            0.02,
            f"Mean over {pivot.shape[1]} paired seeds; bands show {band_label}.",
            ha="center",
            color=STANDARD_GRAY,
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.80))
        task_slug = TASK_LABELS.get(task, task).replace(" ", "_").replace("-", "_")
        save_figure(
            fig,
            figure_dir,
            f"fig_yahpo_{dimension}d_{task_slug}_trajectories_{args.band}",
        )
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
    dimensions = sorted(set(history["dimension"]))
    dimension = int(dimensions[0]) if len(dimensions) == 1 else 0
    plot_endpoint_improvements(paired, figure_dir, dimension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
