#!/usr/bin/env python3
"""Plot best-so-far validation loss for real-world PE experiments."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_ORDER = (
    "standard", "fixed_p010", "fixed_p020", "fixed_p030",
    "fixed_uniform", "decay_uniform_grid_a1_2", "decay_uniform",
    "decay_uniform_a1",
)
COLORS = {
    "standard": "#d62728",
    "fixed_p010": "#9467bd",
    "fixed_p020": "#ff7f0e",
    "fixed_p030": "#17becf",
    "fixed_uniform": "#1f77b4",
    "decay_uniform_grid_a1_2": "#86b6ef",
    "decay_uniform": "#2ca02c",
    "decay_uniform_a1": "#104281",
}
LABELS = {
    "standard": "Standard",
    "fixed_p010": "Fixed p=0.1",
    "fixed_p020": "Fixed p=0.2",
    "fixed_p030": "Fixed p=0.3",
    "fixed_uniform": "Paper Fixed (p=0.2/0.5)",
    "decay_uniform_grid_a1_2": "Decay alpha=1/2",
    "decay_uniform": "Paper Decay",
    "decay_uniform_a1": "Decay alpha=1",
}


def confidence_width(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


def endpoint_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    endpoints: dict[tuple[str, str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (
            row["task"], row["acquisition"], row["policy"], int(row["seed"])
        )
        previous = endpoints.get(key)
        if previous is None or int(row["iteration"]) > int(previous["iteration"]):
            endpoints[key] = row
    return list(endpoints.values())


def plot_endpoint_summary(
    endpoints: list[dict[str, str]],
    output: Path,
    acquisition: str,
    tasks: list[str],
    value_field: str,
    stem_prefix: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(6.2 * len(tasks), 4.8), squeeze=False
    )
    for axis, task in zip(axes[0], tasks):
        relevant = [
            row
            for row in endpoints
            if row["task"] == task and row["acquisition"] == acquisition
        ]
        observed = {row["policy"] for row in relevant}
        policies = [policy for policy in POLICY_ORDER if policy in observed]
        positions = np.arange(len(policies))
        for position, policy in zip(positions, policies):
            values = np.asarray(
                [float(row[value_field]) for row in relevant if row["policy"] == policy],
                dtype=float,
            )
            mean = float(np.mean(values))
            axis.errorbar(
                position,
                mean,
                yerr=confidence_width(values),
                fmt="o",
                color=COLORS.get(policy),
                capsize=4,
                markersize=6,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [LABELS.get(policy, policy) for policy in policies],
            rotation=25,
            ha="right",
        )
        axis.set_title(task.replace("_", " "))
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{acquisition.upper()}: endpoint comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output / f"{stem_prefix}_{acquisition}.png", dpi=180)
    fig.savefig(output / f"{stem_prefix}_{acquisition}.pdf")
    plt.close(fig)


def plot_paired_endpoint_difference(
    endpoints: list[dict[str, str]],
    output: Path,
    acquisition: str,
    tasks: list[str],
) -> None:
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(6.2 * len(tasks), 4.8), squeeze=False
    )
    for axis, task in zip(axes[0], tasks):
        relevant = [
            row
            for row in endpoints
            if row["task"] == task and row["acquisition"] == acquisition
        ]
        standard = {
            int(row["seed"]): float(row["best_validation_loss"])
            for row in relevant
            if row["policy"] == "standard"
        }
        observed = {row["policy"] for row in relevant} - {"standard"}
        policies = [policy for policy in POLICY_ORDER if policy in observed]
        positions = np.arange(len(policies))
        for position, policy in zip(positions, policies):
            differences = np.asarray(
                [
                    float(row["best_validation_loss"])
                    - standard[int(row["seed"])]
                    for row in relevant
                    if row["policy"] == policy and int(row["seed"]) in standard
                ],
                dtype=float,
            )
            axis.errorbar(
                position,
                float(np.mean(differences)),
                yerr=confidence_width(differences),
                fmt="o",
                color=COLORS.get(policy),
                capsize=4,
                markersize=6,
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [LABELS.get(policy, policy) for policy in policies],
            rotation=25,
            ha="right",
        )
        axis.set_title(task.replace("_", " "))
        axis.set_ylabel("Final validation difference vs Standard")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{acquisition.upper()}: negative values favor PE")
    fig.tight_layout()
    fig.savefig(output / f"paired_delta_validation_{acquisition}.png", dpi=180)
    fig.savefig(output / f"paired_delta_validation_{acquisition}.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.result_dir / "figures_ci95"
    output.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    endpoints = endpoint_rows(rows)
    tasks = sorted({row["task"] for row in rows})
    acquisitions = sorted({row["acquisition"] for row in rows})
    for acquisition in acquisitions:
        fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.1), squeeze=False)
        for axis, task in zip(axes[0], tasks):
            groups = defaultdict(list)
            for row in rows:
                if row["task"] == task and row["acquisition"] == acquisition:
                    groups[(row["policy"], int(row["iteration"]))].append(float(row["best_validation_loss"]))
            observed = {policy for policy, _ in groups}
            policies = [policy for policy in POLICY_ORDER if policy in observed]
            policies.extend(sorted(observed - set(policies)))
            for policy in policies:
                iterations = sorted(i for p, i in groups if p == policy)
                if not iterations:
                    continue
                mean = np.asarray([np.mean(groups[(policy, i)]) for i in iterations])
                axis.plot(
                    iterations,
                    mean,
                    color=COLORS.get(policy),
                    label=LABELS.get(policy, policy),
                    linewidth=2.5 if policy == "standard" else 1.9,
                    zorder=4 if policy == "standard" else 3,
                )
            axis.set_title(task.replace("_", " "))
            axis.set_xlabel("Actual model evaluations")
            axis.set_ylabel("Best validation loss")
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.82))
        fig.savefig(output / f"best_validation_loss_{acquisition}.png", dpi=180)
        fig.savefig(output / f"best_validation_loss_{acquisition}.pdf")
        plt.close(fig)
        plot_endpoint_summary(
            endpoints,
            output,
            acquisition,
            tasks,
            "best_validation_loss",
            "final_validation_loss",
            "Final best validation loss",
        )
        plot_endpoint_summary(
            endpoints,
            output,
            acquisition,
            tasks,
            "final_test_loss",
            "final_test_loss",
            "Held-out test loss",
        )
        plot_paired_endpoint_difference(
            endpoints,
            output,
            acquisition,
            tasks,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
