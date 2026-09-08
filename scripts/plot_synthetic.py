#!/usr/bin/env python3
"""Create paper-style regret figures from canonical CSV results.

Figure 5 runtime is intentionally produced only by ``run_fig5_runtime.py``;
the per-iteration history contains evaluation diagnostics that are outside the
dedicated sequential BO-core timing protocol.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = EXPERIMENT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probabilistic_exploration.benchmarks import BENCHMARKS, FIGURE_ORDER
from probabilistic_exploration.config import (
    AVAILABLE_POLICIES,
    FIGURE_SPECS,
    POLICY_LABELS as CONFIG_POLICY_LABELS,
)
from probabilistic_exploration.plot_style import (
    ACQUISITION_LABELS,
    POLICY_COLORS as COLORS,
    POLICY_LABELS,
    apply_publication_style,
    save_figure,
    style_axis,
)

PLOT_POLICY_ORDER = (
    "standard",
    "fixed_p010",
    "fixed_p020",
    "fixed_p030",
    "fixed_uniform",
    "fixed_mvr",
    "decay_uniform",
    "decay_mvr",
    "decay_uniform_a1",
    "decay_mvr_a1",
    "decay_uniform_a4_3",
    "decay_mvr_a4_3",
)
TRAJECTORY_STEMS = {
    ("ucb", "increasing"): "fig_synthetic_ucb_increasing_trajectories",
    ("ts", "constant"): "fig_synthetic_ts_constant_trajectories",
    ("ucb", "constant"): "fig_synthetic_ucb_constant_trajectories",
    ("ts", "increasing"): "fig_synthetic_ts_increasing_trajectories",
}

def uncertainty_width(values: np.ndarray, mode: str) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1])
    sample_std = np.std(values, axis=0, ddof=1)
    if mode == "std":
        return sample_std
    sem = sample_std / math.sqrt(values.shape[0])
    return sem if mode == "sem" else 1.96 * sem


def _trajectory_matrix(
    frame: pd.DataFrame, value: str, x_column: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if frame.empty:
        return None
    pivot = frame.pivot(index="seed", columns=x_column, values=value)
    pivot = pivot.dropna(axis=1, how="any")
    if pivot.empty:
        return None
    return pivot.columns.to_numpy(dtype=int), pivot.to_numpy(dtype=float)


def plot_regret_grid(
    history: pd.DataFrame,
    output_dir: Path,
    acquisition: str,
    beta_mode: str,
    uncertainty: str,
    x_axis: str,
) -> None:
    selected = history[
        (history["acquisition"] == acquisition) & (history["beta_mode"] == beta_mode)
    ].copy()
    x_column = "iteration"
    x_label = "Evaluation"
    if x_axis == "bo":
        selected = selected[selected["event"] != "initial"]
        x_column = "bo_iteration"
        x_label = "BO iteration"
    if selected.empty:
        return

    present = [name for name in FIGURE_ORDER if name in set(selected["objective"])]
    ncols = 2 if len(present) <= 4 else 3
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.6 * ncols, 4.4 * nrows), squeeze=False
    )
    sensitivity_plot = selected["policy"].nunique() > 3
    for axis, objective in zip(axes.flat, present):
        objective_frame = selected[selected["objective"] == objective]
        observed_policies = set(objective_frame["policy"].astype(str))
        policies = [
            policy for policy in PLOT_POLICY_ORDER if policy in observed_policies
        ]
        policies.extend(sorted(observed_policies - set(policies)))
        for policy in policies:
            matrix = _trajectory_matrix(
                objective_frame[objective_frame["policy"] == policy],
                "inference_regret",
                x_column,
            )
            if matrix is None:
                continue
            iterations, values = matrix
            mean = np.mean(values, axis=0)
            width = uncertainty_width(values, uncertainty)
            axis.plot(
                iterations,
                mean,
                color=COLORS.get(policy),
                label=POLICY_LABELS.get(
                    policy, CONFIG_POLICY_LABELS.get(policy, policy)
                ),
                linewidth=2.5 if policy == "standard" else 1.9,
                linestyle="--" if policy == "standard" else "-",
                zorder=4 if policy == "standard" else 3,
            )
            if not sensitivity_plot:
                axis.fill_between(
                    iterations,
                    np.maximum(mean - width, 0.0),
                    mean + width,
                    color=COLORS.get(policy),
                    alpha=0.18,
                    linewidth=0,
                )
        display = BENCHMARKS[objective].display_name
        acquisition_label = ACQUISITION_LABELS.get(acquisition, acquisition.upper())
        if beta_mode == "native":
            axis.set_title(f"{display}: {acquisition_label}")
        else:
            axis.set_title(
                f"{display}: {acquisition_label} ({beta_mode} $\\beta_t$)"
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(r"$f(x^*) - f(\hat{x}_t)$")
        style_axis(axis, grid_axis="both")
        if not sensitivity_plot:
            axis.legend(fontsize=8)
    for axis in axes.flat[len(present):]:
        axis.set_visible(False)
    if sensitivity_plot:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.0),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    stem = TRAJECTORY_STEMS.get(
        (acquisition, beta_mode),
        f"fig_synthetic_{acquisition}_{beta_mode}_trajectories",
    )
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_paired_delta_grid(
    history: pd.DataFrame,
    output_dir: Path,
    acquisition: str,
    beta_mode: str,
    uncertainty: str,
) -> None:
    """Plot seed-paired regret improvements; positive values favor PE."""
    selected = history[
        (history["acquisition"] == acquisition)
        & (history["beta_mode"] == beta_mode)
        & (history["event"] != "initial")
    ].copy()
    if selected.empty or selected["policy"].nunique() <= 3:
        return
    present = [name for name in FIGURE_ORDER if name in set(selected["objective"])]
    ncols = 2 if len(present) <= 4 else 3
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.6 * ncols, 4.4 * nrows), squeeze=False
    )
    for axis, objective in zip(axes.flat, present):
        frame = selected[selected["objective"] == objective]
        standard = frame[frame["policy"] == "standard"].pivot(
            index="seed", columns="bo_iteration", values="inference_regret"
        )
        observed = set(frame["policy"].astype(str)) - {"standard"}
        policies = [policy for policy in PLOT_POLICY_ORDER if policy in observed]
        for policy in policies:
            policy_values = frame[frame["policy"] == policy].pivot(
                index="seed", columns="bo_iteration", values="inference_regret"
            )
            shared_seeds = standard.index.intersection(policy_values.index)
            shared_steps = standard.columns.intersection(policy_values.columns)
            if shared_seeds.empty or shared_steps.empty:
                continue
            differences = (
                standard.loc[shared_seeds, shared_steps]
                - policy_values.loc[shared_seeds, shared_steps]
            ).to_numpy(dtype=float)
            mean = np.mean(differences, axis=0)
            width = uncertainty_width(differences, uncertainty)
            steps = shared_steps.to_numpy(dtype=int)
            axis.plot(
                steps,
                mean,
                color=COLORS.get(policy),
                label=POLICY_LABELS.get(
                    policy, CONFIG_POLICY_LABELS.get(policy, policy)
                ),
                linewidth=1.9,
            )
            axis.fill_between(
                steps,
                mean - width,
                mean + width,
                color=COLORS.get(policy),
                alpha=0.10,
                linewidth=0,
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(BENCHMARKS[objective].display_name)
        axis.set_xlabel("BO iteration")
        axis.set_ylabel("PE improvement over Standard")
        style_axis(axis, grid_axis="both")
    for axis in axes.flat[len(present):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=9,
    )
    fig.suptitle("Paired improvement: positive values favor PE", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = f"fig_paired_improvement_{acquisition}_{beta_mode}"
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_final_regret_grid(
    history: pd.DataFrame,
    output_dir: Path,
    acquisition: str,
    beta_mode: str,
    uncertainty: str,
) -> None:
    """Plot final mean regret and uncertainty without trajectory overlap."""
    selected = history[
        (history["acquisition"] == acquisition)
        & (history["beta_mode"] == beta_mode)
    ].copy()
    if selected.empty or selected["policy"].nunique() <= 3:
        return
    endpoints = selected.sort_values("iteration").groupby(
        ["objective", "policy", "seed"], as_index=False
    ).tail(1)
    present = [name for name in FIGURE_ORDER if name in set(endpoints["objective"])]
    ncols = 2 if len(present) <= 4 else 3
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.6 * ncols, 4.4 * nrows), squeeze=False
    )
    for axis, objective in zip(axes.flat, present):
        frame = endpoints[endpoints["objective"] == objective]
        observed = set(frame["policy"].astype(str))
        policies = [policy for policy in PLOT_POLICY_ORDER if policy in observed]
        means, widths = [], []
        for policy in policies:
            values = frame[frame["policy"] == policy]["inference_regret"].to_numpy(
                dtype=float
            )
            means.append(float(np.mean(values)))
            widths.append(float(uncertainty_width(values[:, None], uncertainty)[0]))
        positions = np.arange(len(policies))
        for position, mean, width, policy in zip(
            positions, means, widths, policies
        ):
            axis.errorbar(
                position,
                mean,
                yerr=width,
                fmt="o",
                color=COLORS.get(policy, "black"),
                capsize=4,
                markersize=6,
                linewidth=1.5,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                POLICY_LABELS.get(policy, CONFIG_POLICY_LABELS.get(policy, policy))
                for policy in policies
            ],
            rotation=25,
            ha="right",
        )
        axis.set_title(BENCHMARKS[objective].display_name)
        axis.set_ylabel("Final inference regret")
        style_axis(axis, grid_axis="y")
    for axis in axes.flat[len(present):]:
        axis.set_visible(False)
    fig.tight_layout()
    stem = f"fig_final_regret_{acquisition}_{beta_mode}"
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def main() -> int:
    apply_publication_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--uncertainty", choices=["std", "sem", "ci95"], default="std")
    parser.add_argument("--x-axis", choices=["evaluation", "bo"], default="evaluation")
    args = parser.parse_args()

    history_path = args.results_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_path)
    pairs = list(dict.fromkeys(FIGURE_SPECS.values()))
    observed_pairs = sorted(
        set(zip(history["acquisition"].astype(str), history["beta_mode"].astype(str)))
    )
    pairs.extend(pair for pair in observed_pairs if pair not in pairs)
    for acquisition, beta_mode in pairs:
        plot_regret_grid(
            history,
            output_dir,
            acquisition,
            beta_mode,
            args.uncertainty,
            args.x_axis,
        )
        plot_paired_delta_grid(
            history,
            output_dir,
            acquisition,
            beta_mode,
            args.uncertainty,
        )
        plot_final_regret_grid(
            history,
            output_dir,
            acquisition,
            beta_mode,
            args.uncertainty,
        )
    print(f"Figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
