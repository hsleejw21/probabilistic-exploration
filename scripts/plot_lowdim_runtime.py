#!/usr/bin/env python3
"""Rebuild the low-dimensional performance and runtime appendix figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("Standard", "Uniform", "MVR")
DISPLAY_METHODS = ("Standard", "Uniform PE", "MVR PE")
OBJECTIVES = (
    "ackley_shifted_narrow_2d",
    "rastrigin_mean_rotated_shifted_2d",
)
TITLES = ("Ackley 2D", "Rastrigin 2D")
COLORS = {"Standard": "#52514E", "Uniform": "#2A78D6", "MVR": "#8C564B"}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D8D7D2")
    axis.grid(which="major", axis="y", color="#CBD5E1", linewidth=0.42)
    axis.grid(which="minor", axis="y", color="#E2E8F0", linewidth=0.25)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#52514E")


def distribution_panel(
    axis: plt.Axes, frame: pd.DataFrame, value_column: str, *, log: bool = False
) -> None:
    wide = frame.pivot(index="seed", columns="method", values=value_column)
    wide = wide.loc[:, list(METHODS)]
    rng = np.random.default_rng(20261006)
    for index, method in enumerate(METHODS):
        values = wide[method].to_numpy(float)
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        axis.scatter(
            index + jitter,
            values,
            s=16,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=2,
        )
        axis.plot(
            [index - 0.13, index + 0.13],
            [np.median(values)] * 2,
            color="#0B0B0B",
            linewidth=1.6,
            solid_capstyle="round",
            zorder=3,
        )
        axis.scatter(
            index,
            values.mean(),
            marker="D",
            s=44,
            color=COLORS[method],
            edgecolor="#0B0B0B",
            linewidth=0.7,
            zorder=4,
        )
    axis.set_xticks(np.arange(len(METHODS)), DISPLAY_METHODS)
    axis.set_xlim(-0.35, 2.35)
    if log:
        axis.set_yscale("log")
    style_axis(axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("results/synthetic/lowdim_mvr_runtime")
    parser.add_argument("--results-dir", type=Path, default=default)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.results_dir / "per_seed.csv")
    observed = set(zip(frame["objective"], frame["method"]))
    expected = {(objective, method) for objective in OBJECTIVES for method in METHODS}
    missing = expected - observed
    if missing:
        raise ValueError(f"Incomplete runtime aggregate; missing groups: {sorted(missing)}")
    output = args.output_dir or args.results_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)

    configure()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 4.35), sharex=True)
    for column, (objective, title) in enumerate(zip(OBJECTIVES, TITLES)):
        part = frame.loc[frame["objective"].eq(objective)]
        distribution_panel(axes[0, column], part, "final_regret", log=True)
        axes[0, column].set_title(title, fontweight="bold", pad=3.5, fontsize=8.6)
        distribution_panel(axes[1, column], part, "runtime")
        axes[1, column].set_xlabel("Query rule")

    axes[0, 0].set_ylabel("Final inference regret")
    axes[1, 0].set_ylabel("End-to-end runtime (s)")
    fig.subplots_adjust(
        left=0.09,
        right=0.992,
        top=0.95,
        bottom=0.12,
        wspace=0.26,
        hspace=0.18,
    )
    stem = output / "fig_appendix_lowdim_mvr_runtime"
    metadata = {"Creator": "probabilistic-exploration", "CreationDate": None}
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
