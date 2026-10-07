#!/usr/bin/env python3
"""Rebuild the released YAHPO and Lunar endpoint figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from probabilistic_exploration.plot_style import (
    LINE_WIDTH,
    MARKER_SIZE,
    PE_BLUE,
    TEXT_SECONDARY,
    WIDTH_FULL,
    apply_publication_style,
    save_figure,
    style_axis,
)


UNIFORM_BLUE = PE_BLUE
GREEDY_GREEN = "#16845B"
DIRECT_GREEN = "#0C5B3C"
CONTRASTS = (
    ("Uniform over Standard", UNIFORM_BLUE, "o", 0.18),
    ("Greedy over Standard", GREEDY_GREEN, "s", 0.00),
)
ACQUISITIONS = ("ucb", "ts", "logei")
ACQUISITION_LABELS = {
    "ucb": "GP-UCB",
    "ts": "GP-TS",
    "logei": "LogEI",
    "mes_gumbel": "MES-Gumbel",
}
YAHPO_TASK_LABELS = {
    "rbv2_xgboost_31": "credit-g (14D)",
    "rbv2_xgboost_40975": "car (14D)",
    "rbv2_xgboost_1464": "blood transfusion (14D)",
    "iaml_super_40981": "OpenML-40981 (28D)",
    "iaml_super_41146": "sylvine (28D)",
}


def draw_point(axis, row: pd.Series, y: float, color: str, marker: str) -> None:
    significant = row["ci_low"] > 0.0 or row["ci_high"] < 0.0
    axis.errorbar(
        row["improvement"],
        y,
        xerr=[
            [row["improvement"] - row["ci_low"]],
            [row["ci_high"] - row["improvement"]],
        ],
        fmt=marker,
        markersize=MARKER_SIZE,
        markerfacecolor=color if significant else "white",
        markeredgecolor=color,
        markeredgewidth=LINE_WIDTH,
        color=color,
        capsize=0,
        linewidth=LINE_WIDTH,
        zorder=3,
    )


def load_yahpo_contrasts(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dimension in (14, 28):
        frame = pd.read_csv(root / f"{dimension}d" / "paired_comparisons.csv")
        masks = (
            (
                "Uniform over Standard",
                (frame["baseline_policy"] == "standard")
                & frame["candidate_policy"].str.contains("uniform"),
            ),
            (
                "Greedy over Standard",
                (frame["baseline_policy"] == "standard")
                & frame["candidate_policy"].str.contains("greedy_packing"),
            ),
            (
                "Greedy over Uniform",
                frame["baseline_policy"].str.contains("uniform")
                & frame["candidate_policy"].str.contains("greedy_packing"),
            ),
        )
        for label, mask in masks:
            selected = frame.loc[mask].copy()
            selected["contrast"] = label
            selected["improvement"] = -selected[
                "mean_difference_candidate_minus_baseline"
            ]
            selected["ci_low"] = -selected["bootstrap_ci95_upper"]
            selected["ci_high"] = -selected["bootstrap_ci95_lower"]
            frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def draw_yahpo_panel(axis, frame: pd.DataFrame, task: str) -> None:
    y_positions = np.arange(len(ACQUISITIONS))[::-1]
    axis.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    for contrast, color, marker, offset in CONTRASTS:
        rows = frame[frame["contrast"] == contrast].set_index("acquisition")
        for y, acquisition in zip(y_positions, ACQUISITIONS):
            draw_point(axis, rows.loc[acquisition], y + offset, color, marker)
    axis.set_yticks(
        y_positions, [ACQUISITION_LABELS[value] for value in ACQUISITIONS]
    )
    axis.set_title(YAHPO_TASK_LABELS[task])
    axis.set_xlabel("Endpoint loss improvement")
    style_axis(axis, grid_axis="x")


def plot_yahpo(frame: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(WIDTH_FULL, 3.6))
    grid = GridSpec(1, 2, figure=fig, wspace=0.4)
    specifications = ((grid[0, 0], "rbv2_xgboost_40975"),
                      (grid[0, 1], "iaml_super_41146"))
    for slot, task in specifications:
        draw_yahpo_panel(fig.add_subplot(slot), frame[frame["task"] == task], task)
    handles = [
        Line2D(
            [0], [0], marker=marker, color=color, linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE, label=label,
        )
        for label, color, marker, _ in CONTRASTS
    ]
    handles.append(
        Line2D(
            [0], [0], marker="o", color=TEXT_SECONDARY, linewidth=0,
            markerfacecolor="white", markeredgewidth=LINE_WIDTH,
            markersize=MARKER_SIZE, label="95% CI contains 0",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.text(
        0.5, 0.015,
        "Thirty paired seeds. Positive values favour the first method named; panels use independent scales.",
        ha="center", color=TEXT_SECONDARY,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.23)
    save_figure(fig, output_dir, "fig_appendix_hpo_acquisitions")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yahpo-dir", type=Path,
        default=ROOT / "results" / "hpo_yahpo" / "greedy_packing" / "default",
    )
    parser.add_argument(
        "--yahpo-output", type=Path,
        default=ROOT / "results" / "hpo_yahpo" / "figures",
    )
    args = parser.parse_args()
    args.yahpo_output.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    plot_yahpo(load_yahpo_contrasts(args.yahpo_dir), args.yahpo_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
