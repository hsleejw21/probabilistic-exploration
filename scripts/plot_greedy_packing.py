#!/usr/bin/env python3
"""Generate the public Greedy Packing figures and compact summary tables."""

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
    WIDTH_TWO_THIRDS,
    apply_publication_style,
    save_figure,
    style_axis,
)


RESULT_DIR = ROOT / "results" / "synthetic" / "greedy_packing"
FIGURE_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"

UNIFORM_BLUE = PE_BLUE
GREEDY_GREEN = "#16845B"
DIRECT_GREEN = "#0C5B3C"
CONTRASTS = (
    ("Uniform over Standard", UNIFORM_BLUE, "o", 0.18),
    ("Greedy over Standard", GREEDY_GREEN, "s", 0.00),
    ("Greedy over Uniform", DIRECT_GREEN, "D", -0.18),
)

OBJECTIVE_ORDER = (
    "bent_cigar_shifted_30d",
    "hartmann6_sparse_30d",
    "rosenbrock_mean_30d",
)
OBJECTIVE_LABELS = {
    "bent_cigar_shifted_30d": "Shifted Bent Cigar",
    "hartmann6_sparse_30d": "Sparse Hartmann",
    "rosenbrock_mean_30d": "Rosenbrock",
}
ACQUISITION_ORDER = ("ucb", "ts", "logei", "mes_gumbel")
ACQUISITION_LABELS = {
    "ucb": "GP-UCB",
    "ts": "GP-TS",
    "logei": "LogEI",
    "mes_gumbel": "MES-Gumbel",
}
DIMENSIONS = (2, 5, 10, 20, 30)
ALPHA_BY_DIMENSION = {2: "1/2", 5: "1", 10: "2/3", 20: "1/2", 30: "5/6"}
VALIDATION_SETTINGS = {10: ("1/2", 4), 30: ("1/2", 16)}


def load_contrasts(path: Path) -> pd.DataFrame:
    """Return all public comparisons under one positive-is-better convention."""
    frame = pd.read_csv(path)
    rows: list[pd.DataFrame] = []
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
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != len(frame):
        raise ValueError(f"Unclassified comparisons in {path}")
    return result


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


def shared_legend(fig, *, y: float = 0.99) -> None:
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
    fig.legend(
        handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, y)
    )


def draw_forest(
    axis,
    frame: pd.DataFrame,
    keys: list,
    labels: list[str],
    *,
    key_column: str,
    title: str | None = None,
) -> None:
    y_positions = np.arange(len(keys))[::-1]
    axis.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", zorder=0)
    for contrast, color, marker, offset in CONTRASTS:
        rows = frame[frame["contrast"] == contrast].set_index(key_column)
        for y, key in zip(y_positions, keys):
            draw_point(axis, rows.loc[key], y + offset, color, marker)
    axis.set_yticks(y_positions, labels)
    axis.set_xlabel("Endpoint regret improvement")
    if title:
        axis.set_title(title)
    style_axis(axis, grid_axis="x")


def plot_30d(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str = "fig_greedy_packing_acquisition_30d",
    note: str = (
        "Ten paired seeds. Positive values favour the first method named; "
        "panels use independent scales."
    ),
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(WIDTH_FULL, 4.45), sharey=True, constrained_layout=False
    )
    for axis, objective in zip(axes, OBJECTIVE_ORDER):
        draw_forest(
            axis,
            frame[frame["objective"] == objective],
            list(ACQUISITION_ORDER),
            [ACQUISITION_LABELS[value] for value in ACQUISITION_ORDER],
            key_column="acquisition",
            title=OBJECTIVE_LABELS[objective],
        )
    shared_legend(fig)
    fig.text(
        0.5, 0.025,
        note,
        ha="center", color=TEXT_SECONDARY,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.22, wspace=0.30)
    save_figure(fig, output_dir, stem)
    plt.close(fig)


YAHPO_TASK_LABELS = {
    "rbv2_xgboost_31": "credit-g (14D)",
    "rbv2_xgboost_40975": "car (14D)",
    "rbv2_xgboost_1464": "blood transfusion (14D)",
    "iaml_super_40981": "OpenML-40981 (28D)",
    "iaml_super_41146": "sylvine (28D)",
}


def load_yahpo(root: Path) -> pd.DataFrame:
    frames = []
    for dimension in (14, 28):
        frame = load_contrasts(root / f"{dimension}d" / "paired_comparisons.csv")
        frame["dimension"] = dimension
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_yahpo(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    note: str,
) -> None:
    fig = plt.figure(figsize=(WIDTH_FULL, 8.2))
    grid = GridSpec(2, 6, figure=fig, hspace=0.68, wspace=0.58)
    specifications = (
        (grid[0, 0:2], "rbv2_xgboost_31"),
        (grid[0, 2:4], "rbv2_xgboost_40975"),
        (grid[0, 4:6], "rbv2_xgboost_1464"),
        (grid[1, 1:3], "iaml_super_40981"),
        (grid[1, 3:5], "iaml_super_41146"),
    )
    for slot, task in specifications:
        draw_forest(
            fig.add_subplot(slot),
            frame[frame["task"] == task],
            list(ACQUISITION_ORDER),
            [ACQUISITION_LABELS[value] for value in ACQUISITION_ORDER],
            key_column="acquisition",
            title=YAHPO_TASK_LABELS[task],
        )
    for axis in fig.axes:
        axis.set_xlabel("Endpoint loss improvement")
    shared_legend(fig)
    fig.text(0.5, 0.015, note, ha="center", color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.10)
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_lunar(path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(path)
    keys = (
        ("decay_uniform_grid_a1_2", "standard"),
        ("decay_greedy_packing_grid_a1_2", "standard"),
        ("decay_greedy_packing_grid_a1_2", "decay_uniform_grid_a1_2"),
    )
    labels = tuple(value[0] for value in CONTRASTS)
    metrics = (
        ("mean_return", "Mean return", 1.0),
        ("episode_success_rate", "Landing rate (percentage points)", 100.0),
        ("bottom_10pct_mean_return", "Worst-10% mean return", 1.0),
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(WIDTH_FULL, 4.0), sharey=True, constrained_layout=False
    )
    for axis, (metric, xlabel, scale) in zip(axes, metrics):
        selected = frame[frame["metric"] == metric]
        axis.axvline(
            0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", zorder=0
        )
        for y, key, (_, color, marker, _) in zip(
            np.arange(3)[::-1], keys, CONTRASTS
        ):
            source = selected[
                (selected["policy"] == key[0])
                & (selected["reference"] == key[1])
            ].iloc[0]
            row = pd.Series(
                {
                    "improvement": scale * source["paired_mean_difference"],
                    "ci_low": scale * source["bootstrap_ci95_lower"],
                    "ci_high": scale * source["bootstrap_ci95_upper"],
                }
            )
            draw_point(axis, row, y, color, marker)
        axis.set_yticks(np.arange(3)[::-1], labels)
        axis.set_xlabel(xlabel)
        style_axis(axis, grid_axis="x")
    fig.text(
        0.5, 0.02,
        "Thirty paired controllers on the same 200 unseen terrains. Positive favours the first method.",
        ha="center", color=TEXT_SECONDARY,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.96, bottom=0.25, wspace=0.30)
    save_figure(fig, output_dir, "fig_greedy_packing_lunar")
    plt.close(fig)


def plot_grid_confirmation(
    synthetic_path: Path,
    lunar_root: Path,
    output_dir: Path,
) -> None:
    synthetic = pd.read_csv(synthetic_path)
    fig = plt.figure(figsize=(WIDTH_FULL, 7.4))
    grid = GridSpec(2, 6, figure=fig, hspace=0.72, wspace=0.62)
    objective_specs = (
        (grid[0, 0:2], "rastrigin", "Rastrigin"),
        (grid[0, 2:4], "ackley", "Ackley"),
        (grid[0, 4:6], "rosenbrock", "Rosenbrock"),
    )
    synthetic_contrasts = (
        ("large_improvement_over_current", "Synthetic: 4x over default", "#C46D2D", "^", 0.10),
        ("large_improvement_over_uniform", "Synthetic: 4x over Uniform", DIRECT_GREEN, "D", -0.10),
    )
    for slot, needle, title in objective_specs:
        axis = fig.add_subplot(slot)
        selected = synthetic[synthetic["objective"].str.contains(needle)]
        axis.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", zorder=0)
        for comparison, _, color, marker, offset in synthetic_contrasts:
            rows = selected[selected["comparison"] == comparison].set_index("dimension")
            for y, dimension in zip(np.arange(3)[::-1], (30, 10, 2)):
                source = rows.loc[dimension]
                draw_point(
                    axis,
                    pd.Series(
                        {
                            "improvement": source["mean_regret_improvement"],
                            "ci_low": source["bootstrap_ci95_lower"],
                            "ci_high": source["bootstrap_ci95_upper"],
                        }
                    ),
                    y + offset,
                    color,
                    marker,
                )
        axis.set_yticks(np.arange(3)[::-1], ["30D", "10D", "2D"])
        axis.set_title(title)
        axis.set_xlabel("Endpoint regret improvement")
        style_axis(axis, grid_axis="x")

    lunar_axis = fig.add_subplot(grid[1, 1:5])
    lunar_axis.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", zorder=0)
    lunar_specs = (
        ("Default: 256+16t", lunar_root / "grid2x" / "paired_comparisons.csv", "greedy_current"),
        ("2x: 512+32t", lunar_root / "grid2x" / "paired_comparisons.csv", "greedy_2x"),
        ("4x: 1024+64t", lunar_root / "grid4x" / "paired_comparisons.csv", "greedy_4x"),
    )
    lunar_contrasts = (
        ("uniform", "Lunar: Greedy over Uniform", DIRECT_GREEN, "D", 0.10),
        ("greedy_current", "Lunar: larger grid over default", "#C46D2D", "^", -0.10),
    )
    for y, (_, path, method) in zip(np.arange(3)[::-1], lunar_specs):
        frame = pd.read_csv(path)
        frame = frame[(frame["method"] == method) & (frame["metric"] == "mean_return")]
        for reference, _, color, marker, offset in lunar_contrasts:
            selected = frame[frame["reference"] == reference]
            if selected.empty:
                continue
            source = selected.iloc[0]
            draw_point(
                lunar_axis,
                pd.Series(
                    {
                        "improvement": source["paired_mean_difference"],
                        "ci_low": source["bootstrap_ci95_lower"],
                        "ci_high": source["bootstrap_ci95_upper"],
                    }
                ),
                y + offset,
                color,
                marker,
            )
    lunar_axis.set_yticks(np.arange(3)[::-1], [value[0] for value in lunar_specs])
    lunar_axis.set_title("Lunar Lander")
    lunar_axis.set_xlabel("Held-out mean-return improvement")
    style_axis(lunar_axis, grid_axis="x")

    handles = [
        Line2D([0], [0], marker=marker, color=color, linewidth=LINE_WIDTH,
               markersize=MARKER_SIZE, label=label)
        for _, label, color, marker, _ in synthetic_contrasts
    ]
    handles.extend(
        Line2D([0], [0], marker=marker, color=color, linewidth=LINE_WIDTH,
               markersize=MARKER_SIZE, label=label)
        for _, label, color, marker, _ in lunar_contrasts
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99))
    fig.text(
        0.5, 0.02,
        "Fifteen fresh paired seeds per panel. Positive values favour the first method named.",
        ha="center", color=TEXT_SECONDARY,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.09)
    save_figure(fig, output_dir, "fig_greedy_grid_confirmation")
    plt.close(fig)


def load_dimension_family(root: Path, dimensions: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for dimension in dimensions:
        frame = load_contrasts(root / f"d{dimension}" / "paired_comparisons.csv")
        frame["dimension"] = dimension
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_dimensions(
    frame: pd.DataFrame,
    dimensions: tuple[int, ...],
    output_dir: Path,
    stem: str,
    note: str,
) -> None:
    fig, axis = plt.subplots(figsize=(WIDTH_TWO_THIRDS, 4.35))
    draw_forest(
        axis,
        frame,
        list(dimensions),
        [f"{dimension}D" for dimension in dimensions],
        key_column="dimension",
    )
    shared_legend(fig, y=1.01)
    fig.text(0.5, 0.025, note, ha="center", color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.78, bottom=0.21)
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_cross_benchmark(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH_FULL, 4.45), sharey=True, constrained_layout=False
    )
    for axis, benchmark, title in zip(
        axes, ("ackley", "rosenbrock"), ("Shifted Ackley", "Rosenbrock")
    ):
        draw_forest(
            axis,
            frame[frame["benchmark"] == benchmark],
            list(DIMENSIONS),
            [f"{dimension}D" for dimension in DIMENSIONS],
            key_column="dimension",
            title=title,
        )
    shared_legend(fig)
    fig.text(
        0.5, 0.025,
        "Five paired seeds. The Rastrigin decay schedule is transferred unchanged; panels use independent scales.",
        ha="center", color=TEXT_SECONDARY,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.78, bottom=0.22, wspace=0.25)
    save_figure(fig, output_dir, "fig_greedy_packing_cross_benchmark")
    plt.close(fig)


def endpoint_means(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    return {
        "standard": float(frame.loc[frame["policy"] == "standard", "mean"].iloc[0]),
        "uniform": float(frame.loc[frame["policy"].str.contains("uniform"), "mean"].iloc[0]),
        "greedy": float(frame.loc[frame["policy"].str.contains("greedy_packing"), "mean"].iloc[0]),
    }


def find_contrast(frame: pd.DataFrame, label: str) -> pd.Series:
    return frame.loc[frame["contrast"] == label].iloc[0]


def format_effect(row: pd.Series) -> str:
    text = f"{float(row['improvement']):+.3f}"
    if row["ci_low"] > 0:
        return rf"\pewin{{{text}}}"
    if row["ci_high"] < 0:
        return rf"\stdwin{{{text}}}"
    return rf"\pesoft{{{text}}}" if row["improvement"] >= 0 else rf"\stdsoft{{{text}}}"


def classification(row: pd.Series) -> str:
    if row["ci_low"] > 0:
        return "better"
    if row["ci_high"] < 0:
        return "worse"
    return "inconclusive"


def write_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Comparison & Better & Inconclusive & Worse \\",
        r"\midrule",
    ]
    for label, _, _, _ in CONTRASTS:
        counts = frame[frame["contrast"] == label].apply(classification, axis=1).value_counts()
        lines.append(
            f"{label} & {counts.get('better', 0)} & {counts.get('inconclusive', 0)} & {counts.get('worse', 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(path, lines)


def write_dimension_table(
    root: Path,
    dimensions: tuple[int, ...],
    alphas: dict[int, str],
    path: Path,
    grid_constants: dict[int, int] | None = None,
) -> None:
    with_grid = grid_constants is not None
    specification = "lrrrrrrr" if with_grid else "lrrrrrr"
    header = (
        r"Dim. & $\alpha$ & $c$ & Standard & Uniform & Greedy & G--S & G--U \\"
        if with_grid else
        r"Dim. & $\alpha$ & Standard & Uniform & Greedy & G--S & G--U \\"
    )
    lines = [rf"\begin{{tabular}}{{@{{}}{specification}@{{}}}}", r"\toprule", header, r"\midrule"]
    for dimension in dimensions:
        directory = root / f"d{dimension}"
        means = endpoint_means(directory / "endpoint_summary.csv")
        comparisons = load_contrasts(directory / "paired_comparisons.csv")
        gs = format_effect(find_contrast(comparisons, "Greedy over Standard"))
        gu = format_effect(find_contrast(comparisons, "Greedy over Uniform"))
        prefix = f"{dimension} & ${alphas[dimension]}$"
        if with_grid:
            prefix += f" & {grid_constants[dimension]}"
        lines.append(
            f"{prefix} & {means['standard']:.3f} & {means['uniform']:.3f} & {means['greedy']:.3f} & {gs} & {gu} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--yahpo-result-dir",
        type=Path,
        default=ROOT / "results" / "hpo_yahpo" / "greedy_packing",
    )
    parser.add_argument(
        "--lunar-result-dir",
        type=Path,
        default=ROOT / "results" / "lunar" / "greedy_packing",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Build only the YAHPO and Lunar endpoint figures used by the current release.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    if args.paper_only:
        plot_yahpo(
            load_yahpo(args.yahpo_result_dir / "default"),
            args.output_dir,
            "fig_greedy_packing_yahpo",
            "Thirty paired seeds. Positive values favour the first method named; panels use independent scales.",
        )
        plot_lunar(
            args.lunar_result_dir / "default" / "paired_comparisons.csv",
            args.output_dir,
        )
        return 0

    acquisition = load_contrasts(
        args.result_dir / "acquisition_30d" / "paired_comparisons.csv"
    )
    dimensions = load_dimension_family(
        args.result_dir / "rastrigin_dimensions", DIMENSIONS
    )
    validation = load_dimension_family(
        args.result_dir / "rastrigin_validation", (10, 30)
    )
    cross_frames = []
    for benchmark in ("ackley", "rosenbrock"):
        frame = load_dimension_family(
            args.result_dir / "cross_benchmark" / benchmark, DIMENSIONS
        )
        frame["benchmark"] = benchmark
        cross_frames.append(frame)
    cross = pd.concat(cross_frames, ignore_index=True)

    plot_30d(acquisition, args.output_dir)
    plot_dimensions(
        dimensions,
        DIMENSIONS,
        args.output_dir,
        "fig_greedy_packing_rastrigin_dimensions",
        "Fifteen paired seeds. Positive values favour the first method named.",
    )
    plot_dimensions(
        validation,
        (10, 30),
        args.output_dir,
        "fig_greedy_packing_rastrigin_validation",
        "Ten fresh paired seeds; settings were fixed using a disjoint seed block.",
    )
    plot_cross_benchmark(cross, args.output_dir)

    plot_30d(
        load_contrasts(
            args.result_dir / "acquisition_30d_grid4x" / "paired_comparisons.csv"
        ),
        args.output_dir,
        stem="fig_greedy_packing_acquisition_30d_grid4x",
        note=(
            "Ten paired seeds; Greedy uses the fourfold grid. Positive values "
            "favour the first method named."
        ),
    )
    plot_yahpo(
        load_yahpo(args.yahpo_result_dir / "default"),
        args.output_dir,
        "fig_greedy_packing_yahpo",
        "Thirty paired seeds. Positive values favour the first method named; panels use independent scales.",
    )
    plot_yahpo(
        load_yahpo(args.yahpo_result_dir / "grid4x_fresh"),
        args.output_dir,
        "fig_yahpo_grid4x_confirmation",
        "Fifteen fresh paired seeds with the fourfold grid. Positive values favour the first method named.",
    )
    plot_lunar(
        args.lunar_result_dir / "default" / "paired_comparisons.csv",
        args.output_dir,
    )
    plot_grid_confirmation(
        args.result_dir / "grid4x_confirmation" / "summary.csv",
        args.lunar_result_dir,
        args.output_dir,
    )

    write_summary_table(
        acquisition, args.table_dir / "tab_greedy_packing_30d_summary.tex"
    )
    write_dimension_table(
        args.result_dir / "rastrigin_dimensions",
        DIMENSIONS,
        ALPHA_BY_DIMENSION,
        args.table_dir / "tab_greedy_packing_rastrigin_dimensions.tex",
    )
    write_dimension_table(
        args.result_dir / "rastrigin_validation",
        (10, 30),
        {dimension: setting[0] for dimension, setting in VALIDATION_SETTINGS.items()},
        args.table_dir / "tab_greedy_packing_rastrigin_validation.tex",
        {dimension: setting[1] for dimension, setting in VALIDATION_SETTINGS.items()},
    )
    write_summary_table(
        cross, args.table_dir / "tab_greedy_packing_cross_benchmark_summary.tex"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
