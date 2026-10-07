#!/usr/bin/env python3
"""Rebuild the released six-benchmark synthetic figures from aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PANELS = (
    ("Trid", 10),
    ("Rosenbrock", 20),
    ("Shifted Bent Cigar", 30),
    ("Sphere", 30),
    ("Griewank", 30),
    ("Sum of Different Powers", 30),
)
ACQUISITIONS = ("ucb", "logei", "ts")
ACQ_LABEL = {"ucb": "GP-UCB", "logei": "LogEI", "ts": "GP-TS"}
ACQ_COLOR = {"ucb": "#0057FF", "logei": "#F04438", "ts": "#00A651"}
ALPHAS = ("1/4", "1/3", "1/2", "2/3", "3/4")


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.1,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_rows(frame: pd.DataFrame, benchmark: str, dimension: int) -> pd.DataFrame:
    return frame[
        frame["benchmark"].eq(benchmark) & frame["dimension"].eq(dimension)
    ]


def plot_main(trajectories: pd.DataFrame, output: Path, fixed: bool = False) -> None:
    configure()
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.28), sharex=True)
    for index, (axis, (benchmark, dimension)) in enumerate(zip(axes.flat, PANELS)):
        panel = panel_rows(trajectories, benchmark, dimension)
        selected: dict[str, str] = {}
        means = panel["mean"].to_numpy()
        uppers = (panel["mean"] + panel["sd"]).to_numpy()
        y_min = max(float(np.nanmin(means)) / 3.0, 1e-6)
        y_max = float(np.nanmax(uppers)) * 1.15
        for acquisition in ACQUISITIONS:
            for method in ("standard", "uniform"):
                rows = panel[
                    panel["acquisition"].eq(acquisition)
                    & panel["method"].eq(method)
                ].sort_values("bo_iteration")
                if len(rows) != 400:
                    raise ValueError(
                        f"Incomplete trajectory for {benchmark} {dimension}D/"
                        f"{acquisition}/{method}"
                    )
                if method == "uniform":
                    selected[acquisition] = "1/2" if fixed else str(rows["alpha"].iloc[0])
                x = rows["bo_iteration"].to_numpy()
                mean = rows["mean"].to_numpy()
                sd = rows["sd"].to_numpy()
                color = ACQ_COLOR[acquisition]
                axis.fill_between(
                    x,
                    np.maximum(mean - sd, y_min),
                    mean + sd,
                    color=color,
                    alpha=0.08 if method == "standard" else 0.14,
                    linewidth=0,
                )
                axis.plot(
                    x,
                    np.maximum(mean, 1e-6),
                    color=color,
                    linestyle=(0, (4.8, 2.0)) if method == "standard" else "-",
                    linewidth=1.25 if method == "standard" else 1.75,
                    alpha=0.88 if method == "standard" else 1.0,
                )
        title_size = 8.2 if benchmark == "Sum of Different Powers" else 9.0
        axis.set_title(
            f"{benchmark} ({dimension}D)", weight="bold", pad=14.5,
            fontsize=title_size,
        )
        axis.text(
            0.5,
            1.025,
            (r"$\alpha=1/2$" if fixed else
             rf"$\alpha^*$: GP-UCB {selected['ucb']} $\cdot$ "
             rf"LogEI {selected['logei']} $\cdot$ GP-TS {selected['ts']}"),
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#111111",
        )
        axis.set(xlim=(1, 400), ylim=(y_min, y_max), yscale="log")
        axis.set_xticks([1, 100, 200, 300, 400])
        axis.grid(which="major", color="#CBD5E1", linewidth=0.43)
        axis.grid(which="minor", axis="y", color="#E2E8F0", linewidth=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if index >= 3:
            axis.set_xlabel("BO evaluations")
        if index % 3 == 0:
            axis.set_ylabel("Mean inference regret")

    fig.subplots_adjust(
        left=0.072, right=0.995, top=0.925, bottom=0.250,
        wspace=0.31, hspace=0.54,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#222222", linewidth=1.9,
                   linestyle=(0, (4.8, 2.0)), label="Standard"),
            Line2D([0], [0], color="#222222", linewidth=1.9,
                   linestyle="-", label="Uniform"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.072), ncol=2,
        frameon=True, edgecolor="#CBD5E1", handlelength=2.2,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=ACQ_COLOR[a], linewidth=1.9,
                   label=ACQ_LABEL[a])
            for a in ACQUISITIONS
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.014), ncol=3,
        frameon=False, handlelength=2.18, fontsize=9.0,
    )
    save(fig, output / ("fig_appendix_synthetic_uniform_alpha_half" if fixed else "fig_appendix_synthetic_selected_alpha"))


def plot_sweep(
    sweep: pd.DataFrame, endpoints: pd.DataFrame, rule: str, output: Path
) -> None:
    configure()
    label = "Uniform" if rule == "Uniform PE" else "Greedy Packing"
    style = "-" if rule == "Uniform PE" else "-."
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.15), sharex=True)
    for index, (axis, (benchmark, dimension)) in enumerate(zip(axes.flat, PANELS)):
        for acquisition in ("GP-UCB", "LogEI", "GP-TS"):
            rows = panel_rows(sweep, benchmark, dimension)
            rows = rows[
                rows["acquisition"].eq(acquisition) & rows["pe_rule"].eq(rule)
            ].copy()
            rows["alpha"] = rows["alpha"].astype(str)
            rows = rows.set_index("alpha").loc[list(ALPHAS)].reset_index()
            mean = rows["mean_final_inference_regret"].to_numpy()
            sd = rows["std_final_inference_regret"].to_numpy()
            acq_key = {"GP-UCB": "ucb", "LogEI": "logei", "GP-TS": "ts"}[acquisition]
            color = ACQ_COLOR[acq_key]
            x = np.arange(len(ALPHAS))
            axis.fill_between(x, np.maximum(mean - sd, 1e-6), mean + sd,
                              color=color, alpha=0.14, linewidth=0)
            axis.plot(x, mean, color=color, linestyle=style, marker="o",
                      markersize=2.4, linewidth=1.25)
            standard = panel_rows(endpoints, benchmark, dimension)
            standard = standard[standard["acquisition"].eq(acquisition)][
                "standard_mean"
            ].iloc[0]
            axis.axhline(
                standard, color=color, linestyle=(0, (4.8, 2.0)),
                linewidth=0.95, alpha=0.58,
            )
        title_size = 7.9 if benchmark == "Sum of Different Powers" else 8.6
        axis.set_title(
            f"{benchmark} ({dimension}D)", weight="bold", pad=3.5,
            fontsize=title_size,
        )
        axis.set_yscale("log")
        axis.set_xticks(np.arange(len(ALPHAS)), [rf"${a}$" for a in ALPHAS])
        axis.grid(which="major", color="#CBD5E1", linewidth=0.42)
        axis.grid(which="minor", axis="y", color="#E2E8F0", linewidth=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        if index >= 3:
            axis.set_xlabel(r"Exploration exponent $\alpha$")
        if index % 3 == 0:
            axis.set_ylabel("Final inference regret")
    fig.subplots_adjust(
        left=0.073, right=0.992, top=0.95, bottom=0.250,
        wspace=0.30, hspace=0.40,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#222222", linestyle=(0, (4.8, 2.0)),
                   linewidth=1.5, label="Standard"),
            Line2D([0], [0], color="#222222", linestyle=style,
                   linewidth=1.7, marker="o", markersize=3.0, label=label),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.069), ncol=2,
        frameon=True, edgecolor="#CBD5E1", handlelength=2.2,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=ACQ_COLOR[a], linewidth=1.8,
                   label=ACQ_LABEL[a])
            for a in ACQUISITIONS
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=3,
        frameon=False, handlelength=2.0,
    )
    slug = "uniform" if rule == "Uniform PE" else "greedy_packing"
    save(fig, output / "fig_main_synthetic_alpha_sweep")


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    default = Path("results/synthetic/six_benchmark_alpha_sweep")
    parser.add_argument("--results-dir", type=Path, default=default)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.results_dir / "figures"
    trajectories = pd.read_csv(args.results_dir / "trajectory_summary.csv")
    sweep = pd.read_csv(args.results_dir / "alpha_sweep_summary.csv", dtype={"alpha": str})
    endpoints = pd.read_csv(args.results_dir / "selected_uniform_endpoints.csv")
    plot_main(trajectories, output)
    plot_sweep(sweep, endpoints, "Uniform PE", output)
    plot_main(pd.read_csv(args.results_dir / "alpha_half_trajectory_summary.csv"), output, fixed=True)


if __name__ == "__main__":
    main()
