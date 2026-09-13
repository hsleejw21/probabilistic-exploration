"""Shared visual language for all public experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


FONT_FAMILY = "DejaVu Sans"
TEXT_COLOR = "#0B0B0B"
TEXT_SECONDARY = "#52514E"
GRID_COLOR = "#D8D7D2"
SPINE_COLOR = GRID_COLOR

WIDTH_FULL = 13.8
WIDTH_TWO_THIRDS = 9.2
MARKER_SIZE = 5.5
LINE_WIDTH = 1.7

PE_BLUE = "#2A78D6"
PE_BLUE_LIGHT = "#86B6EF"
PE_BLUE_DARK = "#104281"
FIXED_ORANGE = "#EB6434"
STANDARD_GRAY = TEXT_SECONDARY
STANDARD_RED = "#D94343"
NEUTRAL = "#EFEEEA"

POLICY_COLORS = {
    "standard": STANDARD_GRAY,
    "fixed_p010": "#9467BD",
    "fixed_p020": FIXED_ORANGE,
    "fixed_p030": "#17A5A5",
    "fixed_uniform": PE_BLUE,
    "fixed_mvr": "#2A9D8F",
    "decay_uniform": PE_BLUE,
    "decay_mvr": "#8C564B",
    "decay_uniform_grid_a1_2": PE_BLUE_LIGHT,
    "decay_uniform_grid_a2_3": PE_BLUE,
    "decay_uniform_grid_a5_6": "#1F62AA",
    "decay_uniform_a1": PE_BLUE_DARK,
    "decay_uniform_grid_a1": PE_BLUE_DARK,
    "decay_uniform_grid_a7_6": "#0D416F",
    "decay_uniform_a4_3": "#08345C",
    "decay_uniform_grid_a4_3": "#08345C",
    "decay_uniform_grid_a3_2": "#052744",
    "decay_mvr_a1": "#7A4E9D",
    "decay_mvr_a4_3": "#222222",
    "decay_greedy_packing_grid_a1_2": "#53A67A",
    "decay_greedy_packing_grid_a2_3": "#16845B",
    "decay_greedy_packing_grid_a5_6": "#0F6F4B",
    "decay_greedy_packing_grid_a1": "#0C5B3C",
}

POLICY_LABELS = {
    "standard": "Standard",
    "fixed_p010": r"Fixed $p=0.1$",
    "fixed_p020": r"Fixed $p=0.2$",
    "fixed_p030": r"Fixed $p=0.3$",
    "fixed_uniform": "Fixed Uniform",
    "fixed_mvr": "Fixed MVR",
    "decay_uniform": "Decay Uniform",
    "decay_mvr": r"Decay MVR ($\alpha=2/3$)",
    "decay_uniform_grid_a1_2": r"Decay $\alpha=1/2$",
    "decay_uniform_grid_a2_3": r"Decay $\alpha=2/3$",
    "decay_uniform_grid_a5_6": r"Decay $\alpha=5/6$",
    "decay_uniform_a1": r"Decay $\alpha=1$",
    "decay_uniform_grid_a1": r"Decay $\alpha=1$",
    "decay_uniform_grid_a7_6": r"Decay $\alpha=7/6$",
    "decay_uniform_a4_3": r"Decay $\alpha=4/3$",
    "decay_uniform_grid_a4_3": r"Decay $\alpha=4/3$",
    "decay_uniform_grid_a3_2": r"Decay $\alpha=3/2$",
    "decay_greedy_packing_grid_a1_2": r"Greedy Packing $\alpha=1/2$",
    "decay_greedy_packing_grid_a2_3": r"Greedy Packing $\alpha=2/3$",
    "decay_greedy_packing_grid_a5_6": r"Greedy Packing $\alpha=5/6$",
    "decay_greedy_packing_grid_a1": r"Greedy Packing $\alpha=1$",
}

ACQUISITION_LABELS = {
    "random_search": "Random Search",
    "ucb": "GP-UCB",
    "ts": "GP-TS",
    "ts_rff": "RFF GP-TS",
    "ei": "EI",
    "logei": "LogEI",
    "mes_gumbel": "MES-Gumbel",
    "kg": "KG",
}
ACQUISITION_ORDER = ("ucb", "ts", "logei", "mes_gumbel", "kg")


def apply_publication_style() -> None:
    """Apply one typography and axis style before constructing any figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "font.size": 9,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis, *, grid_axis: str = "y") -> None:
    """Apply the common light grid and spine treatment to one axis."""
    axis.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)


def save_figure(fig, output_dir: Path, stem: str) -> None:
    """Write matching vector and raster versions with shared export settings."""
    pdf_metadata = {
        "Creator": "probabilistic-exploration",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / f"{stem}.pdf", metadata=pdf_metadata)
    fig.savefig(output_dir / f"{stem}.png")
