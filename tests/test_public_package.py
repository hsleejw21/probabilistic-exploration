import csv
import json
import math
from pathlib import Path

import numpy as np

from probabilistic_exploration.acquisitions import (
    acquisition_names,
    beta_value,
    discrete_knowledge_gradient,
)
from probabilistic_exploration.config import paper_config
from probabilistic_exploration.benchmarks import BENCHMARKS
from probabilistic_exploration.exploration import (
    GreedyPackingExplorer,
    exploration_probability,
    exploration_rule,
)
from probabilistic_exploration.lunar_benchmark import lunar_controller
from probabilistic_exploration.plot_style import (
    ACQUISITION_LABELS,
    ACQUISITION_ORDER,
    FIXED_ORANGE,
    POLICY_COLORS,
    STANDARD_GRAY,
)


def test_decay_probability_decreases_and_alpha_controls_strength():
    early = exploration_probability("decay_uniform_grid_a1_2", 30, 2)
    late = exploration_probability("decay_uniform_grid_a1_2", 30, 200)
    faster = exploration_probability("decay_uniform_grid_a1", 30, 200)
    assert 0.0 <= faster < late < early <= 1.0


def test_standard_never_explores():
    assert exploration_probability("standard", 30, 1) == 0.0
    assert exploration_rule("standard") == "none"


def test_greedy_packing_uses_the_same_decay_schedule_as_uniform():
    for suffix in ("a1_4", "a1_3", "a1_2", "a2_3", "a3_4"):
        greedy = f"decay_greedy_packing_grid_{suffix}"
        uniform = f"decay_uniform_grid_{suffix}"
        assert exploration_rule(greedy) == "greedy_packing"
        for iteration in (1, 7, 50):
            assert math.isclose(
                exploration_probability(greedy, 30, iteration),
                exploration_probability(uniform, 30, iteration),
            )


def test_current_synthetic_benchmarks_and_alpha_grid_are_registered():
    objectives = {
        "hartmann6_active_6d",
        "griewank_mean_shifted_10d",
        "trid_normalized_10d",
        "rosenbrock_mean_20d",
        "bent_cigar_shifted_30d",
        "sphere_mean_shifted_30d",
        "griewank_mean_shifted_30d",
        "sum_powers_normalized_30d",
    }
    assert objectives <= set(BENCHMARKS)
    for suffix in ("a1_4", "a1_3", "a1_2", "a2_3", "a3_4"):
        assert exploration_rule(f"decay_uniform_grid_{suffix}") == "uniform"
        assert exploration_rule(f"decay_greedy_packing_grid_{suffix}") == "greedy_packing"


def test_greedy_packing_grid_grows_linearly_and_selects_farthest_point():
    explorer = GreedyPackingExplorer(
        dim=2,
        seed=9,
        max_iteration=5,
        grid_initial=8,
        grid_growth=4,
        distance_batch_size=3,
    )
    assert explorer.grid_size(1) == 12
    assert explorer.grid_size(5) == 28

    references = np.asarray([[0.5, 0.5]])
    selected = explorer.select(1, references)
    active = explorer._candidates[: explorer.grid_size(1)]
    distances = np.min(
        np.linalg.norm(active[:, None, :] - references[None, :, :], axis=2),
        axis=1,
    )
    assert np.allclose(selected, active[np.argmax(distances)])


def test_shared_gp_protocol_fixes_unit_signal_variance():
    config = paper_config()
    assert config.initial_signal_variance == 1.0
    assert config.learn_signal_variance is False


def test_theory_aligned_ts_beta_is_logarithmic():
    config = paper_config()
    assert math.isclose(beta_value("logarithmic", 9, config), math.log(10.0))


def test_lunar_controller_action_is_valid():
    action = lunar_controller(np.zeros(8), np.ones(12))
    assert action in {0, 1, 2, 3}


def test_knowledge_gradient_is_registered_and_values_information():
    assert "kg" in acquisition_names()
    score = discrete_knowledge_gradient(
        np.asarray([0.0, 0.0]),
        np.asarray([[1.0], [0.0]]),
        np.asarray([1.0]),
        observation_noise_variance=0.0,
        num_fantasies=32,
        candidate_batch_size=1,
    )
    assert np.isclose(score[0], 1.0 / math.sqrt(2.0 * math.pi), atol=0.01)


def test_public_figures_use_stable_semantic_colours_and_labels():
    assert POLICY_COLORS["standard"] == STANDARD_GRAY
    assert POLICY_COLORS["fixed_p020"] == FIXED_ORANGE
    assert POLICY_COLORS["decay_uniform_a1"] == "#104281"
    assert ACQUISITION_LABELS["ucb"] == "GP-UCB"
    assert ACQUISITION_LABELS["mes_gumbel"] == "MES-Gumbel"
    assert ACQUISITION_ORDER[:4] == ("ucb", "ts", "logei", "mes_gumbel")


def test_completed_yahpo_28d_release_is_audited():
    result_dir = Path(__file__).resolve().parents[1] / "results" / "hpo_yahpo"
    protocol = json.loads((result_dir / "protocol.json").read_text())
    assert protocol["completed_dimensions"] == [14, 28]
    assert protocol["in_progress_dimensions"] == []

    with (result_dir / "paired_improvements_28d.csv").open(newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 24
    assert sum(float(row["bootstrap_ci95_low"]) > 0 for row in comparisons) == 2
    assert sum(float(row["bootstrap_ci95_high"]) < 0 for row in comparisons) == 0

    audit = (result_dir / "audit_28d.txt").read_text()
    assert audit.startswith("PASS ")
    assert "trials=960" in audit
    assert "budget=200" in audit


def test_greedy_application_results_are_complete_and_audited():
    root = Path(__file__).resolve().parents[1] / "results"
    yahpo = root / "hpo_yahpo" / "greedy_packing"
    expected_rows = {14: 36, 28: 24}
    for stage, expected_seeds in (("default", 30), ("grid4x_fresh", 15)):
        for dimension, expected_comparisons in expected_rows.items():
            directory = yahpo / stage / f"{dimension}d"
            protocol = json.loads((directory / "protocol.json").read_text())
            assert protocol["num_seeds"] == expected_seeds
            with (directory / "paired_comparisons.csv").open(newline="") as handle:
                assert len(list(csv.DictReader(handle))) == expected_comparisons
            assert (directory / "audit.txt").read_text().startswith("PASS ")

    lunar = root / "lunar" / "greedy_packing"
    protocol = json.loads((lunar / "protocol.json").read_text())
    assert protocol["paired_training_seeds"] == 30
    assert protocol["heldout_terrains_per_controller"] == 200
    with (lunar / "default" / "paired_comparisons.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 9

    synthetic = root / "synthetic" / "greedy_packing" / "grid4x_confirmation"
    with (synthetic / "summary.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 18


def test_current_synthetic_release_is_complete():
    directory = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "synthetic"
        / "eight_benchmark_alpha_sweep"
    )
    protocol = json.loads((directory / "protocol.json").read_text())
    assert len(protocol["objectives"]) == 8
    assert protocol["num_paired_runs"] == 20
    assert protocol["bo_evaluations"] == 400

    with (directory / "selected_uniform_endpoints.csv").open(newline="") as handle:
        endpoints = list(csv.DictReader(handle))
    assert len(endpoints) == 24
    assert sum(float(row["mean_gain"]) > 0 for row in endpoints) == 23
    assert sum(
        float(row["gain_ci_low"]) > 0 or float(row["gain_ci_high"]) < 0
        for row in endpoints
    ) == 21

    with (directory / "alpha_sweep_summary.csv").open(newline="") as handle:
        sweep = list(csv.DictReader(handle))
    assert len(sweep) == 8 * 3 * 2 * 5
    assert {row["n_seeds"] for row in sweep} == {"20"}

    with (directory / "trajectory_summary.csv").open(newline="") as handle:
        trajectories = list(csv.DictReader(handle))
    assert len(trajectories) == 8 * 3 * 2 * 400
    assert {row["n_seeds"] for row in trajectories} == {"20"}


def test_lowdim_runtime_release_is_complete():
    directory = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "synthetic"
        / "lowdim_mvr_runtime"
    )
    protocol = json.loads((directory / "protocol.json").read_text())
    assert protocol["num_paired_seeds"] == 15
    assert protocol["bo_evaluations"] == 400
    assert protocol["alpha"] == 0.5

    with (directory / "per_seed.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 3 * 15
    assert {row["method"] for row in rows} == {"Standard", "Uniform", "MVR"}
    pairs = {(row["objective"], row["seed"]) for row in rows}
    assert len(pairs) == 2 * 15

    with (directory / "summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 2 * 3
    assert {row["n"] for row in summary} == {"15"}
    with (directory / "paired_comparisons.csv").open(newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 2 * 2 * 3
    assert {row["n"] for row in comparisons} == {"15"}
    assert (directory / "figures" / "fig_appendix_lowdim_mvr_runtime.pdf").is_file()
    assert (directory / "figures" / "fig_appendix_lowdim_mvr_runtime.png").is_file()


def test_released_figure_builder_only_uses_public_entry_points():
    root = Path(__file__).resolve().parents[1]
    builder = root / "scripts" / "build_released_figures.sh"
    text = builder.read_text()
    assert "plot_aistats2027_synthetic.py" in text
    assert "plot_lowdim_runtime.py" in text
    assert "plot_greedy_packing.py" in text
    assert "paper/" not in text
    assert "history.csv" not in text


def test_public_text_artifacts_do_not_contain_server_paths():
    root = Path(__file__).resolve().parents[1]
    extensions = {".csv", ".json", ".md", ".py", ".tex", ".txt", ".tsv"}
    private_markers = (
        "/" + "nfsdata" + "/ho" + "me/",
        "/" + "Users" + "/",
        "hong" + "jungwoo",
        "lgresearch" + ".ai",
    )
    for path in root.rglob("*"):
        excluded = {".git", ".venv", "paper", "__pycache__"}
        if (
            path.is_file()
            and path.suffix in extensions
            and not excluded.intersection(path.parts)
        ):
            text = path.read_text(errors="ignore")
            assert not any(marker in text for marker in private_markers), path
