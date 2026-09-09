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
from probabilistic_exploration.exploration import (
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
    assert protocol["in_progress_dimensions"] == [38]

    with (result_dir / "paired_improvements_28d.csv").open(newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 24
    assert sum(float(row["bootstrap_ci95_low"]) > 0 for row in comparisons) == 2
    assert sum(float(row["bootstrap_ci95_high"]) < 0 for row in comparisons) == 0

    audit = (result_dir / "audit_28d.txt").read_text()
    assert audit.startswith("PASS ")
    assert "trials=960" in audit
    assert "budget=200" in audit
