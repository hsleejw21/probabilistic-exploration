import math

import numpy as np

from probabilistic_exploration.acquisitions import beta_value
from probabilistic_exploration.config import paper_config
from probabilistic_exploration.exploration import (
    exploration_probability,
    exploration_rule,
)
from probabilistic_exploration.lunar_benchmark import lunar_controller


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
