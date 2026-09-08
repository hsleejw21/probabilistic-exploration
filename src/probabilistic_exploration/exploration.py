"""Probabilistic-exploration schedules from Section 5."""

from __future__ import annotations

import math

import numpy as np


FIXED_PROBABILITIES = {
    "fixed_p010": 0.1,
    "fixed_p020": 0.2,
    "fixed_p030": 0.3,
}

# Feedback-study policies.  ``decay_uniform`` retains its historical name and
# exactly the high-dimensional alpha=2/3 schedule used in the 8/24 results.
FEEDBACK_POLICIES = (
    "standard",
    "fixed_uniform",
    "fixed_mvr",
    "decay_uniform",
    "decay_mvr",
    "decay_uniform_a1",
    "decay_mvr_a1",
    "decay_uniform_a4_3",
    "decay_mvr_a4_3",
)

# The 9/7 dimension--alpha study uses explicit policy names so its common
# log(t + 1) / (t + 1)^alpha schedule does not inherit the historical
# low-dimensional alpha=2/3 compatibility branch used by ``decay_uniform``.
DIMENSION_ALPHA_POLICIES = (
    "decay_uniform_grid_a1_2",
    "decay_mvr_grid_a1_2",
    "decay_uniform_grid_a2_3",
    "decay_mvr_grid_a2_3",
    "decay_uniform_grid_a5_6",
    "decay_mvr_grid_a5_6",
    "decay_uniform_grid_a1",
    "decay_mvr_grid_a1",
    "decay_uniform_grid_a7_6",
    "decay_mvr_grid_a7_6",
    "decay_uniform_grid_a4_3",
    "decay_mvr_grid_a4_3",
    "decay_uniform_grid_a3_2",
    "decay_mvr_grid_a3_2",
)

_DECAY_EXPONENTS = {
    "decay_uniform": 2.0 / 3.0,
    "decay_mvr": 2.0 / 3.0,
    "decay_uniform_a1": 1.0,
    "decay_mvr_a1": 1.0,
    "decay_uniform_a4_3": 4.0 / 3.0,
    "decay_mvr_a4_3": 4.0 / 3.0,
    "decay_uniform_grid_a1_2": 1.0 / 2.0,
    "decay_mvr_grid_a1_2": 1.0 / 2.0,
    "decay_uniform_grid_a2_3": 2.0 / 3.0,
    "decay_mvr_grid_a2_3": 2.0 / 3.0,
    "decay_uniform_grid_a5_6": 5.0 / 6.0,
    "decay_mvr_grid_a5_6": 5.0 / 6.0,
    "decay_uniform_grid_a1": 1.0,
    "decay_mvr_grid_a1": 1.0,
    "decay_uniform_grid_a7_6": 7.0 / 6.0,
    "decay_mvr_grid_a7_6": 7.0 / 6.0,
    "decay_uniform_grid_a4_3": 4.0 / 3.0,
    "decay_mvr_grid_a4_3": 4.0 / 3.0,
    "decay_uniform_grid_a3_2": 3.0 / 2.0,
    "decay_mvr_grid_a3_2": 3.0 / 2.0,
}


def exploration_rule(policy: str) -> str:
    """Return the point-selection rule used on a PE round."""
    if policy == "standard":
        return "none"
    if policy.endswith("_mvr") or "_mvr_" in policy:
        return "mvr"
    return "uniform"


def decay_exponent(policy: str) -> float:
    """Return alpha for a decay policy and NaN for non-decay policies."""
    return float(_DECAY_EXPONENTS.get(policy, float("nan")))


def exploration_probability(policy: str, dim: int, t: int) -> float:
    """Return p_t for a one-based BO round index."""
    if t < 1:
        raise ValueError("t must be one-based and positive")
    if policy == "standard":
        probability = 0.0
    elif policy in FIXED_PROBABILITIES:
        probability = FIXED_PROBABILITIES[policy]
    elif policy in {"fixed_uniform", "fixed_mvr"}:
        probability = 0.2 if dim <= 5 else 0.5
    elif policy in _DECAY_EXPONENTS:
        alpha = _DECAY_EXPONENTS[policy]
        # Preserve the historical low-dimensional decay_uniform formula.  The
        # 8/31 feedback matrix is d>=8 and therefore uses the common formula.
        if (
            policy in {"decay_uniform", "decay_mvr"}
            and dim <= 5
            and math.isclose(alpha, 2.0 / 3.0)
        ):
            probability = math.log(t + 1.0) / t
        else:
            probability = math.log(t + 1.0) / (t + 1.0) ** alpha
    else:
        raise ValueError(f"unknown policy: {policy}")
    return float(np.clip(probability, 0.0, 1.0))
