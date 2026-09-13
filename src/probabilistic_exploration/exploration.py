"""Probabilistic-exploration schedules from Section 5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc


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

GREEDY_PACKING_POLICIES = (
    "decay_greedy_packing_grid_a1_2",
    "decay_greedy_packing_grid_a2_3",
    "decay_greedy_packing_grid_a5_6",
    "decay_greedy_packing_grid_a1",
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
    "decay_greedy_packing_grid_a1_2": 1.0 / 2.0,
    "decay_greedy_packing_grid_a2_3": 2.0 / 3.0,
    "decay_greedy_packing_grid_a5_6": 5.0 / 6.0,
    "decay_greedy_packing_grid_a1": 1.0,
}


def exploration_rule(policy: str) -> str:
    """Return the point-selection rule used on a PE round."""
    if policy == "standard":
        return "none"
    if "_greedy_packing_" in policy:
        return "greedy_packing"
    if policy.endswith("_mvr") or "_mvr_" in policy:
        return "mvr"
    return "uniform"


def _sobol_prefix(n: int, dim: int, seed: int) -> np.ndarray:
    """Return a deterministic nested scrambled-Sobol prefix."""
    if n < 1:
        raise ValueError("Sobol prefix size must be positive")
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    exponent = int(math.ceil(math.log2(n)))
    return sampler.random_base2(exponent)[:n]


@dataclass
class GreedyPackingExplorer:
    """Approximate farthest-point exploration on a growing Sobol grid.

    At BO round ``t``, the active grid contains
    ``grid_initial + ceil(grid_growth * t)`` candidates. The returned point
    maximizes its Euclidean distance to the nearest previously evaluated input
    in the unit cube. Cached distances make the update linear in the number of
    active candidates for observations added since the previous PE round.
    """

    dim: int
    seed: int
    max_iteration: int
    grid_initial: int = 256
    grid_growth: float = 16.0
    distance_batch_size: int = 2048
    _candidates: np.ndarray = field(init=False, repr=False)
    _nearest_sq_distance: np.ndarray = field(init=False, repr=False)
    _active_size: int = field(default=0, init=False, repr=False)
    _reference_count: int = field(default=0, init=False, repr=False)
    last_grid_size: int = field(default=0, init=False)
    last_nearest_distance: float = field(default=float("nan"), init=False)

    def __post_init__(self) -> None:
        if self.dim < 1 or self.max_iteration < 1:
            raise ValueError("dim and max_iteration must be positive")
        if self.grid_initial < 1 or self.grid_growth <= 0.0:
            raise ValueError("grid_initial and grid_growth must be positive")
        if self.distance_batch_size < 1:
            raise ValueError("distance_batch_size must be positive")
        maximum_size = self.grid_size(self.max_iteration)
        self._candidates = _sobol_prefix(maximum_size, self.dim, self.seed)
        self._nearest_sq_distance = np.full(maximum_size, np.inf, dtype=float)

    def grid_size(self, iteration: int) -> int:
        if iteration < 1:
            raise ValueError("iteration must be one-based and positive")
        return int(self.grid_initial + math.ceil(self.grid_growth * iteration))

    def _update_distances(
        self,
        candidate_start: int,
        candidate_stop: int,
        reference_points: np.ndarray,
    ) -> None:
        if candidate_start >= candidate_stop or len(reference_points) == 0:
            return
        references = np.asarray(reference_points, dtype=float)
        reference_sq = np.sum(references * references, axis=1)
        for start in range(candidate_start, candidate_stop, self.distance_batch_size):
            stop = min(start + self.distance_batch_size, candidate_stop)
            candidates = self._candidates[start:stop]
            squared = (
                np.sum(candidates * candidates, axis=1)[:, None]
                + reference_sq[None, :]
                - 2.0 * candidates @ references.T
            )
            np.maximum(squared, 0.0, out=squared)
            nearest = np.min(squared, axis=1)
            self._nearest_sq_distance[start:stop] = np.minimum(
                self._nearest_sq_distance[start:stop], nearest
            )

    def select(self, iteration: int, evaluated_points: np.ndarray) -> np.ndarray:
        """Return the active-grid point farthest from all evaluated points."""
        references = np.atleast_2d(np.asarray(evaluated_points, dtype=float))
        if references.shape[1] != self.dim:
            raise ValueError("evaluated_points has the wrong dimension")
        if len(references) < self._reference_count:
            raise ValueError("evaluated_points must grow monotonically")

        active_size = self.grid_size(iteration)
        if active_size > len(self._candidates):
            raise ValueError("iteration exceeds the configured maximum")
        self._update_distances(
            0, self._active_size, references[self._reference_count :]
        )
        self._update_distances(self._active_size, active_size, references)
        self._active_size = active_size
        self._reference_count = len(references)

        selected_index = int(np.argmax(self._nearest_sq_distance[:active_size]))
        selected_sq_distance = self._nearest_sq_distance[selected_index]
        self.last_grid_size = active_size
        self.last_nearest_distance = float(math.sqrt(selected_sq_distance))
        return self._candidates[selected_index].copy()


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
