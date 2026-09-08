"""Lunar Lander controller-tuning benchmark used in the PE experiments.

The controller and parameter domain follow the example shipped with TuRBO.
Each objective evaluation averages a fixed set of terrain seeds so paired BO
policies see exactly the same simulation conditions.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from .benchmarks import Benchmark


Array = np.ndarray


@dataclass(frozen=True)
class LunarSettings:
    task: str = "lunar_lander_12d"
    rollouts: int = 50
    initial_random: float = 1500.0
    reward_scale: float = 0.01
    success_target_raw: float = 200.0

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def lunar_controller(state: Array, weights: Array) -> int:
    """Return one of the four Lunar Lander actions for 12 controller weights."""
    state = np.asarray(state, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (12,):
        raise ValueError(f"Lunar controller requires 12 weights, got {weights.shape}")
    angle_target = np.clip(
        state[0] * weights[0] + state[2] * weights[1],
        -weights[2],
        weights[2],
    )
    hover_target = weights[3] * abs(state[0])
    angle_todo = (angle_target - state[4]) * weights[4] - state[5] * weights[5]
    hover_todo = (hover_target - state[1]) * weights[6] - state[3] * weights[7]
    if bool(state[6]) or bool(state[7]):
        angle_todo = weights[8]
        hover_todo = -state[3] * weights[9]
    if hover_todo > abs(angle_todo) and hover_todo > weights[10]:
        return 2
    if angle_todo < -weights[11]:
        return 3
    if angle_todo > weights[11]:
        return 1
    return 0


class LunarControllerReward:
    def __init__(self, settings: LunarSettings, episode_seeds: tuple[int, ...]):
        try:
            import gymnasium as gym
            from gymnasium.envs.box2d import lunar_lander as lunar_module
        except ImportError as error:
            raise RuntimeError(
                "Install the optional 'lunar' dependencies before running Lunar Lander"
            ) from error
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        lunar_module.INITIAL_RANDOM = settings.initial_random
        self.settings = settings
        self.episode_seeds = episode_seeds
        self.env = gym.make("LunarLander-v3")

    def _rollout(self, weights: Array, episode_seed: int) -> float:
        state, _ = self.env.reset(seed=int(episode_seed))
        total = 0.0
        while True:
            state, reward, terminated, truncated, _ = self.env.step(
                lunar_controller(state, weights)
            )
            total += float(reward)
            if terminated or truncated:
                return total

    def __call__(self, native_x: Array) -> Array:
        output = []
        for weights in np.atleast_2d(np.asarray(native_x, dtype=float)):
            returns = [self._rollout(weights, seed) for seed in self.episode_seeds]
            output.append(self.settings.reward_scale * float(np.mean(returns)))
        return np.asarray(output)

    def close(self) -> None:
        self.env.close()


def lunar_episode_seeds(trial_seed: int, rollouts: int) -> tuple[int, ...]:
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    sequence = np.random.SeedSequence([20260907, int(trial_seed), 1200])
    return tuple(
        int(value) for value in sequence.generate_state(rollouts, dtype=np.uint32)
    )


def build_lunar_benchmark(
    settings: LunarSettings,
    *,
    trial_seed: int,
) -> tuple[Benchmark, dict[str, object]]:
    seeds = lunar_episode_seeds(trial_seed, settings.rollouts)
    reward = LunarControllerReward(settings, seeds)
    reference = settings.reward_scale * settings.success_target_raw
    benchmark = Benchmark(
        name=settings.task,
        display_name="Lunar Lander 12D",
        dim=12,
        lower=(0.0,) * 12,
        upper=(2.0,) * 12,
        reward_native=reward,
        optimum_native=(1.0,) * 12,
        f_star=float(reference),
    )
    metadata = {
        "decision_dimension": 12,
        "episode_seeds": list(seeds),
        "environment": "Gymnasium LunarLander-v3",
        "initial_random": settings.initial_random,
        "reference_value": reference,
        "reference_meaning": "scaled Gymnasium solved-score target, not true optimum",
    }
    return benchmark, metadata


def build_lunar_placeholder(
    settings: LunarSettings,
    *,
    trial_seed: int = 0,
) -> tuple[Benchmark, dict[str, object]]:
    """Build parent-side metadata without opening a Box2D environment."""
    seeds = lunar_episode_seeds(trial_seed, settings.rollouts)
    reference = settings.reward_scale * settings.success_target_raw
    benchmark = Benchmark(
        name=settings.task,
        display_name="Lunar Lander 12D",
        dim=12,
        lower=(0.0,) * 12,
        upper=(2.0,) * 12,
        reward_native=lambda x: np.zeros(len(np.atleast_2d(x))),
        optimum_native=(1.0,) * 12,
        f_star=float(reference),
    )
    metadata = {
        "decision_dimension": 12,
        "episode_seeds": list(seeds),
        "environment": "Gymnasium LunarLander-v3",
        "initial_random": settings.initial_random,
        "reference_value": reference,
        "reference_meaning": "scaled Gymnasium solved-score target, not true optimum",
    }
    return benchmark, metadata


def close_lunar_benchmark(benchmark: Benchmark) -> None:
    close = getattr(benchmark.reward_native, "close", None)
    if callable(close):
        close()
