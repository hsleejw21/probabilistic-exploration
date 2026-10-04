"""The six paper benchmarks, exposed as maximization objectives on [0,1]^d."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Benchmark:
    name: str
    display_name: str
    dim: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    reward_native: Callable[[Array], Array]
    optimum_native: tuple[float, ...]
    f_star: float

    @property
    def bounds(self) -> Array:
        return np.column_stack([self.lower, self.upper])

    def from_unit(self, unit_x: Array) -> Array:
        unit_x = np.atleast_2d(np.asarray(unit_x, dtype=float))
        lower, upper = np.asarray(self.lower), np.asarray(self.upper)
        return lower + unit_x * (upper - lower)

    def to_unit(self, native_x: Array) -> Array:
        native_x = np.atleast_2d(np.asarray(native_x, dtype=float))
        lower, upper = np.asarray(self.lower), np.asarray(self.upper)
        return (native_x - lower) / (upper - lower)

    def evaluate_unit(self, unit_x: Array) -> Array:
        return self.reward_native(self.from_unit(unit_x))


def _bukin6_reward(x: Array) -> Array:
    x = np.atleast_2d(x)
    loss = 100.0 * np.sqrt(np.abs(x[:, 1] - 0.01 * x[:, 0] ** 2))
    loss += 0.01 * np.abs(x[:, 0] + 10.0)
    return -loss


def _michalewicz_reward(x: Array) -> Array:
    x = np.atleast_2d(x)
    indices = np.arange(1, x.shape[1] + 1, dtype=float)
    return np.sum(
        np.sin(x) * np.sin(indices[None, :] * x**2 / math.pi) ** 20,
        axis=1,
    )


def _styblinski_tang_reward(x: Array) -> Array:
    x = np.atleast_2d(x)
    loss = 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1)
    return -loss


def _ackley_reward(x: Array) -> Array:
    x = np.atleast_2d(x)
    loss = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1)))
    loss -= np.exp(np.mean(np.cos(2.0 * math.pi * x), axis=1))
    loss += 20.0 + math.e
    return -loss


def _levy_reward(x: Array) -> Array:
    x = np.atleast_2d(x)
    w = 1.0 + (x - 1.0) / 4.0
    first = np.sin(math.pi * w[:, 0]) ** 2
    middle = np.sum(
        (w[:, :-1] - 1.0) ** 2
        * (1.0 + 10.0 * np.sin(math.pi * w[:, :-1] + 1.0) ** 2),
        axis=1,
    )
    last = (w[:, -1] - 1.0) ** 2
    last *= 1.0 + np.sin(2.0 * math.pi * w[:, -1]) ** 2
    return -(first + middle + last)


def _rastrigin_reward(x: Array) -> Array:
    x = np.atleast_2d(x)
    loss = 10.0 * x.shape[1]
    loss += np.sum(x**2 - 10.0 * np.cos(2.0 * math.pi * x), axis=1)
    return -loss


def _mean_rastrigin_reward(x: Array) -> Array:
    """Dimension-averaged Rastrigin reward for cross-dimensional studies."""
    values = np.atleast_2d(x)
    return _rastrigin_reward(values) / values.shape[1]


def _mean_griewank_reward(x: Array) -> Array:
    """Dimension-scaled Griewank reward with optimum zero at the origin."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    indices = np.sqrt(np.arange(1, values.shape[1] + 1, dtype=float))
    loss = np.sum(values**2, axis=1) / 4000.0
    loss -= np.prod(np.cos(values / indices[None, :]), axis=1)
    loss += 1.0
    return -loss / values.shape[1]


def _mean_rosenbrock_reward(x: Array) -> Array:
    """Mean Rosenbrock-chain reward with optimum zero at the all-ones point."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] < 2:
        raise ValueError("Rosenbrock requires at least two dimensions")
    loss = 100.0 * (values[:, 1:] - values[:, :-1] ** 2) ** 2
    loss += (1.0 - values[:, :-1]) ** 2
    # A fixed factor keeps the high-dimensional target on the same numerical
    # order as the other panel objectives without changing its optimizer.
    return -np.mean(loss, axis=1) / 100.0


def _mean_ellipsoid_reward(x: Array) -> Array:
    """Normalized ill-conditioned ellipsoid reward with condition 1e6."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] < 2:
        raise ValueError("ellipsoid requires at least two dimensions")
    exponents = np.linspace(0.0, 1.0, values.shape[1])
    weights = 1e6**exponents
    loss = np.mean(weights[None, :] * values**2, axis=1) / 1e5
    return -loss


def _mean_sphere_reward(x: Array) -> Array:
    """Dimension-averaged Sphere reward with optimum zero."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    return -np.mean(values**2, axis=1)


def _normalized_bent_cigar_reward(x: Array) -> Array:
    """Scaled Bent Cigar reward with condition number 1e6."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] < 2:
        raise ValueError("Bent Cigar requires at least two dimensions")
    loss = values[:, 0] ** 2 + 1e6 * np.sum(values[:, 1:] ** 2, axis=1)
    return -loss / (1e6 * values.shape[1])


def _mean_levy_reward(x: Array) -> Array:
    """Dimension-scaled Levy reward with optimum zero at the all-ones point."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    return _levy_reward(values) / values.shape[1]


_STYBLINSKI_TANG_PER_DIM_MAX = float(
    _styblinski_tang_reward(np.asarray([[-2.903534]]))[0]
)


def _centered_mean_styblinski_tang_reward(x: Array) -> Array:
    """Mean Styblinski--Tang reward shifted to have maximum zero."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    return (
        _styblinski_tang_reward(values) / values.shape[1]
        - _STYBLINSKI_TANG_PER_DIM_MAX
    )


def _schaffers_f7_reward(x: Array) -> Array:
    """Schaffers F7 reward, scaled by the number of adjacent pairs."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] < 2:
        raise ValueError("Schaffers F7 requires at least two dimensions")
    radii = np.sqrt(values[:, :-1] ** 2 + values[:, 1:] ** 2)
    terms = np.sqrt(radii) * (1.0 + np.sin(50.0 * radii**0.2) ** 2)
    return -np.mean(terms, axis=1) ** 2


def _normalized_dixon_price_reward(x: Array) -> Array:
    """Dixon--Price reward scaled by its uniform-domain mean loss."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] < 2:
        raise ValueError("Dixon--Price requires at least two dimensions")
    indices = np.arange(2, values.shape[1] + 1, dtype=float)
    loss = (values[:, 0] - 1.0) ** 2
    loss += np.sum(
        indices[None, :]
        * (2.0 * values[:, 1:] ** 2 - values[:, :-1]) ** 2,
        axis=1,
    )
    # For independent U[-10,10] coordinates, this is the exact mean loss.
    expected = 103.0 / 3.0 + (24100.0 / 3.0) * np.sum(indices)
    return -loss / expected


def _normalized_powell_reward(x: Array) -> Array:
    """Powell singular reward averaged and scaled over four-variable blocks."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] % 4 != 0:
        raise ValueError("Powell requires a dimension divisible by four")
    blocks = values.reshape(len(values), -1, 4)
    x1, x2, x3, x4 = (blocks[:, :, index] for index in range(4))
    loss = (x1 + 10.0 * x2) ** 2
    loss += 5.0 * (x3 - x4) ** 2
    loss += (x2 - 2.0 * x3) ** 4
    loss += 10.0 * (x1 - x4) ** 4
    return -np.mean(loss, axis=1) / 5000.0


def _normalized_zakharov_reward(x: Array) -> Array:
    """Zakharov reward scaled to remain order one as dimension grows."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    indices = np.arange(1, values.shape[1] + 1, dtype=float)
    linear = np.sum(0.5 * indices[None, :] * values, axis=1)
    loss = np.sum(values**2, axis=1) + linear**2 + linear**4
    # The native domain is [-5,10]. Its nonzero mean makes the fourth-order
    # term grow rapidly; this reference scale is its value at the domain mean.
    reference_linear = 1.25 * np.sum(indices)
    scale = values.shape[1] * 25.0 + reference_linear**2 + reference_linear**4
    return -loss / scale


def _mean_schwefel_reward(x: Array) -> Array:
    """Dimension-averaged Schwefel reward with optimum zero."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    loss = 418.9828872724338
    loss -= np.mean(values * np.sin(np.sqrt(np.abs(values))), axis=1)
    return -loss / 500.0


def _normalized_trid_reward(x: Array) -> Array:
    """Trid reward shifted by the known optimum and dimension normalized."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    dimension = values.shape[1]
    loss = np.sum((values - 1.0) ** 2, axis=1)
    loss -= np.sum(values[:, 1:] * values[:, :-1], axis=1)
    optimum_loss = -dimension * (dimension + 4.0) * (dimension - 1.0) / 6.0
    # The leading uniform-domain contribution is approximately d^5 / 3.
    scale = dimension**5 / 3.0
    return -(loss - optimum_loss) / scale


def _normalized_sum_powers_reward(x: Array) -> Array:
    """Sum of Different Powers reward normalized by its uniform mean."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    exponents = np.arange(2, values.shape[1] + 2, dtype=float)
    loss = np.sum(np.abs(values) ** exponents[None, :], axis=1)
    expected = np.sum(1.0 / (exponents + 1.0))
    return -loss / expected


def _additive_ackley5_reward(x: Array) -> Array:
    """Average of independent five-dimensional Ackley blocks."""
    values = np.atleast_2d(x)
    if values.shape[1] % 5 != 0:
        raise ValueError("additive Ackley requires a dimension divisible by five")
    blocks = values.reshape(len(values), values.shape[1] // 5, 5)
    flat_rewards = _ackley_reward(blocks.reshape(-1, 5))
    return flat_rewards.reshape(len(values), -1).mean(axis=1)


def _fixed_ackley_shift(dim: int) -> Array:
    """Deterministic interior shift used to remove center-optimum bias."""
    coordinates = np.arange(1, dim + 1, dtype=float)
    return 1.75 * np.sin(math.sqrt(2.0) * coordinates)


def _fixed_shift(dim: int, amplitude: float) -> Array:
    """Deterministic non-central shift with a caller-controlled native scale."""
    coordinates = np.arange(1, dim + 1, dtype=float)
    return amplitude * np.sin(math.sqrt(2.0) * coordinates)


def _fixed_rotation(dim: int) -> Array:
    """Return a deterministic orthogonal matrix for rotated objectives."""
    rng = np.random.default_rng(20260815 + dim)
    orthogonal, triangular = np.linalg.qr(rng.standard_normal((dim, dim)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    return orthogonal * signs[None, :]


def _rotated_shifted_reward(
    reward: Callable[[Array], Array], shift: Array, rotation: Array
) -> Callable[[Array], Array]:
    fixed_shift = np.asarray(shift, dtype=float).copy()
    fixed_rotation = np.asarray(rotation, dtype=float).copy()

    def transformed(x: Array) -> Array:
        centered = np.atleast_2d(x) - fixed_shift[None, :]
        return reward(centered @ fixed_rotation)

    return transformed


def _shifted_reward(
    reward: Callable[[Array], Array], shift: Array
) -> Callable[[Array], Array]:
    fixed_shift = np.asarray(shift, dtype=float).copy()

    def shifted(x: Array) -> Array:
        return reward(np.atleast_2d(x) - fixed_shift[None, :])

    return shifted


_HARTMANN6_ALPHA = np.asarray([1.0, 1.2, 3.0, 3.2])
_HARTMANN6_A = np.asarray(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_HARTMANN6_P = 1e-4 * np.asarray(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ]
)
_HARTMANN6_OPTIMUM = np.asarray(
    [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
)


def _hartmann6_score(x: Array) -> Array:
    values = np.atleast_2d(np.asarray(x, dtype=float))
    squared = (values[:, None, :] - _HARTMANN6_P[None, :, :]) ** 2
    exponent = -np.sum(_HARTMANN6_A[None, :, :] * squared, axis=2)
    return np.sum(_HARTMANN6_ALPHA[None, :] * np.exp(exponent), axis=1)


_HARTMANN6_MAX = float(_hartmann6_score(_HARTMANN6_OPTIMUM)[0])


def _sparse_hartmann6_reward(active: tuple[int, ...]) -> Callable[[Array], Array]:
    """Return a centered Hartmann-6 reward on fixed active coordinates."""
    if len(active) != 6:
        raise ValueError("sparse Hartmann requires exactly six active coordinates")

    def reward(x: Array) -> Array:
        values = np.atleast_2d(x)
        return _hartmann6_score(values[:, active]) - _HARTMANN6_MAX

    return reward


def _sparse_hartmann_optimum(dim: int, active: tuple[int, ...]) -> tuple[float, ...]:
    point = np.full(dim, 0.5)
    point[np.asarray(active)] = _HARTMANN6_OPTIMUM
    return tuple(float(value) for value in point)


def _additive_hartmann6_reward(x: Array) -> Array:
    """Centered sum of independent Hartmann-6 blocks."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] % 6 != 0:
        raise ValueError("additive Hartmann-6 requires a multiple of six dimensions")
    blocks = values.reshape(len(values), -1, 6)
    scores = _hartmann6_score(blocks.reshape(-1, 6)).reshape(len(values), -1)
    return np.sum(scores - _HARTMANN6_MAX, axis=1)


_CURRIN_OPTIMUM = np.asarray([13.0 / 60.0, 0.0])


def _currin_exponential_score(x: Array) -> Array:
    values = np.atleast_2d(np.asarray(x, dtype=float))
    x1, x2 = values[:, 0], values[:, 1]
    exponential = np.zeros_like(x2)
    positive = x2 > 0.0
    exponential[positive] = np.exp(-1.0 / (2.0 * x2[positive]))
    numerator = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
    denominator = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0
    return (1.0 - exponential) * numerator / denominator


_CURRIN_MAX = float(_currin_exponential_score(_CURRIN_OPTIMUM)[0])


def _additive_currin_reward(x: Array) -> Array:
    """Centered sum of independent two-dimensional Currin blocks."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] % 2 != 0:
        raise ValueError("additive Currin requires an even dimension")
    blocks = values.reshape(len(values), -1, 2)
    scores = _currin_exponential_score(blocks.reshape(-1, 2)).reshape(
        len(values), -1
    )
    return np.sum(scores - _CURRIN_MAX, axis=1)


_PARK2_OPTIMUM = np.asarray([1.0, 1.0, 1.0, 0.0])


def _park2_score(x: Array) -> Array:
    values = np.atleast_2d(np.asarray(x, dtype=float))
    return (
        (2.0 / 3.0) * np.exp(values[:, 0] + values[:, 1])
        - values[:, 3] * np.sin(values[:, 2])
        + values[:, 2]
    )


_PARK2_MAX = float(_park2_score(_PARK2_OPTIMUM)[0])


def _additive_park2_reward(x: Array) -> Array:
    """Centered sum of independent four-dimensional Park2 blocks."""
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if values.shape[1] % 4 != 0:
        raise ValueError("additive Park2 requires a multiple of four dimensions")
    blocks = values.reshape(len(values), -1, 4)
    scores = _park2_score(blocks.reshape(-1, 4)).reshape(len(values), -1)
    return np.sum(scores - _PARK2_MAX, axis=1)


BENCHMARKS: dict[str, Benchmark] = {
    "bukin6_2d": Benchmark(
        "bukin6_2d", "Bukin6 2D", 2,
        (-15.0, -3.0), (-5.0, 3.0), _bukin6_reward,
        (-10.0, 1.0), 0.0,
    ),
    "michalewicz_2d": Benchmark(
        "michalewicz_2d", "Michalewicz 2D", 2,
        (0.0, 0.0), (math.pi, math.pi), _michalewicz_reward,
        (2.20290552, 1.57079633), 1.801303410098553,
    ),
    "styblinski_tang_5d": Benchmark(
        "styblinski_tang_5d", "Styblinski–Tang 5D", 5,
        (-5.0,) * 5, (5.0,) * 5, _styblinski_tang_reward,
        (-2.903534,) * 5, 195.830828518857,
    ),
    "ackley_6d": Benchmark(
        "ackley_6d", "Ackley 6D", 6,
        (-32.768,) * 6, (32.768,) * 6, _ackley_reward,
        (0.0,) * 6, 0.0,
    ),
    "levy_7d": Benchmark(
        "levy_7d", "Levy 7D", 7,
        (-10.0,) * 7, (10.0,) * 7, _levy_reward,
        (1.0,) * 7, 0.0,
    ),
    "rastrigin_8d": Benchmark(
        "rastrigin_8d", "Rastrigin 8D", 8,
        (-5.12,) * 8, (5.12,) * 8, _rastrigin_reward,
        (0.0,) * 8, 0.0,
    ),
}


# Intermediate-dimensional benchmarks requested for the feedback study. The
# additive constructions match Kandasamy et al. (AISTATS 2018): Hartmann-6 is
# repeated three times, Currin seven times, and Park2 four times.
_ACKLEY10_SHIFT = _fixed_ackley_shift(10)
BENCHMARKS.update(
    {
        "ackley_wide_10d": Benchmark(
            "ackley_wide_10d",
            "Ackley 10D on [-32.768,32.768]",
            10,
            (-32.768,) * 10,
            (32.768,) * 10,
            _ackley_reward,
            (0.0,) * 10,
            0.0,
        ),
        "ackley_narrow_10d": Benchmark(
            "ackley_narrow_10d",
            "Ackley 10D on [-5,5]",
            10,
            (-5.0,) * 10,
            (5.0,) * 10,
            _ackley_reward,
            (0.0,) * 10,
            0.0,
        ),
        "ackley_shifted_narrow_10d": Benchmark(
            "ackley_shifted_narrow_10d",
            "Shifted Ackley 10D on [-5,5]",
            10,
            (-5.0,) * 10,
            (5.0,) * 10,
            _shifted_reward(_ackley_reward, _ACKLEY10_SHIFT),
            tuple(float(value) for value in _ACKLEY10_SHIFT),
            0.0,
        ),
        # Oracle active-subspace control for the Sparse Hartmann diagnostic.
        # It uses the identical centered Hartmann-6 objective as the 20D/30D
        # sparse variants, without their inactive coordinates.
        "hartmann6_active_6d": Benchmark(
            "hartmann6_active_6d",
            "Hartmann-6 active subspace (6D)",
            6,
            (0.0,) * 6,
            (1.0,) * 6,
            _sparse_hartmann6_reward(tuple(range(6))),
            tuple(float(value) for value in _HARTMANN6_OPTIMUM),
            0.0,
        ),
        "hartmann18_additive": Benchmark(
            "hartmann18_additive",
            "Additive Hartmann18",
            18,
            (0.0,) * 18,
            (1.0,) * 18,
            _additive_hartmann6_reward,
            tuple(float(value) for value in np.tile(_HARTMANN6_OPTIMUM, 3)),
            0.0,
        ),
        "currinexp14_additive": Benchmark(
            "currinexp14_additive",
            "Additive CurrinExp-14",
            14,
            (0.0,) * 14,
            (1.0,) * 14,
            _additive_currin_reward,
            tuple(float(value) for value in np.tile(_CURRIN_OPTIMUM, 7)),
            0.0,
        ),
        "park2_16_additive": Benchmark(
            "park2_16_additive",
            "Additive Park2-16",
            16,
            (0.0,) * 16,
            (1.0,) * 16,
            _additive_park2_reward,
            tuple(float(value) for value in np.tile(_PARK2_OPTIMUM, 4)),
            0.0,
        ),
    }
)

FEEDBACK_INTERMEDIATE_BENCHMARKS = (
    "hartmann18_additive",
    "currinexp14_additive",
    "park2_16_additive",
)


# Additional scalable candidates from the SFU Virtual Library of Simulation
# Experiments. Each reward is centered at zero and scaled so that a random
# point has order-one regret under the native domain. These candidates are
# screened separately from the fixed six-objective paper comparison.
_DIXON_PRICE_DIM = 10
_DIXON_PRICE_OPTIMUM = tuple(
    1.0
    if index == 1
    else 2.0 ** (-(2.0**index - 2.0) / 2.0**index)
    for index in range(1, _DIXON_PRICE_DIM + 1)
)
_POWELL_DIM = 12
_ZAKHAROV_DIM = 20
_SCHWEFEL_DIM = 20
_SCHWEFEL_OPTIMUM = 420.9687462275036
_TRID_DIM = 10
_TRID_OPTIMUM = tuple(
    float(index * (_TRID_DIM + 1 - index))
    for index in range(1, _TRID_DIM + 1)
)
_SUM_POWERS_DIM = 30

BENCHMARKS.update(
    {
        "dixon_price_normalized_10d": Benchmark(
            "dixon_price_normalized_10d",
            "Dixon--Price 10D (normalized)",
            _DIXON_PRICE_DIM,
            (-10.0,) * _DIXON_PRICE_DIM,
            (10.0,) * _DIXON_PRICE_DIM,
            _normalized_dixon_price_reward,
            _DIXON_PRICE_OPTIMUM,
            0.0,
        ),
        "powell_normalized_12d": Benchmark(
            "powell_normalized_12d",
            "Powell 12D (normalized)",
            _POWELL_DIM,
            (-4.0,) * _POWELL_DIM,
            (5.0,) * _POWELL_DIM,
            _normalized_powell_reward,
            (0.0,) * _POWELL_DIM,
            0.0,
        ),
        "zakharov_normalized_20d": Benchmark(
            "zakharov_normalized_20d",
            "Zakharov 20D (normalized)",
            _ZAKHAROV_DIM,
            (-5.0,) * _ZAKHAROV_DIM,
            (10.0,) * _ZAKHAROV_DIM,
            _normalized_zakharov_reward,
            (0.0,) * _ZAKHAROV_DIM,
            0.0,
        ),
        "schwefel_mean_20d": Benchmark(
            "schwefel_mean_20d",
            "Schwefel 20D (normalized mean)",
            _SCHWEFEL_DIM,
            (-500.0,) * _SCHWEFEL_DIM,
            (500.0,) * _SCHWEFEL_DIM,
            _mean_schwefel_reward,
            (_SCHWEFEL_OPTIMUM,) * _SCHWEFEL_DIM,
            0.0,
        ),
        "trid_normalized_10d": Benchmark(
            "trid_normalized_10d",
            "Trid 10D (normalized)",
            _TRID_DIM,
            (-float(_TRID_DIM**2),) * _TRID_DIM,
            (float(_TRID_DIM**2),) * _TRID_DIM,
            _normalized_trid_reward,
            _TRID_OPTIMUM,
            0.0,
        ),
        "sum_powers_normalized_30d": Benchmark(
            "sum_powers_normalized_30d",
            "Sum of Different Powers 30D (normalized)",
            _SUM_POWERS_DIM,
            (-1.0,) * _SUM_POWERS_DIM,
            (1.0,) * _SUM_POWERS_DIM,
            _normalized_sum_powers_reward,
            (0.0,) * _SUM_POWERS_DIM,
            0.0,
        ),
    }
)

ADDITIONAL_SFU_SCREEN = (
    "dixon_price_normalized_10d",
    "powell_normalized_12d",
    "zakharov_normalized_20d",
    "schwefel_mean_20d",
    "trid_normalized_10d",
    "sum_powers_normalized_30d",
)


# Cross-dimensional panels for the 9/7 decay-exponent study. The existing
# high-dimensional block below already registers the 20D and 30D variants;
# these loops add the missing low/intermediate dimensions. Ackley is shifted
# away from the center to avoid making its known optimum unusually easy, while
# Rosenbrock is already nonseparable and has an interior noncentral optimum.
for _dimension in (2, 5, 10):
    _rotated_rastrigin_shift = _fixed_shift(_dimension, 1.25)
    _rotation = _fixed_rotation(_dimension)
    BENCHMARKS[f"rastrigin_mean_rotated_shifted_{_dimension}d"] = Benchmark(
        f"rastrigin_mean_rotated_shifted_{_dimension}d",
        f"Rotated shifted Rastrigin {_dimension}D (mean)",
        _dimension,
        (-5.12,) * _dimension,
        (5.12,) * _dimension,
        _rotated_shifted_reward(
            _mean_rastrigin_reward,
            _rotated_rastrigin_shift,
            _rotation,
        ),
        tuple(float(value) for value in _rotated_rastrigin_shift),
        0.0,
    )

for _dimension in (2, 5):
    _ackley_shift = _fixed_ackley_shift(_dimension)
    BENCHMARKS[f"ackley_shifted_narrow_{_dimension}d"] = Benchmark(
        f"ackley_shifted_narrow_{_dimension}d",
        f"Shifted Ackley {_dimension}D on [-5,5]",
        _dimension,
        (-5.0,) * _dimension,
        (5.0,) * _dimension,
        _shifted_reward(_ackley_reward, _ackley_shift),
        tuple(float(value) for value in _ackley_shift),
        0.0,
    )

for _dimension in (2, 5, 10):
    BENCHMARKS[f"rosenbrock_mean_{_dimension}d"] = Benchmark(
        f"rosenbrock_mean_{_dimension}d",
        f"Rosenbrock {_dimension}D (mean)",
        _dimension,
        (-2.048,) * _dimension,
        (2.048,) * _dimension,
        _mean_rosenbrock_reward,
        (1.0,) * _dimension,
        0.0,
    )


# The 10D Griewank panel is used by the main six-objective comparison and by
# the acquisition-wise alpha sweeps. Keep it alongside the lower-dimensional
# registrations because the high-dimensional loop below covers only 20D/30D.
_GRIEWANK10_SHIFT = _fixed_shift(10, 200.0)
BENCHMARKS["griewank_mean_shifted_10d"] = Benchmark(
    "griewank_mean_shifted_10d",
    "Shifted Griewank 10D (mean)",
    10,
    (-600.0,) * 10,
    (600.0,) * 10,
    _shifted_reward(_mean_griewank_reward, _GRIEWANK10_SHIFT),
    tuple(float(value) for value in _GRIEWANK10_SHIFT),
    0.0,
)


# Controlled high-dimensional extension. Ackley and Rastrigin use the standard
# Surjan domains. Experiments pass gp_input_space="unit", so the objective is
# evaluated on the native domain while the GP geometry is shared on [0,1]^d.
_HD_ACTIVE_DIMS = {
    20: (1, 4, 7, 10, 13, 16),
    30: (2, 7, 12, 17, 22, 27),
}

for _dimension, _active in _HD_ACTIVE_DIMS.items():
    _ackley_shift = _fixed_ackley_shift(_dimension)
    _rastrigin_shift = _fixed_shift(_dimension, 1.75)
    _griewank_shift = _fixed_shift(_dimension, 200.0)
    _ellipsoid_shift = _fixed_shift(_dimension, 1.75)
    _sphere_shift = _fixed_shift(_dimension, 1.5)
    _bent_cigar_shift = _fixed_shift(_dimension, 1.5)
    _rotated_rastrigin_shift = _fixed_shift(_dimension, 1.25)
    _levy_shift = _fixed_shift(_dimension, 2.0)
    _styblinski_shift = _fixed_shift(_dimension, 0.75)
    _schaffers_shift = _fixed_shift(_dimension, 1.5)
    _rotation = _fixed_rotation(_dimension)
    BENCHMARKS.update(
        {
            f"ackley_{_dimension}d": Benchmark(
                f"ackley_{_dimension}d",
                f"Ackley {_dimension}D",
                _dimension,
                (-32.768,) * _dimension,
                (32.768,) * _dimension,
                _ackley_reward,
                (0.0,) * _dimension,
                0.0,
            ),
            f"rastrigin_mean_{_dimension}d": Benchmark(
                f"rastrigin_mean_{_dimension}d",
                f"Rastrigin {_dimension}D (mean)",
                _dimension,
                (-5.12,) * _dimension,
                (5.12,) * _dimension,
                _mean_rastrigin_reward,
                (0.0,) * _dimension,
                0.0,
            ),
            f"hartmann6_sparse_{_dimension}d": Benchmark(
                f"hartmann6_sparse_{_dimension}d",
                f"Sparse Hartmann-6 in {_dimension}D",
                _dimension,
                (0.0,) * _dimension,
                (1.0,) * _dimension,
                _sparse_hartmann6_reward(_active),
                _sparse_hartmann_optimum(_dimension, _active),
                0.0,
            ),
            f"ackley5_additive_{_dimension}d": Benchmark(
                f"ackley5_additive_{_dimension}d",
                f"Additive Ackley(5) in {_dimension}D",
                _dimension,
                (-32.768,) * _dimension,
                (32.768,) * _dimension,
                _additive_ackley5_reward,
                (0.0,) * _dimension,
                0.0,
            ),
            f"ackley_narrow_{_dimension}d": Benchmark(
                f"ackley_narrow_{_dimension}d",
                f"Ackley {_dimension}D on [-5,5]",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _ackley_reward,
                (0.0,) * _dimension,
                0.0,
            ),
            f"ackley5_additive_narrow_{_dimension}d": Benchmark(
                f"ackley5_additive_narrow_{_dimension}d",
                f"Additive Ackley(5) in {_dimension}D on [-5,5]",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _additive_ackley5_reward,
                (0.0,) * _dimension,
                0.0,
            ),
            f"ackley_shifted_narrow_{_dimension}d": Benchmark(
                f"ackley_shifted_narrow_{_dimension}d",
                f"Shifted Ackley {_dimension}D on [-5,5]",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(_ackley_reward, _ackley_shift),
                tuple(float(value) for value in _ackley_shift),
                0.0,
            ),
            f"ackley5_additive_shifted_narrow_{_dimension}d": Benchmark(
                f"ackley5_additive_shifted_narrow_{_dimension}d",
                f"Shifted additive Ackley(5) in {_dimension}D on [-5,5]",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(_additive_ackley5_reward, _ackley_shift),
                tuple(float(value) for value in _ackley_shift),
                0.0,
            ),
            f"rastrigin_mean_shifted_{_dimension}d": Benchmark(
                f"rastrigin_mean_shifted_{_dimension}d",
                f"Shifted Rastrigin {_dimension}D (mean)",
                _dimension,
                (-5.12,) * _dimension,
                (5.12,) * _dimension,
                _shifted_reward(_mean_rastrigin_reward, _rastrigin_shift),
                tuple(float(value) for value in _rastrigin_shift),
                0.0,
            ),
            f"griewank_mean_shifted_{_dimension}d": Benchmark(
                f"griewank_mean_shifted_{_dimension}d",
                f"Shifted Griewank {_dimension}D (mean)",
                _dimension,
                (-600.0,) * _dimension,
                (600.0,) * _dimension,
                _shifted_reward(_mean_griewank_reward, _griewank_shift),
                tuple(float(value) for value in _griewank_shift),
                0.0,
            ),
            f"rosenbrock_mean_{_dimension}d": Benchmark(
                f"rosenbrock_mean_{_dimension}d",
                f"Rosenbrock {_dimension}D (mean)",
                _dimension,
                (-2.048,) * _dimension,
                (2.048,) * _dimension,
                _mean_rosenbrock_reward,
                (1.0,) * _dimension,
                0.0,
            ),
            f"ellipsoid_mean_shifted_{_dimension}d": Benchmark(
                f"ellipsoid_mean_shifted_{_dimension}d",
                f"Shifted ellipsoid {_dimension}D (normalized mean)",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(_mean_ellipsoid_reward, _ellipsoid_shift),
                tuple(float(value) for value in _ellipsoid_shift),
                0.0,
            ),
            f"sphere_mean_shifted_{_dimension}d": Benchmark(
                f"sphere_mean_shifted_{_dimension}d",
                f"Shifted Sphere {_dimension}D (mean)",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(_mean_sphere_reward, _sphere_shift),
                tuple(float(value) for value in _sphere_shift),
                0.0,
            ),
            f"bent_cigar_shifted_{_dimension}d": Benchmark(
                f"bent_cigar_shifted_{_dimension}d",
                f"Shifted Bent Cigar {_dimension}D",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(
                    _normalized_bent_cigar_reward, _bent_cigar_shift
                ),
                tuple(float(value) for value in _bent_cigar_shift),
                0.0,
            ),
            f"rastrigin_mean_rotated_shifted_{_dimension}d": Benchmark(
                f"rastrigin_mean_rotated_shifted_{_dimension}d",
                f"Rotated shifted Rastrigin {_dimension}D (mean)",
                _dimension,
                (-5.12,) * _dimension,
                (5.12,) * _dimension,
                _rotated_shifted_reward(
                    _mean_rastrigin_reward,
                    _rotated_rastrigin_shift,
                    _rotation,
                ),
                tuple(float(value) for value in _rotated_rastrigin_shift),
                0.0,
            ),
            f"levy_mean_shifted_{_dimension}d": Benchmark(
                f"levy_mean_shifted_{_dimension}d",
                f"Shifted Levy {_dimension}D (mean)",
                _dimension,
                (-10.0,) * _dimension,
                (10.0,) * _dimension,
                _shifted_reward(_mean_levy_reward, _levy_shift),
                tuple(float(1.0 + value) for value in _levy_shift),
                0.0,
            ),
            f"styblinski_tang_mean_shifted_{_dimension}d": Benchmark(
                f"styblinski_tang_mean_shifted_{_dimension}d",
                f"Shifted Styblinski--Tang {_dimension}D (mean)",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(
                    _centered_mean_styblinski_tang_reward,
                    _styblinski_shift,
                ),
                tuple(float(-2.903534 + value) for value in _styblinski_shift),
                0.0,
            ),
            f"schaffers_f7_shifted_{_dimension}d": Benchmark(
                f"schaffers_f7_shifted_{_dimension}d",
                f"Shifted Schaffers F7 {_dimension}D",
                _dimension,
                (-5.0,) * _dimension,
                (5.0,) * _dimension,
                _shifted_reward(_schaffers_f7_reward, _schaffers_shift),
                tuple(float(value) for value in _schaffers_shift),
                0.0,
            ),
        }
    )


HIGH_DIMENSIONAL_CORE: dict[int, tuple[str, ...]] = {
    dimension: (
        f"ackley_{dimension}d",
        f"rastrigin_mean_{dimension}d",
        f"hartmann6_sparse_{dimension}d",
        f"ackley5_additive_{dimension}d",
    )
    for dimension in _HD_ACTIVE_DIMS
}

# Primary suite for the calibrated high-dimensional GP profile. The original
# wide-domain Ackley variants remain in HIGH_DIMENSIONAL_CORE as stress tests.
HIGH_DIMENSIONAL_CALIBRATED_CORE: dict[int, tuple[str, ...]] = {
    dimension: (
        f"ackley_narrow_{dimension}d",
        f"rastrigin_mean_{dimension}d",
        f"hartmann6_sparse_{dimension}d",
        f"ackley5_additive_narrow_{dimension}d",
    )
    for dimension in _HD_ACTIVE_DIMS
}

HIGH_DIMENSIONAL_SHIFTED_CORE: dict[int, tuple[str, ...]] = {
    dimension: (
        f"ackley_shifted_narrow_{dimension}d",
        f"rastrigin_mean_{dimension}d",
        f"hartmann6_sparse_{dimension}d",
        f"ackley5_additive_shifted_narrow_{dimension}d",
    )
    for dimension in _HD_ACTIVE_DIMS
}

HIGH_DIMENSIONAL_SYNTHETIC_PANEL_V4: dict[int, tuple[str, ...]] = {
    dimension: (
        f"rastrigin_mean_shifted_{dimension}d",
        f"griewank_mean_shifted_{dimension}d",
        f"rosenbrock_mean_{dimension}d",
        f"ellipsoid_mean_shifted_{dimension}d",
    )
    for dimension in _HD_ACTIVE_DIMS
}

HIGH_DIMENSIONAL_GENERALIZATION_PANEL_V5: dict[int, tuple[str, ...]] = {
    dimension: (
        f"sphere_mean_shifted_{dimension}d",
        f"bent_cigar_shifted_{dimension}d",
        f"rastrigin_mean_rotated_shifted_{dimension}d",
        f"levy_mean_shifted_{dimension}d",
        f"styblinski_tang_mean_shifted_{dimension}d",
        f"schaffers_f7_shifted_{dimension}d",
    )
    for dimension in _HD_ACTIVE_DIMS
}

SCALABLE_DIMENSION_ALPHA_PANELS: dict[str, tuple[str, ...]] = {
    "rastrigin": tuple(
        f"rastrigin_mean_rotated_shifted_{dimension}d"
        for dimension in (2, 5, 10, 20, 30)
    ),
    "rosenbrock": tuple(
        f"rosenbrock_mean_{dimension}d"
        for dimension in (2, 5, 10, 20, 30)
    ),
    "ackley": tuple(
        f"ackley_shifted_narrow_{dimension}d"
        for dimension in (2, 5, 10, 20, 30)
    ),
}

FIGURE_ORDER = tuple(BENCHMARKS)


def get_benchmark(name: str) -> Benchmark:
    return BENCHMARKS[name]
