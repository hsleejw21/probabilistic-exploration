"""Exact Matérn GP with known raw-scale observation noise.

The draft used scikit-learn but did not state target normalization.  Directly
combining ``normalize_y=True`` and ``alpha=lambda`` changes the raw-scale noise
variance.  This implementation standardizes targets explicitly and transforms
``lambda`` by the same scale, preserving the observation model in the paper.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning

from .config import ExperimentConfig


Array = np.ndarray


def pairwise_distance(x: Array, y: Array, lengthscale: float) -> Array:
    xs, ys = np.asarray(x) / lengthscale, np.asarray(y) / lengthscale
    x2 = np.sum(xs**2, axis=1)[:, None]
    y2 = np.sum(ys**2, axis=1)[None, :]
    return np.sqrt(np.maximum(x2 + y2 - 2.0 * xs @ ys.T, 0.0))


def matern_kernel(
    x: Array,
    y: Array,
    *,
    lengthscale: float,
    nu: float,
    signal_variance: float,
) -> Array:
    r = pairwise_distance(np.atleast_2d(x), np.atleast_2d(y), lengthscale)
    if math.isclose(nu, 1.5):
        z = math.sqrt(3.0) * r
        base = (1.0 + z) * np.exp(-z)
    elif math.isclose(nu, 2.5):
        z = math.sqrt(5.0) * r
        base = (1.0 + z + z**2 / 3.0) * np.exp(-z)
    else:
        raise ValueError("matern_nu must be 1.5 or 2.5")
    return signal_variance * base


def stable_cholesky(matrix: Array, base_jitter: float) -> tuple[Array, float]:
    jitter = max(float(base_jitter), 1e-12)
    identity = np.eye(matrix.shape[0])
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * identity), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    repaired = (vectors * np.maximum(values, jitter)) @ vectors.T
    return np.linalg.cholesky(repaired), jitter


class GaussianProcess:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.lengthscale = config.initial_lengthscale
        self.signal_variance = config.initial_signal_variance
        self.x = np.empty((0, 1))
        self.y = np.empty(0)
        self.y_mean = 0.0
        self.y_scale = 1.0
        self._l = np.empty((0, 0))
        self._alpha = np.empty(0)
        self.used_jitter = config.jitter

    @property
    def training_inputs(self) -> Array:
        """Unit-cube inputs used to fit the current posterior."""
        return self.x.copy()

    def kernel_input_coordinates(self, x: Array) -> Array:
        """Coordinates in which the fitted stationary kernel is evaluated."""
        return np.atleast_2d(np.asarray(x, dtype=float))

    def fitted_kernel_covariance(self, x: Array, y: Array) -> Array:
        """Fitted latent-kernel covariance in the GP's internal target scale."""
        return self._kernel(
            self.kernel_input_coordinates(x), self.kernel_input_coordinates(y)
        )

    @property
    def training_cholesky(self) -> Array:
        """Cholesky factor of K(X,X) plus fitted observation noise."""
        if not self._l.size:
            raise RuntimeError("fit must be called before training_cholesky")
        return self._l.copy()

    @property
    def posterior_output_scale(self) -> float:
        """Convert an internal-scale posterior residual to objective units."""
        return float(self.y_scale)

    @property
    def internal_noise_variance(self) -> float:
        return float(self.config.resolved_gp_nugget_variance() / self.y_scale**2)

    def _kernel(
        self,
        x: Array,
        y: Array,
        lengthscale: float | None = None,
        signal_variance: float | None = None,
    ) -> Array:
        return matern_kernel(
            x,
            y,
            lengthscale=self.lengthscale if lengthscale is None else lengthscale,
            nu=self.config.matern_nu,
            signal_variance=(
                self.signal_variance if signal_variance is None else signal_variance
            ),
        )

    def _negative_lml(self, theta: Array, y_norm: Array, scaled_noise: float) -> float:
        values = np.exp(theta)
        lengthscale = float(values[0])
        signal_variance = (
            float(values[1]) if self.config.learn_signal_variance
            else self.signal_variance
        )
        kernel = self._kernel(
            self.x,
            self.x,
            lengthscale=lengthscale,
            signal_variance=signal_variance,
        )
        try:
            l, _ = stable_cholesky(
                kernel + scaled_noise * np.eye(len(self.x)), self.config.jitter
            )
            alpha = solve_triangular(
                l.T, solve_triangular(l, y_norm, lower=True), lower=False
            )
        except (np.linalg.LinAlgError, ValueError):
            return 1e100
        value = 0.5 * y_norm @ alpha
        value += np.sum(np.log(np.diag(l)))
        value += 0.5 * len(self.x) * math.log(2.0 * math.pi)
        return float(value) if np.isfinite(value) else 1e100

    def fit(self, x: Array, y: Array, *, optimize_hyperparameters: bool) -> "GaussianProcess":
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float).ravel()
        if self.config.center_y:
            self.y_mean = float(np.mean(self.y))
            self.y_scale = 1.0
        elif self.config.normalize_y:
            self.y_mean = float(np.mean(self.y))
            observed_scale = float(np.std(self.y))
            self.y_scale = observed_scale if observed_scale > 1e-8 else 1.0
        else:
            self.y_mean = 0.0
            self.y_scale = 1.0
        y_norm = (self.y - self.y_mean) / self.y_scale
        scaled_noise = self.config.resolved_gp_nugget_variance() / self.y_scale**2

        if optimize_hyperparameters and len(self.x) >= 3:
            initial = [self.lengthscale]
            bounds = [tuple(np.log(self.config.lengthscale_bounds))]
            if self.config.learn_signal_variance:
                initial.append(self.signal_variance)
                bounds.append(tuple(np.log(self.config.signal_variance_bounds)))
            result = minimize(
                self._negative_lml,
                np.log(initial),
                args=(y_norm, scaled_noise),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 80},
            )
            if result.success and np.all(np.isfinite(result.x)):
                values = np.exp(result.x)
                self.lengthscale = float(values[0])
                if self.config.learn_signal_variance:
                    self.signal_variance = float(values[1])

        kernel = self._kernel(self.x, self.x)
        self._l, self.used_jitter = stable_cholesky(
            kernel + scaled_noise * np.eye(len(self.x)), self.config.jitter
        )
        self._alpha = solve_triangular(
            self._l.T,
            solve_triangular(self._l, y_norm, lower=True),
            lower=False,
        )
        return self

    def predict(self, x_star: Array, *, return_covariance: bool = False) -> tuple[Array, Array]:
        x_star = np.atleast_2d(np.asarray(x_star, dtype=float))
        cross = self._kernel(self.x, x_star)
        mean_norm = cross.T @ self._alpha
        triangular = solve_triangular(self._l, cross, lower=True)
        mean = self.y_mean + self.y_scale * mean_norm
        if return_covariance:
            covariance = self._kernel(x_star, x_star) - triangular.T @ triangular
            covariance = (covariance + covariance.T) * 0.5 * self.y_scale**2
            return mean, covariance
        variance = np.maximum(
            self.signal_variance - np.sum(triangular**2, axis=0), 0.0
        )
        return mean, self.y_scale * np.sqrt(variance)


class SklearnGaussianProcess:
    """Adapter exposing the canonical GP interface through scikit-learn.

    Acquisition optimization continues on the unit cube.  When
    ``gp_input_space='native'``, inputs are transformed internally before they
    are passed to scikit-learn, so benchmark-specific coordinate scales affect
    the Matérn geometry exactly as they would in a direct implementation.
    """

    def __init__(self, config: ExperimentConfig, lower: Array, upper: Array):
        self.config = config
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.lengthscale = config.initial_lengthscale
        self.signal_variance = config.initial_signal_variance
        self.used_jitter = 0.0
        self._gp: GaussianProcessRegressor | None = None
        self._x_unit = np.empty((0, len(self.lower)))
        self._y_mean = 0.0
        self._posterior_output_scale = 1.0

    @property
    def training_inputs(self) -> Array:
        """Unit-cube inputs used to fit the current posterior."""
        return self._x_unit.copy()

    def _transform(self, x: Array) -> Array:
        unit = np.atleast_2d(np.asarray(x, dtype=float))
        if self.config.gp_input_space == "unit":
            return unit
        if self.config.gp_input_space == "native":
            return self.lower + unit * (self.upper - self.lower)
        raise ValueError(f"unknown gp_input_space: {self.config.gp_input_space}")

    def kernel_input_coordinates(self, x: Array) -> Array:
        """Coordinates in which scikit-learn evaluates the fitted kernel."""
        return self._transform(x)

    def fitted_kernel_covariance(self, x: Array, y: Array) -> Array:
        """Fitted latent-kernel covariance in scikit-learn's target scale."""
        if self._gp is None:
            raise RuntimeError("fit must be called before fitted_kernel_covariance")
        return np.asarray(
            self._gp.kernel_(self._transform(x), self._transform(y)), dtype=float
        )

    @property
    def training_cholesky(self) -> Array:
        """Cholesky factor of K(X,X) plus scikit-learn's alpha."""
        if self._gp is None:
            raise RuntimeError("fit must be called before training_cholesky")
        return np.asarray(self._gp.L_, dtype=float).copy()

    @property
    def posterior_output_scale(self) -> float:
        """Convert an internal-scale posterior residual to objective units."""
        return float(self._posterior_output_scale)

    @property
    def internal_noise_variance(self) -> float:
        return float(self.config.resolved_gp_nugget_variance())

    def _kernel(self):
        matern = Matern(
            length_scale=self.lengthscale,
            length_scale_bounds=self.config.lengthscale_bounds,
            nu=self.config.matern_nu,
        )
        if not self.config.learn_signal_variance:
            return matern
        return ConstantKernel(
            self.signal_variance,
            self.config.signal_variance_bounds,
        ) * matern

    def fit(self, x: Array, y: Array, *, optimize_hyperparameters: bool) -> "SklearnGaussianProcess":
        self._x_unit = np.atleast_2d(np.asarray(x, dtype=float)).copy()
        targets = np.asarray(y, dtype=float).ravel()
        self._y_mean = float(np.mean(targets)) if self.config.center_y else 0.0
        centered_targets = targets - self._y_mean
        if self.config.normalize_y:
            target_scale = float(np.std(centered_targets))
            self._posterior_output_scale = target_scale if target_scale > 0.0 else 1.0
        else:
            self._posterior_output_scale = 1.0
        optimizer = "fmin_l_bfgs_b" if optimize_hyperparameters else None
        self._gp = GaussianProcessRegressor(
            kernel=self._kernel(),
            alpha=self.config.resolved_gp_nugget_variance(),
            optimizer=optimizer,
            n_restarts_optimizer=(
                self.config.sklearn_n_restarts_optimizer
                if optimize_hyperparameters else 0
            ),
            normalize_y=self.config.normalize_y,
            random_state=0,
        )
        # Boundary hits are preserved in the logged lengthscale trajectory.  A
        # warning on every BO iteration would otherwise overwhelm long-run logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self._gp.fit(self._transform(x), centered_targets)
        fitted = self._gp.kernel_
        if self.config.learn_signal_variance:
            self.signal_variance = float(fitted.k1.constant_value)
            fitted = fitted.k2
        self.lengthscale = float(np.asarray(fitted.length_scale).ravel()[0])
        return self

    def predict(self, x_star: Array, *, return_covariance: bool = False) -> tuple[Array, Array]:
        if self._gp is None:
            raise RuntimeError("fit must be called before predict")
        transformed = self._transform(x_star)
        if return_covariance:
            mean, covariance = self._gp.predict(transformed, return_cov=True)
            return np.asarray(mean) + self._y_mean, np.asarray(covariance)
        mean, std = self._gp.predict(transformed, return_std=True)
        return np.asarray(mean) + self._y_mean, np.asarray(std)


def build_gaussian_process(
    config: ExperimentConfig, lower: Array, upper: Array
) -> GaussianProcess | SklearnGaussianProcess:
    if config.gp_backend == "custom":
        if config.gp_input_space != "unit":
            raise ValueError("the custom GP currently requires gp_input_space='unit'")
        return GaussianProcess(config)
    if config.gp_backend == "sklearn":
        return SklearnGaussianProcess(config, lower, upper)
    raise ValueError(f"unknown gp_backend: {config.gp_backend}")
