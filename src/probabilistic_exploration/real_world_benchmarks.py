"""Bayesmark-derived real-data hyperparameter optimization tasks.

The datasets, model families, 80/20 split, five-fold validation objective, and
parameter ranges follow Uber's Bayesmark.  The implementation is local so it
works with the project's current scikit-learn version and can enforce paired
model randomness across PE policies.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import expit, logit
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


Array = np.ndarray


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str
    space: str
    lower: float
    upper: float

    def decode(self, unit_value: float) -> float | int:
        value = float(np.clip(unit_value, 0.0, 1.0))
        if self.space == "linear":
            decoded = self.lower + value * (self.upper - self.lower)
        elif self.space == "log":
            decoded = math.exp(
                math.log(self.lower)
                + value * (math.log(self.upper) - math.log(self.lower))
            )
        elif self.space == "logit":
            decoded = float(
                expit(logit(self.lower) + value * (logit(self.upper) - logit(self.lower)))
            )
        else:
            raise ValueError(f"unknown parameter space: {self.space}")
        if self.kind == "int":
            return int(np.clip(np.rint(decoded), self.lower, self.upper))
        if self.kind == "real":
            return float(np.clip(decoded, self.lower, self.upper))
        raise ValueError(f"unknown parameter kind: {self.kind}")


@dataclass(frozen=True)
class BayesmarkTask:
    name: str
    dataset: str
    model: str
    parameters: tuple[Parameter, ...]

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def decode(self, unit_x: Array) -> dict[str, float | int]:
        values = np.asarray(unit_x, dtype=float).ravel()
        if values.shape != (self.dim,):
            raise ValueError(f"{self.name} expects {self.dim} parameters")
        return {
            parameter.name: parameter.decode(value)
            for parameter, value in zip(self.parameters, values)
        }


SVM_PARAMETERS = (
    Parameter("C", "real", "log", 1.0, 1e3),
    Parameter("gamma", "real", "log", 1e-4, 1e-3),
    Parameter("tol", "real", "log", 1e-5, 1e-1),
)

RF_PARAMETERS = (
    Parameter("max_depth", "int", "linear", 1, 15),
    Parameter("max_features", "real", "logit", 0.01, 0.99),
    Parameter("min_samples_split", "real", "logit", 0.01, 0.99),
    Parameter("min_samples_leaf", "real", "logit", 0.01, 0.49),
    Parameter("min_weight_fraction_leaf", "real", "logit", 0.01, 0.49),
    Parameter("min_impurity_decrease", "real", "linear", 0.0, 0.5),
)

MLP_PARAMETERS = (
    Parameter("hidden_layer_sizes", "int", "linear", 50, 200),
    Parameter("alpha", "real", "log", 1e-5, 1e1),
    Parameter("batch_size", "int", "linear", 10, 250),
    Parameter("learning_rate_init", "real", "log", 1e-5, 1e-1),
    Parameter("tol", "real", "log", 1e-5, 1e-1),
    Parameter("validation_fraction", "real", "logit", 0.1, 0.9),
    Parameter("beta_1", "real", "logit", 0.5, 0.99),
    Parameter("beta_2", "real", "logit", 0.9, 1.0 - 1e-6),
    Parameter("epsilon", "real", "log", 1e-9, 1e-6),
)


BAYESMARK_TASKS: dict[str, BayesmarkTask] = {
    f"{dataset}_{suffix}": BayesmarkTask(
        f"{dataset}_{suffix}", dataset, model, parameters
    )
    for dataset in ("breast", "digits")
    for suffix, model, parameters in (
        ("svm", "SVM", SVM_PARAMETERS),
        ("rf", "RF", RF_PARAMETERS),
        ("mlp_adam", "MLP-adam", MLP_PARAMETERS),
    )
}


@lru_cache(maxsize=None)
def task_split(dataset: str, seed: int) -> tuple[Array, Array, Array, Array]:
    if dataset == "breast":
        x, y = load_breast_cancer(return_X_y=True)
    elif dataset == "digits":
        x, y = load_digits(return_X_y=True)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return train_test_split(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=int),
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )


def build_classifier(task: BayesmarkTask, params: dict[str, object], seed: int):
    if task.model == "SVM":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
            **params,
        )
    if task.model == "RF":
        return RandomForestClassifier(
            n_estimators=10,
            max_leaf_nodes=None,
            random_state=seed,
            n_jobs=1,
            **params,
        )
    if task.model == "MLP-adam":
        return MLPClassifier(
            solver="adam",
            early_stopping=True,
            random_state=seed,
            max_iter=200,
            **params,
        )
    raise ValueError(f"unknown model: {task.model}")


def validation_loss(
    task: BayesmarkTask,
    unit_x: Array,
    seed: int,
) -> tuple[float, dict[str, float | int]]:
    params = task.decode(unit_x)
    train_x, _, train_y, _ = task_split(task.dataset, seed)
    estimator = build_classifier(task, params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            estimator,
            train_x,
            train_y,
            scoring="accuracy",
            cv=5,
            n_jobs=1,
            error_score="raise",
        )
    loss = 1.0 - float(np.mean(scores))
    if not np.isfinite(loss):
        raise RuntimeError(f"non-finite validation loss for {task.name}")
    return loss, params


def heldout_test_loss(
    task: BayesmarkTask,
    unit_x: Array,
    seed: int,
) -> float:
    params = task.decode(unit_x)
    train_x, test_x, train_y, test_y = task_split(task.dataset, seed)
    estimator = build_classifier(task, params, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(train_x, train_y)
    return 1.0 - float(estimator.score(test_x, test_y))
