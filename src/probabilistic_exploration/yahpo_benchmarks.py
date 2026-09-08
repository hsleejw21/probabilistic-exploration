"""YAHPO Gym real-data surrogate benchmark adapters.

The first application is ``rbv2_xgboost``: XGBoost hyperparameter tuning on
OpenML data.  YAHPO evaluates a learned surrogate of the original HPO runs,
which makes repeated BO comparisons practical while preserving the mixed and
hierarchical 14-dimensional search space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class YahpoParameter:
    name: str
    kind: str
    lower: float | None = None
    upper: float | None = None
    choices: tuple[str, ...] = ()
    log_scale: bool = False

    def decode(self, unit_value: float) -> float | int | str:
        value = float(np.clip(unit_value, 0.0, 1.0))
        if self.kind == "categorical":
            index = min(int(value * len(self.choices)), len(self.choices) - 1)
            return self.choices[index]
        if self.lower is None or self.upper is None:
            raise ValueError(f"numeric parameter {self.name} needs bounds")
        if self.log_scale:
            decoded = math.exp(
                math.log(self.lower)
                + value * (math.log(self.upper) - math.log(self.lower))
            )
        else:
            decoded = self.lower + value * (self.upper - self.lower)
        if self.kind == "integer":
            return int(np.clip(np.rint(decoded), self.lower, self.upper))
        if self.kind == "real":
            return float(np.clip(decoded, self.lower, self.upper))
        raise ValueError(f"unknown parameter kind: {self.kind}")


XGBOOST_PARAMETERS = (
    YahpoParameter("alpha", "real", 9.118819655545162e-4, 1096.6331584284585, log_scale=True),
    YahpoParameter("booster", "categorical", choices=("gblinear", "gbtree", "dart")),
    YahpoParameter("lambda", "real", 9.118819655545162e-4, 1096.6331584284585, log_scale=True),
    YahpoParameter("nrounds", "integer", 7, 2981, log_scale=True),
    YahpoParameter(
        "num.impute.selected.cpo",
        "categorical",
        choices=("impute.mean", "impute.median", "impute.hist"),
    ),
    YahpoParameter("subsample", "real", 0.1, 1.0),
    YahpoParameter("colsample_bylevel", "real", 0.01, 1.0),
    YahpoParameter("colsample_bytree", "real", 0.01, 1.0),
    YahpoParameter("eta", "real", 9.118819655545162e-4, 1.0, log_scale=True),
    YahpoParameter("gamma", "real", 4.5399929762484854e-5, 7.38905609893065, log_scale=True),
    YahpoParameter("max_depth", "integer", 1, 15),
    YahpoParameter("min_child_weight", "real", math.e, math.exp(5.0), log_scale=True),
    YahpoParameter("rate_drop", "real", 0.0, 1.0),
    YahpoParameter("skip_drop", "real", 0.0, 1.0),
)


@dataclass(frozen=True)
class YahpoTask:
    name: str
    scenario: str
    instance: str
    dataset_name: str
    target: str
    expected_dim: int
    parameters: tuple[YahpoParameter, ...] = ()
    constants: tuple[tuple[str, object], ...] = ()

    @property
    def dim(self) -> int:
        return self.expected_dim

    def decode(self, unit_x: Array) -> dict[str, object]:
        values = np.asarray(unit_x, dtype=float).ravel()
        if values.shape != (self.dim,):
            raise ValueError(f"{self.name} expects {self.dim} coordinates")
        if not self.parameters:
            return _decode_configspace(self, values)
        parameters = {
            definition.name: definition.decode(value)
            for definition, value in zip(self.parameters, values)
        }

        # YAHPO's ConfigSpace rejects values for inactive conditional fields.
        booster = parameters["booster"]
        tree_parameters = {
            "colsample_bylevel", "colsample_bytree", "eta", "gamma",
            "max_depth", "min_child_weight",
        }
        dart_parameters = {"rate_drop", "skip_drop"}
        if booster == "gblinear":
            for name in tree_parameters | dart_parameters:
                parameters.pop(name)
        elif booster == "gbtree":
            for name in dart_parameters:
                parameters.pop(name)

        parameters.update(dict(self.constants))
        return parameters


YAHPO_TASKS: dict[str, YahpoTask] = {
    "rbv2_xgboost_31": YahpoTask(
        name="rbv2_xgboost_31",
        scenario="rbv2_xgboost",
        instance="31",
        dataset_name="credit-g",
        target="logloss",
        expected_dim=14,
        parameters=XGBOOST_PARAMETERS,
        constants=(("task_id", "31"), ("repl", 1), ("trainsize", 1.0)),
    ),
    "rbv2_xgboost_40975": YahpoTask(
        name="rbv2_xgboost_40975",
        scenario="rbv2_xgboost",
        instance="40975",
        dataset_name="car",
        target="logloss",
        expected_dim=14,
        parameters=XGBOOST_PARAMETERS,
        constants=(("task_id", "40975"), ("repl", 1), ("trainsize", 1.0)),
    ),
    "rbv2_xgboost_1464": YahpoTask(
        name="rbv2_xgboost_1464",
        scenario="rbv2_xgboost",
        instance="1464",
        dataset_name="blood-transfusion-service-center",
        target="logloss",
        expected_dim=14,
        parameters=XGBOOST_PARAMETERS,
        constants=(("task_id", "1464"), ("repl", 1), ("trainsize", 1.0)),
    ),
    "iaml_super_40981": YahpoTask(
        name="iaml_super_40981",
        scenario="iaml_super",
        instance="40981",
        dataset_name="OpenML-40981",
        target="logloss",
        expected_dim=28,
        constants=(("task_id", "40981"), ("trainsize", 1.0)),
    ),
    "iaml_super_41146": YahpoTask(
        name="iaml_super_41146",
        scenario="iaml_super",
        instance="41146",
        dataset_name="sylvine",
        target="logloss",
        expected_dim=28,
        constants=(("task_id", "41146"), ("trainsize", 1.0)),
    ),
    "rbv2_super_31": YahpoTask(
        name="rbv2_super_31",
        scenario="rbv2_super",
        instance="31",
        dataset_name="credit-g",
        target="logloss",
        expected_dim=38,
        constants=(("task_id", "31"), ("repl", 1), ("trainsize", 1.0)),
    ),
    "rbv2_super_40975": YahpoTask(
        name="rbv2_super_40975",
        scenario="rbv2_super",
        instance="40975",
        dataset_name="car",
        target="logloss",
        expected_dim=38,
        constants=(("task_id", "40975"), ("repl", 1), ("trainsize", 1.0)),
    ),
}


@lru_cache(maxsize=None)
def _benchmark(scenario: str):
    # Import lazily so the main paper environment does not need yahpo-gym.
    from yahpo_gym import BenchmarkSet

    return BenchmarkSet(scenario, multithread=False)


def _decode_configspace(task: YahpoTask, values: Array) -> dict[str, object]:
    """Decode a fixed unit cube into YAHPO's active hierarchical parameters."""
    from ConfigSpace.util import deactivate_inactive_hyperparameters

    config_space = _benchmark(task.scenario).get_opt_space(
        drop_fidelity_params=False
    )
    constant_names = {name for name, _ in task.constants}
    hyperparameters = [
        hp for hp in config_space.get_hyperparameters()
        if hp.name not in constant_names
    ]
    if len(hyperparameters) != task.expected_dim:
        raise RuntimeError(
            f"{task.scenario} expected {task.expected_dim} tunable parameters, "
            f"found {len(hyperparameters)}"
        )

    decoded: dict[str, object] = {}
    for hp, unit_value in zip(hyperparameters, values):
        unit_value = float(np.clip(unit_value, 0.0, 1.0))
        if hasattr(hp, "choices"):
            choices = tuple(hp.choices)
            index = min(int(unit_value * len(choices)), len(choices) - 1)
            decoded[hp.name] = choices[index]
            continue
        lower, upper = float(hp.lower), float(hp.upper)
        if bool(getattr(hp, "log", False)):
            value = math.exp(
                math.log(lower) + unit_value * (math.log(upper) - math.log(lower))
            )
        else:
            value = lower + unit_value * (upper - lower)
        if "Integer" in type(hp).__name__:
            decoded[hp.name] = int(np.clip(np.rint(value), lower, upper))
        else:
            decoded[hp.name] = float(np.clip(value, lower, upper))

    decoded.update(dict(task.constants))
    active = deactivate_inactive_hyperparameters(
        configuration=decoded,
        configuration_space=config_space,
    )
    return active.get_dictionary()


def objective_loss(task: YahpoTask, unit_x: Array) -> tuple[float, dict[str, object]]:
    parameters = task.decode(unit_x)
    result = _benchmark(task.scenario).objective_function(parameters)[0]
    loss = float(result[task.target])
    if not np.isfinite(loss):
        raise RuntimeError(f"non-finite {task.target} for {task.name}")
    return loss, parameters


def empirical_target_range(task: YahpoTask) -> tuple[float, float]:
    """Return YAHPO's reference min/max from a large random sample."""
    stats = _benchmark(task.scenario).target_stats
    selected = stats[
        (stats["instance"].astype(str) == task.instance)
        & (stats["metric"] == task.target)
    ]
    values = {
        str(row.statistic): float(row.value)
        for row in selected.itertuples(index=False)
    }
    return values["min"], values["max"]
