#!/usr/bin/env python3
"""Audit completeness and pairing of a YAHPO HPO result directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--n-initial", type=int, default=None)
    args = parser.parse_args()

    metadata = json.loads((args.result_dir / "metadata.json").read_text())
    configs = metadata["gp_config_by_dimension"]
    assert configs, "missing GP configurations"
    for dimension, config in configs.items():
        assert config["initial_signal_variance"] == 1.0, dimension
        assert config["learn_signal_variance"] is False, dimension
        if args.n_initial is not None:
            assert config["n_initial"] == args.n_initial, dimension

    paths = sorted((args.result_dir / "trials").glob("*.csv"))
    assert len(paths) == args.expected_trials, (len(paths), args.expected_trials)
    initial = defaultdict(list)
    total_rows = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == args.budget, (path, len(rows))
        total_rows += len(rows)
        assert all(float(row["signal_variance"]) == 1.0 for row in rows), path
        first = rows[0]
        key = (first["task"], first["seed"])
        n_initial = int(configs[first["dimension"]]["n_initial"])
        signature = tuple(
            (row["query_x_unit"], row["loss"])
            for row in rows[:n_initial]
        )
        initial[key].append(signature)
    for key, signatures in initial.items():
        assert len(set(signatures)) == 1, f"unpaired initial design: {key}"

    message = (
        f"PASS {args.result_dir}: trials={len(paths)}, budget={args.budget}, "
        f"rows={total_rows}, n_initial={sorted({c['n_initial'] for c in configs.values()})}, "
        "signal_variance=1, paired_initial_design=true"
    )
    (args.result_dir / "audit.txt").write_text(message + "\n", encoding="utf-8")
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
