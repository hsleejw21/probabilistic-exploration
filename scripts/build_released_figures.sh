#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"

cd "$repository_root"

"$python_bin" scripts/plot_aistats2027_synthetic.py
"$python_bin" scripts/plot_lowdim_runtime.py
"$python_bin" scripts/plot_application_results.py

echo "Rebuilt figures from the released aggregate results."
