# Reproducibility

Install the core package with `pip install -e .`. Add `[lunar]`, `[yahpo]`, or
`[test]` for the corresponding optional dependencies.

Every runner writes one file per trial and safely resumes completed trials.
Use a separate output directory whenever the protocol changes. The public
results contain metadata, audits, and aggregate CSV files; large per-iteration
histories and trial files are not committed.

Examples:

```bash
python scripts/run_synthetic.py --help
python scripts/run_sklearn_hpo.py --help
python scripts/run_yahpo_hpo.py --help
```

YAHPO also requires benchmark data downloaded through `yahpo-gym`; local data
paths are machine-specific and are therefore not included in this repository.
