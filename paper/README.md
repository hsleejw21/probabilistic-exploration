# Earlier consolidated research report

`main.tex` and `main.pdf` preserve the consolidated research-report snapshot
that preceded the current eight-benchmark AISTATS experiment design. They are
kept for provenance because the repository already released the earlier 30D,
YAHPO, Lunar, and Greedy Packing studies through this report.

The current paper-facing synthetic protocol and results are maintained in:

- `../configs/aistats2027.yaml`
- `../results/synthetic/eight_benchmark_alpha_sweep/`
- `../scripts/run_aistats2027_synthetic.sh`
- `../scripts/plot_aistats2027_synthetic.py`

Do not use this report's old overview table as the protocol for the current
paper snapshot. The result-directory protocol files are authoritative for each
released stage.

The historical report can still be compiled with Tectonic:

```bash
SOURCE_DATE_EPOCH=0 tectonic -X compile main.tex
```
