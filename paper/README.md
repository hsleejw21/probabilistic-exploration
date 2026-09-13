# Topic-based report

`main.tex` is the consolidated research report for **Probabilistic Exploration
for Efficient Bayesian Optimization**. It is organized by research question,
not by meeting date. All figures and tables required to compile it are included
locally.

Compile with Tectonic:

```bash
SOURCE_DATE_EPOCH=0 tectonic -X compile main.tex
```

Setting `SOURCE_DATE_EPOCH` keeps the checked-in PDF byte-stable across
unchanged rebuilds. The checked-in `main.pdf` was built from this source. The
audited 14D and 28D YAHPO stages and the final Greedy Packing comparisons
across acquisitions, Rastrigin dimensions, fresh seeds, transferred
benchmarks, YAHPO, and Lunar
Lander are included. The finite-grid robustness section reports separate
synthetic, YAHPO, and Lunar checks with larger candidate grids.
Figure and table sources follow the repository-wide naming convention in
[the visual-style guide](../docs/visual_style.md).
