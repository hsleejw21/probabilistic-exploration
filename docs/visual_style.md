# Figure and table style

All public figures and tables use one visual language so they can be moved
between the paper, slides, and result folders without reinterpretation.

## File names

Use lowercase `snake_case` without dates or figure numbers:

- figures: `fig_<study>_<scope>_<content>.pdf` and the matching `.png`
- tables: `tab_<study>_<scope>_<content>.tex`

The PDF and PNG versions of a figure must have exactly the same stem. Figure
numbers belong in LaTeX captions and labels, not in file names, because numbers
change when the report is reorganised.

## Visual semantics

- DejaVu Sans is the figure font. Sizes, spines, grids, and export settings are
  defined in `probabilistic_exploration.plot_style`.
- Standard is dashed grey. Decay policies use the shared light-to-dark blue
  palette, and fixed `p=0.2` is orange. A policy never changes colour between
  studies.
- Every difference is written as **PE improvement over Standard**, so positive
  values favour PE.
- In point-and-interval figures, a hollow point means that the 95% interval
  contains zero; a filled point means that it excludes zero.
- In heatmaps, blue favours PE, red favours Standard, and a black outline means
  that the 95% interval excludes zero.
- Acquisition order is GP-UCB, GP-TS, LogEI, MES-Gumbel, then KG when present.
  Policy order is Standard, increasing decay exponent, then fixed controls.
- Endpoint comparison panels use horizontal intervals and a vertical zero
  line. Trajectory panels use a shared legend above the panels and uncertainty
  bands of the same definition within a figure.

## Tables

Tables use the same positive-is-better sign convention. Pale green/red marks
the direction of the mean; deeper colour and bold type mean that the paired
95% interval excludes zero. Dataset, acquisition, and policy ordering must
match the corresponding figure.
