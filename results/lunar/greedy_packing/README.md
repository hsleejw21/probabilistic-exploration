# Greedy Packing on Lunar Lander

This study replaces Uniform exploration by finite-grid Greedy Packing in the
12D Lunar Lander controller-tuning problem. It uses LogEI, `alpha=1/2`, 24
scrambled-Sobol initial points, and 400 evaluation steps. A BO evaluation is
the mean return over 50 fixed terrains for that run. Signal variance is fixed
at one.

Final controllers are compared on a shared bank of 200 terrains not used by
BO. This produces paired held-out comparisons for mean return, landing rate,
and bottom-10% return.

- `default/`: Standard, Uniform PE, and Greedy PE on 30 paired seeds with
  `|G_t| = 256 + ceil(16 t)`.
- `grid2x/`: a 15-seed confirmation with `|G_t| = 512 + ceil(32 t)`.
- `grid4x/`: a 15-seed confirmation with `|G_t| = 1024 + ceil(64 t)`.

Uniform and default Greedy PE both improve held-out mean return over Standard.
The larger grid improves Greedy over its default implementation, but its
advantage over Uniform remains inconclusive. Raw BO histories and controller
files are omitted; the public files retain only audited held-out summaries and
the shared episode identifiers.
