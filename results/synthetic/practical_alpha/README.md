# Practical alpha selection

The tested procedure selects alpha on one seed block and evaluates it on fresh
seeds. Selection from five seeds fails to replicate at every dimension.
Increasing the selection block to twenty seeds helps only in 2D; the 20D choice
reverses significantly on untouched seeds.

A fixed alpha of 0.5 is more stable on average than the tested selectors, but
it is not uniformly better than Standard. The current result is therefore a
limitation: it motivates adaptive policies but does not yet justify a practical
selection rule.
