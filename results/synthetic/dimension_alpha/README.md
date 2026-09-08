# Dimension and decay exponent

Rastrigin, Rosenbrock, and shifted Ackley were tested from 2D to 30D under
GP-UCB. Seven decay exponents were swept with five paired seeds per cell, 400
BO steps after two initial points, and fixed GP signal variance one.

Useful exponents cluster mostly between 0.5 and 1, but the best exponent is not
monotone in dimension and no dimension has the same winner on all three
functions. These results support problem dependence, not a universal mapping
from dimension to alpha.
