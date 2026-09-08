# Lunar Lander 12D

Each BO evaluation tunes a 12-parameter heuristic controller and reports its
mean return on 50 terrains fixed within the run. Thirty paired seeds share the
same 24-point Sobol initial design. Runs use 400 total evaluations, a Matérn GP,
unit fixed signal variance, and learned lengthscales.

LogEI gives the clearest result: Standard mean return 215.8 rises to 261.8 for
decay alpha 0.5, and solved runs rise from 21/30 to 27/30. Decay alpha 1 uses
17.4 PE queries on average yet beats fixed p=0.2, which uses 73.3. On 200 new
terrains per saved controller, alpha 0.5 still improves mean return by 49.6 and
raises solved controllers from 20/30 to 27/30.

UCB and MES move in the same direction at alpha 0.5 but their intervals include
zero. Theory-aligned GP-TS is weak on Lunar and gives no clear PE improvement.
