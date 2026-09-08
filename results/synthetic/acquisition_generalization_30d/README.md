# 30D acquisition generalization

Three objectives were tested with GP-UCB, GP-TS, LogEI, and MES-Gumbel at 500
total evaluations. The held-out block contains ten paired seeds (5--14), uses
two shared initial Sobol observations, and fixes GP signal variance to one.

The retained GP-TS result uses \(\beta_t=\log(t+1)\) and a fresh Sobol
candidate set that grows with the BO iteration. The audit confirms 180 valid
GP-TS trials and 90,000 history rows. Older fixed-beta results are not present.

The clearest pattern is that slower decay often helps: \(\alpha=1/2\) is
strongest in seven of twelve objective--acquisition cells. GP-UCB and LogEI
show broad gains; MES is mixed on Bent Cigar; theory-aligned GP-TS improves
Rosenbrock but not Sparse Hartmann or Bent Cigar.
