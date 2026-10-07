# Method

At BO iteration \(t\), exploration-aided BO either follows the base
acquisition function or makes an explicitly exploratory query. The current
experiments use

\[
p_t=\min\left\{1,\frac{\log(t+1)}{(t+1)^\alpha}\right\}.
\]

Smaller \(\alpha\) keeps exploration active longer, while larger \(\alpha\) returns to
the base acquisition sooner.

The implemented comparisons are:

- **Standard:** the base acquisition only.
- **Fixed exploration:** an exploratory query with constant probability \(p\).
- **Decaying exploration:** a time-decaying exploratory probability controlled by
  \(\alpha\).
- **MVR:** on exploration rounds, maximize posterior variance.

The default exploratory point is sampled uniformly. The optional finite-grid
Greedy Packing rule instead chooses

\[
x_t^{\mathrm{exp}} \in \arg\max_{x\in G_t}
\min_{z\in D_{t-1}}\lVert x-z\rVert_2,
\]

where \(D_{t-1}\) contains all previously evaluated inputs and \(G_t\) is a
nested scrambled-Sobol grid in the unit cube. The default implementation uses
\(|G_t|=256+\lceil16t\rceil\). It adapts Algorithm 1 of Pronzato and
Zhigljavsky's [Greedy Packing study](https://arxiv.org/abs/2112.10401). Their
Theorem 3.8 bounds the finite-candidate approximation through the candidate
grid's fill distance. The theorem concerns the packing design itself; using
the rule only on stochastic exploration rounds inside BO is an empirical hybrid and is
not claimed to inherit the theorem unchanged.

The shared GP protocol fixes signal variance to one, satisfying the usual
bounded-kernel convention \(k(x,x)\leq 1\), and learns lengthscales. All paired
comparisons share the initial design and random conditions within a seed.

For GP-TS, only the theory-aligned implementation is retained: logarithmic
\(\beta_t\) and a newly generated Sobol candidate set whose size grows over
time. Earlier fixed-beta GP-TS runs are intentionally excluded.

MVR is available for targeted comparisons, but it is not part of the main
six-benchmark figure. The main empirical comparison uses Uniform, while
the applications also evaluate Greedy Packing.
