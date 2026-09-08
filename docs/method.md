# Method

At BO iteration \(t\), Probabilistic Exploration (PE) either follows the base
acquisition function or makes an explicitly exploratory query. Under decay
uniform PE, the exploratory event is sampled with a probability that decreases
over time; its decay is controlled by \(\alpha\). Smaller \(\alpha\) keeps PE
active longer, while larger \(\alpha\) returns to the base acquisition sooner.

The main comparisons are:

- **Standard:** the base acquisition only.
- **Fixed PE:** an exploratory query with constant probability \(p\).
- **Decay PE:** a time-decaying exploratory probability controlled by
  \(\alpha\).

The shared GP protocol fixes signal variance to one, satisfying the usual
bounded-kernel convention \(k(x,x)\leq 1\), and learns lengthscales. All paired
comparisons share the initial design and random conditions within a seed.

For GP-TS, only the theory-aligned implementation is retained: logarithmic
\(\beta_t\) and a newly generated Sobol candidate set whose size grows over
time. Earlier fixed-beta GP-TS runs are intentionally excluded.
