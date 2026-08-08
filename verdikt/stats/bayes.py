"""Beta-Binomial posteriors.

Overlapping confidence intervals are systematically misread as "no difference" - they are not
the same statement. P(A > B) answers the question people are actually asking when they squint
at two error bars, and it degrades gracefully at small n instead of collapsing.
"""

from __future__ import annotations

import numpy as np


def posterior(k: int, n: int, prior_a: float = 1.0, prior_b: float = 1.0) -> tuple[float, float]:
    """Beta posterior parameters under a Beta(prior_a, prior_b) prior. Default is uniform."""
    return (prior_a + k, prior_b + n - k)


def prob_a_beats_b(ka: int, na: int, kb: int, nb: int, draws: int = 200_000,
                   seed: int = 0) -> float:
    """P(rate_A > rate_B) under independent uniform priors, by Monte-Carlo integration."""
    rng = np.random.default_rng(seed)
    aa, ab = posterior(ka, na)
    ba, bb = posterior(kb, nb)
    return float((rng.beta(aa, ab, draws) > rng.beta(ba, bb, draws)).mean())


def hdi_lift(ka: int, na: int, kb: int, nb: int, cred: float = 0.95, draws: int = 200_000,
             seed: int = 0) -> tuple[float, float]:
    """Highest-density interval on the absolute lift (rate_A - rate_B)."""
    rng = np.random.default_rng(seed)
    aa, ab = posterior(ka, na)
    ba, bb = posterior(kb, nb)
    diff = np.sort(rng.beta(aa, ab, draws) - rng.beta(ba, bb, draws))
    width = int(cred * draws)
    spans = diff[width:] - diff[: draws - width]
    i = int(np.argmin(spans))
    return (float(diff[i]), float(diff[i + width]))
