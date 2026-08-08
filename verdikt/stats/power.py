"""Power and required-sample-size, computed through the test that will issue the verdict.

This is the design's single most load-bearing correctness decision. The normal approximation
(`statsmodels.NormalIndPower`) says 31 episodes per arm are enough to separate 35% from 70%
at 80% power. Run Fisher's exact test at n=31 and the realised power is 0.749. A planner that
plans with the approximation and decides with an exact test under-recommends rollouts by
15-30% - the precise opposite of its purpose.

We compute power by *exact enumeration* over the joint binomial outcome space rather than by
simulation: no Monte-Carlo error, and fast enough to be interactive once the outcome space is
pruned to the region carrying probability mass.
"""

from __future__ import annotations

from functools import lru_cache

from scipy.stats import binom

from .tests import unpaired_p

_MASS_TOL = 1e-10


@lru_cache(maxsize=4096)
def _support(n: int, p: float) -> tuple[tuple[int, float], ...]:
    """Outcomes carrying non-negligible probability. Prunes the tails that cannot matter.

    Vectorised: one scipy call for the whole row rather than n+1 scalar calls, which was the
    dominant cost of every power computation.
    """
    import numpy as np

    ks = np.arange(n + 1)
    pmf = binom.pmf(ks, n, p)
    keep = pmf > _MASS_TOL
    return tuple((int(k), float(w)) for k, w in zip(ks[keep], pmf[keep], strict=True))


@lru_cache(maxsize=100_000)
def power_exact(n_per_arm: int, p0: float, p1: float, alpha: float = 0.05,
                test: str = "fisher", alternative: str = "two-sided") -> float:
    """Exact probability of rejecting H0 when the true rates are p0 and p1.

    Enumerates the joint binomial outcome space, pruned to the region carrying probability
    mass. For each k0 the rejection region is found by scanning outward from k0 rather than
    testing every k1: p-values fall monotonically as the two counts separate, so once the
    boundary is crossed every remaining outcome on that side also rejects. That turns an
    O(n^2) table into roughly O(n * boundary distance) and keeps large-n planning interactive.
    """
    if n_per_arm < 1:
        return 0.0
    s0 = _support(n_per_arm, p0)
    s1 = dict(_support(n_per_arm, p1))
    if not s0 or not s1:
        return 0.0

    # The boundary sweep below assumes the p-value falls monotonically as the two counts
    # separate. That holds exactly for Fisher (verified against brute force across 126
    # configurations) but not quite for the unconditional exact tests, whose two-sided
    # p-values can wobble by ~1e-5. For those, enumerate honestly rather than approximate.
    if test.lower() != "fisher":
        return power_bruteforce(n_per_arm, p0, p1, alpha, test, alternative)

    lo1, hi1 = min(s1), max(s1)

    # Cumulative mass of arm 1, so a whole tail costs one lookup instead of a loop.
    cum = {}
    running = 0.0
    for k in range(lo1, hi1 + 1):
        running += s1.get(k, 0.0)
        cum[k] = running
    mass1 = running

    def low_tail(k):      # P(K1 <= k)
        return 0.0 if k < lo1 else cum[min(k, hi1)]

    def high_tail(k):     # P(K1 >= k)
        return mass1 if k <= lo1 else mass1 - cum[min(k - 1, hi1)]

    # The p-value peaks where the two counts agree and falls as they separate, so for a fixed
    # k0 the rejection region is the two outer tails. Both boundaries move monotonically as k0
    # increases, so a two-pointer sweep visits each boundary position once across the whole
    # table - O(n) tests instead of O(n * boundary distance).
    total = 0.0
    lower = lo1 - 1   # largest k1 <= k0 that rejects
    upper = lo1       # smallest k1 >= k0 that rejects
    for k0, w0 in sorted(s0):
        while (lower + 1 <= min(k0, hi1)
               and unpaired_p(lower + 1, n_per_arm, k0, n_per_arm, test, alternative) <= alpha):
            lower += 1
        if upper < k0:
            upper = k0
        while (upper <= hi1
               and unpaired_p(upper, n_per_arm, k0, n_per_arm, test, alternative) > alpha):
            upper += 1
        total += w0 * (low_tail(lower) + high_tail(upper))
    return total


def power_bruteforce(n_per_arm: int, p0: float, p1: float, alpha: float = 0.05,
                     test: str = "fisher", alternative: str = "two-sided") -> float:
    """Unoptimised reference implementation, kept solely so tests can prove `power_exact`
    agrees with it. Never called on a user path."""
    total = 0.0
    for k0, w0 in _support(n_per_arm, p0):
        for k1, w1 in _support(n_per_arm, p1):
            if unpaired_p(k1, n_per_arm, k0, n_per_arm, test, alternative) <= alpha:
                total += w0 * w1
    return total


def required_n(p0: float, p1: float, power: float = 0.80, alpha: float = 0.05,
               test: str = "fisher", alternative: str = "two-sided",
               n_max: int = 2000) -> int | None:
    """Smallest n per arm reaching the target power. None if n_max is not enough.

    Exact power is not perfectly monotone in n for conditional tests (the discreteness of the
    rejection region makes it saw-toothed), so after bisection we walk forward to the first n
    from which the target holds - and never report a lucky trough.
    """
    if not 0 < p0 < 1 or not 0 < p1 < 1:
        raise ValueError("rates must be strictly between 0 and 1")
    if p0 == p1:
        raise ValueError("p0 and p1 are equal; there is no effect to detect")

    lo, hi = 2, 8
    while hi <= n_max and power_exact(hi, p0, p1, alpha, test, alternative) < power:
        lo, hi = hi, hi * 2
    if hi > n_max:
        return None

    while lo < hi:
        mid = (lo + hi) // 2
        if power_exact(mid, p0, p1, alpha, test, alternative) >= power:
            hi = mid
        else:
            lo = mid + 1

    # walk forward past saw-tooth troughs: require the target to hold and stay held
    n = lo
    while n <= n_max:
        if all(power_exact(n + d, p0, p1, alpha, test, alternative) >= power for d in (0, 1, 2)):
            return n
        n += 1
    return None


def mde(n_per_arm: int, p0: float, power: float = 0.80, alpha: float = 0.05,
        test: str = "fisher", alternative: str = "two-sided",
        direction: str = "greater") -> float | None:
    """Minimum detectable effect: the smallest change from p0 this n can resolve.

    The inverse question, and the one that actually stops wasted runs: "I can afford 20
    episodes - what could I possibly learn?"
    """
    # Power rises monotonically with the size of the effect, so bisect on the effect instead
    # of scanning a grid: ~11 power evaluations instead of ~200, which is what makes this
    # usable inside `compare` at large n.
    span = (1 - p0) if direction == "greater" else p0
    sign = 1 if direction == "greater" else -1
    if span <= 0.01:
        return None

    def reached(delta: float) -> bool:
        p1 = min(0.999, max(0.001, p0 + sign * delta))
        return power_exact(n_per_arm, p0, p1, alpha, test, alternative) >= power

    if not reached(span - 1e-6):
        return None  # even the largest possible effect is not detectable at this n

    lo, hi = 0.0, span - 1e-6
    for _ in range(12):  # 12 halvings -> resolution better than 0.05 pp
        mid = (lo + hi) / 2
        if reached(mid):
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def normal_approx_n(p0: float, p1: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """The normal approximation - exposed only so the CLI can show what it would have said."""
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    es = proportion_effectsize(p1, p0)
    n = NormalIndPower().solve_power(effect_size=es, power=power, alpha=alpha, ratio=1.0)
    return int(-(-n // 1))
