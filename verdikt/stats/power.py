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

from scipy.stats import binom

from .tests import unpaired_p

_MASS_TOL = 1e-10


def _support(n: int, p: float) -> list[tuple[int, float]]:
    """Outcomes carrying non-negligible probability. Prunes the tails that cannot matter."""
    ks = range(n + 1)
    out = [(k, float(binom.pmf(k, n, p))) for k in ks]
    return [(k, w) for k, w in out if w > _MASS_TOL]


def power_exact(n_per_arm: int, p0: float, p1: float, alpha: float = 0.05,
                test: str = "fisher", alternative: str = "two-sided") -> float:
    """Exact probability of rejecting H0 when the true rates are p0 and p1."""
    if n_per_arm < 1:
        return 0.0
    s0 = _support(n_per_arm, p0)
    s1 = _support(n_per_arm, p1)
    total = 0.0
    for k0, w0 in s0:
        for k1, w1 in s1:
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
    step = 0.005
    if direction == "greater":
        grid = [p0 + i * step for i in range(1, int((1 - p0) / step))]
    else:
        grid = [p0 - i * step for i in range(1, int(p0 / step))]
    for p1 in grid:
        if power_exact(n_per_arm, p0, p1, alpha, test, alternative) >= power:
            return abs(p1 - p0)
    return None


def normal_approx_n(p0: float, p1: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """The normal approximation - exposed only so the CLI can show what it would have said."""
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    es = proportion_effectsize(p1, p0)
    n = NormalIndPower().solve_power(effect_size=es, power=power, alpha=alpha, ratio=1.0)
    return int(-(-n // 1))
