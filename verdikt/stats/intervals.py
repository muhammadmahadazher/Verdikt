"""Binomial confidence intervals.

The Wald interval is deliberately not implemented. It is the default in most hand-rolled
analysis scripts, it has coverage below nominal for small n, and it produces the degenerate
[0, 0] interval at k=0 - which is exactly the claim this tool exists to prevent.
"""

from __future__ import annotations

from scipy.stats import beta

METHODS = ("wilson", "jeffreys", "clopper-pearson")


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Good coverage at small n; the sensible default."""
    _validate(k, n)
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (_snap(max(0.0, centre - half)), 1.0 - _snap(1.0 - min(1.0, centre + half)))


def _snap(x: float, eps: float = 1e-12) -> float:
    """Collapse floating-point dust to exactly zero.

    At k=0 the Wilson lower bound is zero in exact arithmetic but lands on ~1.7e-18 in
    floating point. That is harmless in a calculation and absurd in a dashboard or a report,
    where it renders as a non-zero bound.
    """
    return 0.0 if abs(x) < eps else x


def jeffreys(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Jeffreys (Beta(1/2, 1/2) posterior) interval. Shortest average length."""
    _validate(k, n)
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 0.5, n - k + 0.5))
    return (lo, hi)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact interval. Guaranteed >= nominal coverage, conservative."""
    _validate(k, n)
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return (lo, hi)


def interval(k: int, n: int, method: str = "wilson", conf: float = 0.95) -> tuple[float, float]:
    """Dispatch by name. Unknown names fail loudly rather than falling back silently."""
    m = method.lower()
    if m == "wilson":
        return wilson(k, n, conf)
    if m == "jeffreys":
        return jeffreys(k, n, conf)
    if m in ("clopper-pearson", "clopper_pearson", "cp", "exact"):
        return clopper_pearson(k, n, conf)
    if m == "wald":
        raise ValueError(
            "the Wald interval is not implemented on purpose: it under-covers at small n and "
            "collapses to [0, 0] at k=0. use wilson, jeffreys or clopper-pearson."
        )
    raise ValueError(f"unknown interval method {method!r}; choose from {METHODS}")


def one_sided_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided upper bound - the honest way to report 0/n.

    At k=0 this reduces to the generalised rule of three: 1 - (1-conf)^(1/n).
    Reporting the *two-sided* upper bound as if it were one-sided is a common and
    inflationary error (0/20 -> 16.8% instead of the correct 13.9%).
    """
    _validate(k, n)
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def one_sided_lower(k: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided lower bound - the honest way to report n/n."""
    _validate(k, n)
    if k == 0:
        return 0.0
    return float(beta.ppf(1 - conf, k, n - k + 1))


def _validate(k: int, n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive; a rate over zero episodes is not a measurement")
    if not (0 <= k <= n):
        raise ValueError(f"successes {k} outside 0..{n}")
