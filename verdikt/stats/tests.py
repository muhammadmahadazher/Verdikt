"""Hypothesis tests for comparing policy success rates, plus multiplicity control.

Every function returns the name of the test that produced the number, because the choice of
test changes verdicts at the margin (7/20 vs 0/20: Fisher p=0.0083, Barnard p=0.0100 - on
opposite sides of a Bonferroni threshold of 0.00833). A tool that hides which test ran is
inviting exactly the test-shopping it should prevent.
"""

from __future__ import annotations

from functools import lru_cache

from scipy.stats import barnard_exact, boschloo_exact, fisher_exact

TESTS = ("fisher", "barnard", "boschloo", "mcnemar")


@lru_cache(maxsize=200_000)
def unpaired_p(k1: int, n1: int, k2: int, n2: int, test: str = "fisher",
               alternative: str = "two-sided") -> float:
    """p-value for two independent binomial arms.

    fisher    - exact, conditions on both margins. Conservative but universally understood.
    barnard   - exact unconditional, maximises over the nuisance parameter. More powerful.
    boschloo  - exact unconditional, uniformly at least as powerful as Fisher.
    """
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    t = test.lower()
    if t == "fisher":
        return float(fisher_exact(table, alternative=alternative)[1])
    if t == "barnard":
        return float(barnard_exact(table, alternative=alternative).pvalue)
    if t == "boschloo":
        return float(boschloo_exact(table, alternative=alternative).pvalue)
    raise ValueError(f"unknown unpaired test {test!r}; choose from {TESTS}")


def paired_p(b: int, c: int, exact: bool = True) -> float:
    """McNemar on discordant pairs.

    b = episodes where A succeeded and B failed, c = the reverse. Concordant pairs carry no
    information about the difference and are correctly ignored - which is why pairing buys
    power: it removes scene difficulty from the comparison.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    if b + c == 0:
        return 1.0
    return float(mcnemar([[0, b], [c, 0]], exact=exact).pvalue)


def bonferroni(alpha: float, m: int) -> float:
    """Per-comparison alpha for a family of m tests."""
    return alpha / max(1, m)


def holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down. Uniformly more powerful than Bonferroni at the same family-wise rate."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values fail
    return reject


def compact_letters(names: list[str], significant: dict[tuple[str, str], bool]) -> dict[str, str]:
    """Compact letter display: arms sharing a letter are not distinguishable at this n.

    Reading a table of pairwise p-values is error-prone; letters make the grouping visible
    at a glance, and make "we cannot tell these apart" as legible as "these differ".
    """

    def differs(a: str, b: str) -> bool:
        return significant.get((a, b), significant.get((b, a), False))

    groups: list[list[str]] = []
    for name in names:
        placed = False
        for g in groups:
            if all(not differs(name, other) for other in g):
                g.append(name)
                placed = True
        if not placed:
            groups.append([name])

    letters: dict[str, str] = {n: "" for n in names}
    for i, g in enumerate(groups):
        ch = chr(ord("a") + i)
        for n in g:
            letters[n] += ch
    return letters
