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

    The contract has two halves and both must hold:
      1. two arms that are NOT significantly different share at least one letter;
      2. two arms that ARE significantly different share no letter.

    A greedy first-fit assignment satisfies (2) but silently violates (1) - with a~b, b~c and
    a!=c it hands out {a:'a', b:'a', c:'b'}, which tells the reader b and c differ when the
    test never said so. The correct construction is the set of maximal cliques of the
    "not significantly different" graph: every non-significant pair is an edge, and every edge
    lies inside some maximal clique, so (1) holds by construction; cliques contain no
    non-edges, so (2) holds too.
    """

    def differs(a: str, b: str) -> bool:
        return significant.get((a, b), significant.get((b, a), False))

    adjacency = {n: {m for m in names if m != n and not differs(n, m)} for n in names}
    cliques = sorted(_maximal_cliques(names, adjacency),
                     key=lambda c: (-len(c), sorted(c)))

    letters: dict[str, str] = {n: "" for n in names}
    for i, clique in enumerate(cliques):
        ch = chr(ord("a") + i) if i < 26 else f"({i + 1})"
        for n in clique:
            letters[n] += ch
    return letters


def _maximal_cliques(names: list[str], adjacency: dict[str, set[str]]) -> list[set[str]]:
    """Bron-Kerbosch with pivoting. Arm counts are small, so the exponential worst case
    is irrelevant here."""
    out: list[set[str]] = []

    def expand(r: set[str], p: set[str], x: set[str]) -> None:
        if not p and not x:
            out.append(set(r))
            return
        pivot = max(p | x, key=lambda v: len(adjacency[v]))
        for v in list(p - adjacency[pivot]):
            expand(r | {v}, p & adjacency[v], x & adjacency[v])
            p = p - {v}
            x = x | {v}

    expand(set(), set(names), set())
    return out
