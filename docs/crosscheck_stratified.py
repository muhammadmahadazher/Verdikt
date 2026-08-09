"""Cross-check the stratified tests against statsmodels, then freeze the agreed numbers.

Verdikt does not depend on statsmodels and neither does its test suite. This script exists so
that the Cochran-Mantel-Haenszel and Breslow-Day implementations in `verdikt/stratified.py`
are checked once, here, against a mature independent implementation - and the values both
agree on are then hard-coded as literals in `tests/test_stratified.py`. A test that only
compares my code to my code proves nothing; a literal that two implementations independently
produced is a real anchor.

Run it manually after touching the maths:

    python docs/crosscheck_stratified.py

It prints a table and exits non-zero if any cell disagrees beyond tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verdikt.stratified import Stratum, breslow_day, cochran_mantel_haenszel  # noqa: E402

# (name, [(a_success, a_n, b_success, b_n), ...])
CASES: list[tuple[str, list[tuple[int, int, int, int]]]] = [
    ("textbook reversal", [(5, 10, 45, 90), (18, 90, 2, 10)]),
    ("real effect, homogeneous", [(30, 50, 15, 50), (28, 50, 14, 50), (32, 50, 16, 50)]),
    ("heterogeneous: wins one, loses one", [(40, 50, 10, 50), (10, 50, 40, 50)]),
    ("small counts", [(3, 10, 1, 10), (4, 12, 2, 11)]),
    ("uneven suite of five", [(9, 20, 4, 20), (14, 30, 8, 31), (2, 8, 1, 9),
                              (25, 40, 18, 42), (6, 15, 3, 14)]),
    ("near null", [(10, 20, 10, 20), (15, 30, 14, 30), (7, 14, 8, 15)]),
    ("one empty-information task", [(0, 10, 0, 10), (12, 20, 5, 20)]),
]

TOL_STAT = 1e-6
TOL_P = 1e-8


def statsmodels_reference(cells):
    """statsmodels wants a 2 x 2 x K array of [[succ_A, fail_A], [succ_B, fail_B]]."""
    tables = np.array([
        [[a_s, a_n - a_s], [b_s, b_n - b_s]] for a_s, a_n, b_s, b_n in cells
    ]).transpose(1, 2, 0)
    st = StratifiedTable(tables)
    return (
        float(st.test_null_odds(correction=True).statistic),
        float(st.test_null_odds(correction=True).pvalue),
        float(st.oddsratio_pooled),
        float(st.test_equal_odds().statistic),
        float(st.test_equal_odds().pvalue),
    )


def main() -> int:
    failures: list[str] = []
    undefined: list[str] = []
    print(f"{'case':38s} {'CMH chi2':>10s} {'CMH p':>10s} {'OR':>8s} "
          f"{'BD chi2':>9s} {'BD p':>9s}")
    print("-" * 92)
    for name, cells in CASES:
        strata = [Stratum(f"t{i}", *c) for i, c in enumerate(cells)]
        stat, p, odds = cochran_mantel_haenszel(strata)
        bd_stat, bd_p, _contributing = breslow_day(strata, odds)
        r_stat, r_p, r_or, r_bd, r_bdp = statsmodels_reference(cells)

        print(f"{name:38s} {stat:10.6f} {p:10.6f} {odds:8.4f} {bd_stat:9.5f} {bd_p:9.6f}")
        print(f"{'  statsmodels':38s} {r_stat:10.6f} {r_p:10.6f} {r_or:8.4f} "
              f"{r_bd:9.5f} {r_bdp:9.6f}")

        for label, mine, ref, tol in [
            ("CMH statistic", stat, r_stat, TOL_STAT),
            ("CMH p", p, r_p, TOL_P),
            ("common OR", odds, r_or, TOL_STAT),
            ("Breslow-Day statistic", bd_stat, r_bd, 1e-4),
            ("Breslow-Day p", bd_p, r_bdp, 1e-6),
        ]:
            # NaN must never be scored as agreement: abs(x - nan) > tol is False, so a
            # comparison written the obvious way passes silently exactly where the
            # reference declined to answer. Those cases are surfaced, not skipped.
            mine_ok, ref_ok = np.isfinite(mine), np.isfinite(ref)
            if not ref_ok:
                undefined.append(f"{name}: statsmodels returns {ref} for {label} "
                                 f"(verdikt says {mine!r})")
                continue
            if not mine_ok:
                failures.append(f"{name}: {label} is {mine!r} but statsmodels gives {ref!r}")
                continue
            if abs(mine - ref) > tol:
                failures.append(f"{name}: {label} {mine!r} vs statsmodels {ref!r}")
        print()

    if undefined:
        print("REFERENCE UNDEFINED - checked by hand instead, see the notes in "
              "tests/test_stratified.py:")
        for u in undefined:
            print("  -", u)
        print()

    if failures:
        print("DISAGREEMENT:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"all {len(CASES)} cases agree with statsmodels on every value it defines "
          f"({len(undefined)} value(s) it declined to compute).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
