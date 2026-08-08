"""Regression tests for bugs found by the adversarial audit.

Each test encodes a specific way the tool once produced a wrong or misleading verdict. They
are grouped by the failure they prevent rather than by module, because that is how they will
be read when one of them goes red.
"""

from __future__ import annotations

import pytest

from verdikt.compare import compare
from verdikt.schema import Rollout, RunManifest, Verdict
from verdikt.stats import compact_letters


def arm(policy: str, n: int, k: int, *, graded: bool = True) -> list[Rollout]:
    return [
        Rollout(run_id="r", policy_id=policy, episode_idx=i,
                success=(i < k) if graded else None,
                progress=None if graded else 0.9)
        for i in range(n)
    ]


def manifests(**budgets: tuple[int, int]) -> dict[str, RunManifest]:
    return {name: RunManifest(run_id=name, policy_id=name, batch_size=b, steps=s)
            for name, (b, s) in budgets.items()}


class TestUngradedRolloutsAreNeverHidden:
    """A broken grader once turned 5/100 into '100% success, BETTER'."""

    def test_ungraded_are_counted_and_exposed(self):
        res = compare(arm("cand", 5, 5) + arm("cand", 95, 0, graded=False))
        (a,) = res.arms
        assert a.n == 5, "graded n must not silently include ungraded episodes"
        assert a.n_ungraded == 95, "ungraded episodes must be reported, not dropped"

    def test_fully_graded_arm_reports_no_ungraded(self):
        res = compare(arm("cand", 20, 7))
        assert res.arms[0].n_ungraded == 0


class TestCompactLetterDisplayIsValid:
    """Greedy first-fit produced letters implying a difference the test never found."""

    def test_transitive_chain_shares_letters(self):
        letters = compact_letters(
            ["a", "b", "c"],
            {("a", "b"): False, ("b", "c"): False, ("a", "c"): True},
        )
        assert set(letters["a"]) & set(letters["b"]), "a~b must share a letter"
        assert set(letters["b"]) & set(letters["c"]), "b~c must share a letter"
        assert not (set(letters["a"]) & set(letters["c"])), "a!=c must not share a letter"

    def test_all_equivalent_arms_share_one_letter(self):
        names = ["p", "q", "r"]
        letters = compact_letters(names, {})
        assert len({letters[n] for n in names}) == 1

    def test_all_different_arms_share_nothing(self):
        sig = {("p", "q"): True, ("q", "r"): True, ("p", "r"): True}
        letters = compact_letters(["p", "q", "r"], sig)
        assert not (set(letters["p"]) & set(letters["q"]))
        assert not (set(letters["q"]) & set(letters["r"]))


class TestMultiplicityCorrection:
    """Suppressed pairs consume no alpha; counting them once masked a real regression."""

    def test_family_size_excludes_suppressed_pairs(self):
        res = compare(
            arm("base", 30, 20) + arm("a", 30, 10) + arm("b", 30, 10),
            "base",
            manifests=manifests(base=(32, 50_000), a=(32, 50_000), b=(8, 20_000)),
        )
        live = [p for p in res.pairs if p.suppressed_reason is None]
        assert live, "at least one comparison should survive"
        assert live[0].alpha_adjusted == pytest.approx(0.05 / len(live))

    def test_holm_records_the_threshold_it_actually_used(self):
        res = compare(arm("base", 30, 20) + arm("x", 30, 5) + arm("y", 30, 19),
                      "base", correction="holm")
        for p in res.pairs:
            if p.suppressed_reason:
                continue
            # a reader must never find p <= alpha_adjusted next to significant=False
            assert (p.p_value <= p.alpha_adjusted) == p.significant


class TestNonInferiority:
    """A margin at or above the baseline rate made the test unfailable."""

    def test_vacuous_margin_is_refused(self):
        res = compare(arm("base", 50, 40) + arm("cand", 50, 0), "base",
                      noninferiority_margin=0.85)
        assert res.verdict == Verdict.NOT_COMPARABLE
        assert "vacuous" in res.reason

    def test_genuine_noninferiority_still_passes(self):
        res = compare(arm("base", 200, 100) + arm("cand", 200, 98), "base",
                      noninferiority_margin=0.10)
        assert res.verdict == Verdict.BETTER

    def test_clearly_inferior_candidate_fails(self):
        res = compare(arm("base", 200, 160) + arm("cand", 200, 60), "base",
                      noninferiority_margin=0.10)
        assert res.verdict == Verdict.REGRESSION


class TestConfoundsCannotBeBypassed:
    """--min-lower-bound once laundered a suppressed comparison into a pass."""

    def test_confound_outranks_a_lower_bound_gate(self):
        res = compare(arm("a", 30, 25) + arm("b", 30, 25), "a",
                      manifests=manifests(a=(32, 50_000), b=(8, 20_000)),
                      min_lower_bound=0.10)
        assert res.verdict == Verdict.NOT_COMPARABLE


class TestLowerBoundGate:
    """The baseline was held to the candidate's bar and reported as a regression."""

    def test_baseline_is_not_gated(self):
        res = compare(arm("base", 20, 5) + arm("cand", 20, 19), "base", min_lower_bound=0.60)
        assert res.verdict == Verdict.BETTER

    def test_failing_candidate_still_caught(self):
        res = compare(arm("base", 20, 19) + arm("cand", 20, 5), "base", min_lower_bound=0.60)
        assert res.verdict == Verdict.REGRESSION

    def test_confidence_level_is_not_hardcoded(self):
        res = compare(arm("cand", 20, 5), min_lower_bound=0.60, conf=0.99)
        assert "99%" in res.reason


class TestNullResultsRequirePower:
    """'Not significant' was reported as 'no regression' by designs that could detect nothing."""

    def test_tiny_sample_is_underpowered_not_better(self):
        res = compare(arm("base", 1, 0) + arm("cand", 1, 0), "base")
        assert res.verdict == Verdict.UNDERPOWERED

    def test_equal_rates_at_small_n_are_underpowered(self):
        res = compare(arm("base", 10, 5) + arm("cand", 10, 5), "base")
        assert res.verdict == Verdict.UNDERPOWERED

    def test_large_matched_sample_gives_an_informative_null(self):
        res = compare(arm("base", 500, 250) + arm("cand", 500, 248), "base")
        assert res.verdict == Verdict.BETTER
        assert "informative" in res.reason

    def test_zero_rate_clamping_is_disclosed(self):
        res = compare(arm("base", 30, 0) + arm("cand", 30, 1), "base")
        if res.verdict == Verdict.UNDERPOWERED:
            assert "floor of 0.5%" in res.reason


class TestSingleArm:
    def test_single_arm_without_a_gate_is_not_called_better(self):
        res = compare(arm("only", 20, 10))
        assert res.verdict == Verdict.UNDERPOWERED
        assert "nothing to compare" in res.reason
