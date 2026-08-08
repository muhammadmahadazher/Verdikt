"""Statistical correctness. These are the numbers the whole tool stands on."""

import pytest

from verdikt.stats import (
    clopper_pearson,
    interval,
    normal_approx_n,
    one_sided_lower,
    one_sided_upper,
    power_exact,
    required_n,
    unpaired_p,
    wilson,
)
from verdikt.stats.power import mde


class TestIntervals:
    def test_wilson_known_value(self):
        lo, hi = wilson(7, 20)
        assert lo == pytest.approx(0.1811, abs=1e-3)
        assert hi == pytest.approx(0.5671, abs=1e-3)

    def test_zero_successes_has_nonzero_upper(self):
        """0/n must never imply the rate is zero."""
        lo, hi = wilson(0, 20)
        assert lo == 0.0
        assert hi > 0.15

    def test_one_sided_upper_is_rule_of_three(self):
        """0/20 -> 13.9%, not the 16.8% two-sided bound that is often misquoted."""
        assert one_sided_upper(0, 20) == pytest.approx(0.1391, abs=1e-3)
        assert clopper_pearson(0, 20)[1] == pytest.approx(0.1684, abs=1e-3)

    def test_one_sided_lower_for_perfect_score(self):
        assert one_sided_lower(20, 20) == pytest.approx(0.8609, abs=1e-3)

    def test_wald_is_refused(self):
        """The Wald interval is not a fallback; asking for it is an error."""
        with pytest.raises(ValueError, match="not implemented on purpose"):
            interval(7, 20, method="wald")

    def test_zero_n_refused(self):
        with pytest.raises(ValueError, match="not a measurement"):
            wilson(0, 0)


class TestTests:
    def test_fisher_known_value(self):
        assert unpaired_p(7, 20, 0, 20, "fisher") == pytest.approx(0.00832, abs=1e-4)

    def test_barnard_disagrees_with_fisher_at_the_margin(self):
        """The reason the tool always names its test.

        At a Bonferroni-corrected alpha of 0.00833 these two land on opposite sides.
        """
        p_fisher = unpaired_p(7, 20, 0, 20, "fisher")
        p_barnard = unpaired_p(7, 20, 0, 20, "barnard")
        alpha_adj = 0.05 / 6
        assert p_fisher <= alpha_adj < p_barnard

    def test_identical_arms_give_p_one(self):
        assert unpaired_p(0, 20, 0, 20, "fisher") == pytest.approx(1.0)

    def test_published_headline_is_not_significant(self):
        """35% vs 70% at n=20 - the comparison this project was launched by correcting."""
        assert unpaired_p(7, 20, 14, 20, "fisher") == pytest.approx(0.0562, abs=1e-3)


class TestPower:
    def test_normal_approximation_overpromises(self):
        """The load-bearing claim: planning with the approximation under-recommends n.

        statsmodels says 31/arm for 35% vs 70% at 80% power; the exact test delivers less.
        """
        n_approx = normal_approx_n(0.35, 0.70, power=0.80, alpha=0.05)
        realised = power_exact(n_approx, 0.35, 0.70, alpha=0.05, test="fisher")
        assert realised < 0.80, "if this ever passes, the tool's central claim is wrong"

    def test_required_n_reaches_target(self):
        n = required_n(0.35, 0.70, power=0.80, alpha=0.05, test="fisher")
        assert n is not None
        assert power_exact(n, 0.35, 0.70, alpha=0.05, test="fisher") >= 0.80
        assert n > normal_approx_n(0.35, 0.70, 0.80, 0.05)

    def test_power_increases_with_n(self):
        lo = power_exact(20, 0.35, 0.70, test="fisher")
        hi = power_exact(80, 0.35, 0.70, test="fisher")
        assert hi > lo

    def test_mde_at_n20_is_large(self):
        """Why n=20 cannot settle most robotics comparisons."""
        d = mde(20, 0.35, power=0.80, alpha=0.05, test="fisher")
        assert d is not None and d > 0.30

    def test_equal_rates_refused(self):
        with pytest.raises(ValueError, match="no effect to detect"):
            required_n(0.5, 0.5)
