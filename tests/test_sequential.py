"""Calibration gate for the sequential test.

This file is not a normal test module - it is the release condition for `verdikt watch`. A
sequential test whose realised false-positive rate exceeds alpha is strictly worse than no
sequential test at all: instead of declining to answer, it manufactures confident wrong
answers, and it does so most often on exactly the marginal comparisons people care about.

If anything here goes red, `watch` must not ship.
"""

from __future__ import annotations

import numpy as np
import pytest

from verdikt.sequential import SequentialState, false_positive_rate, replay_savings, run, update


class TestMartingaleProperty:
    """Under the null the capital process must be a martingale: E[X_t] stays at 1."""

    def test_capital_starts_at_one(self):
        assert SequentialState().capital == pytest.approx(1.0)

    def test_expected_capital_is_preserved_under_the_null(self):
        rng = np.random.default_rng(7)
        finals = []
        for _ in range(4000):
            state = SequentialState(alpha=0.05)
            for _ in range(40):
                update(state, float(rng.random() < 0.4), float(rng.random() < 0.4))
                if state.stopped_at:
                    break
            finals.append(state.capital)
        # the mean of a martingale is invariant; sampling noise is wide because the
        # distribution is heavy-tailed, so this is a sanity band, not a tight bound
        assert 0.75 < float(np.mean(finals)) < 1.35

    def test_capital_stays_positive(self):
        state = SequentialState()
        for _ in range(200):  # adversarial: always maximally against the bet
            update(state, 1.0, 0.0)
        assert state.capital > 0

    def test_outcomes_outside_unit_interval_are_refused(self):
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            update(SequentialState(), 0.5, 1.7)


class TestFalsePositiveCalibration:
    """The release gate. Empirical FPR must not exceed alpha at ANY stopping time."""

    @pytest.mark.parametrize("p", [0.05, 0.25, 0.5, 0.75])
    def test_fpr_within_alpha_at_5_percent(self, p):
        fpr = false_positive_rate(p, n=200, alpha=0.05, trials=6000, seed=11)
        assert fpr <= 0.05, f"sequential test is anti-conservative at p={p}: FPR={fpr}"

    def test_fpr_within_alpha_at_1_percent(self):
        fpr = false_positive_rate(0.5, n=200, alpha=0.01, trials=6000, seed=3)
        assert fpr <= 0.01

    def test_fpr_holds_for_a_long_run(self):
        """Ville's bound is over all time, so a longer run must not erode it."""
        fpr = false_positive_rate(0.5, n=600, alpha=0.05, trials=4000, seed=5)
        assert fpr <= 0.05


class TestPower:
    """Being valid is not enough; it has to actually stop on real differences."""

    def test_detects_a_large_difference_quickly(self):
        rng = np.random.default_rng(0)
        a = (rng.random(300) < 0.10).astype(float)
        b = (rng.random(300) < 0.70).astype(float)
        state = run(a, b, alpha=0.05)
        assert state.stopped_at is not None
        assert state.stopped_at < 60

    def test_declines_to_reject_when_there_is_no_difference(self):
        rng = np.random.default_rng(1)
        a = (rng.random(400) < 0.30).astype(float)
        b = (rng.random(400) < 0.30).astype(float)
        assert run(a, b, alpha=0.05).stopped_at is None


class TestReplay:
    def test_savings_are_measured_not_assumed(self):
        rng = np.random.default_rng(2)
        a = (rng.random(200) < 0.05).astype(float)
        b = (rng.random(200) < 0.65).astype(float)
        res = replay_savings(a, b, alpha=0.05, trials=200, seed=0)
        assert res["stop_rate"] == pytest.approx(1.0)
        assert res["median_stop"] < 60
        assert 0.0 < res["median_saving"] < 1.0

    def test_identical_arms_report_no_saving(self):
        rng = np.random.default_rng(4)
        a = (rng.random(150) < 0.4).astype(float)
        b = (rng.random(150) < 0.4).astype(float)
        res = replay_savings(a, b, alpha=0.05, trials=150, seed=0)
        assert res["stop_rate"] < 0.10

    def test_empty_input_refused(self):
        with pytest.raises(ValueError, match="no episodes"):
            replay_savings([], [], trials=2)


class TestOnRealRollouts:
    """The corpus is 800 real PushT rollouts, so these numbers are not synthetic."""

    def test_real_difference_stops_early(self, ):
        import json
        from pathlib import Path

        fixtures = Path(__file__).parent / "fixtures" / "pusht_n200"
        if not (fixtures / "act.json").exists():
            pytest.skip("n=200 corpus not present")

        def successes(name):
            m = json.loads((fixtures / f"{name}.json").read_text())["per_task"][0]["metrics"]
            return [1.0 if s else 0.0 for s in m["successes"]]

        res = replay_savings(successes("act"), successes("upstream"), trials=200, seed=0)
        assert res["stop_rate"] == pytest.approx(1.0)
        assert res["median_saving"] > 0.75, "should need well under a quarter of the episodes"

    def test_indistinguishable_pair_does_not_stop(self):
        """act (1%) vs smolvla (0%) is not significant at n=200; the sequential test must
        not claim otherwise on binary success."""
        import json
        from pathlib import Path

        fixtures = Path(__file__).parent / "fixtures" / "pusht_n200"
        if not (fixtures / "act.json").exists():
            pytest.skip("n=200 corpus not present")

        def successes(name):
            m = json.loads((fixtures / f"{name}.json").read_text())["per_task"][0]["metrics"]
            return [1.0 if s else 0.0 for s in m["successes"]]

        res = replay_savings(successes("act"), successes("smolvla"), trials=200, seed=0)
        assert res["stop_rate"] < 0.05
