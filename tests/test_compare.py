"""Decision-engine behaviour: the refusals must be structural, not advisory."""

import pytest

from verdikt.compare import compare, find_confounds, summarise
from verdikt.schema import Rollout, RunManifest, Verdict


def arm(policy: str, n: int, k: int) -> list[Rollout]:
    return [
        Rollout(run_id=f"{policy}:t", policy_id=policy, episode_idx=i, success=i < k,
                label_source="simulator")
        for i in range(n)
    ]


class TestSummarise:
    def test_rate_carries_its_evidence(self):
        (a,) = summarise(arm("p", 20, 7))
        assert a.n == 20 and a.successes == 7
        assert a.ci_low < a.rate < a.ci_high

    def test_zero_arm_gets_one_sided_bound(self):
        (a,) = summarise(arm("p", 20, 0))
        assert a.one_sided_bound == pytest.approx(0.1391, abs=1e-3)


class TestVerdicts:
    def test_regression_detected(self):
        res = compare(arm("base", 40, 28) + arm("cand", 40, 8), baseline="base")
        assert res.verdict == Verdict.REGRESSION

    def test_underpowered_not_reported_as_no_difference(self):
        """35% vs 70% at n=20: a real gap the design cannot resolve."""
        res = compare(arm("base", 20, 7) + arm("cand", 20, 14), baseline="base")
        assert res.verdict == Verdict.UNDERPOWERED
        assert res.required_n is not None and res.required_n > 20

    def test_min_lower_bound_gates_on_the_bound_not_the_estimate(self):
        """18/20 = 90% point estimate, but the lower bound is below 90%."""
        res = compare(arm("cand", 20, 18), min_lower_bound=0.90)
        assert res.verdict == Verdict.REGRESSION
        assert "lower bound" in res.reason

    def test_exit_codes_match_schema(self):
        assert int(Verdict.BETTER) == 0
        assert int(Verdict.REGRESSION) == 1
        assert int(Verdict.UNDERPOWERED) == 2
        assert int(Verdict.NOT_COMPARABLE) == 3


class TestConfounds:
    def _manifests(self, steps_b: int, batch_b: int):
        return {
            "a": RunManifest(run_id="a", policy_id="a", batch_size=32, steps=50_000),
            "b": RunManifest(run_id="b", policy_id="b", batch_size=batch_b, steps=steps_b),
        }

    def test_ten_x_sample_gap_blocks_ranking(self):
        mans = self._manifests(20_000, 8)
        found = find_confounds(mans, "a", "b")
        assert any(c.kind == "COMPUTE_CONFOUND" for c in found)

    def test_equal_budgets_are_comparable(self):
        mans = self._manifests(50_000, 32)
        assert find_confounds(mans, "a", "b") == []

    def test_confounded_pair_is_suppressed_not_ranked(self):
        res = compare(arm("a", 20, 0) + arm("b", 20, 7), baseline="a",
                      manifests=self._manifests(20_000, 8))
        assert res.verdict == Verdict.NOT_COMPARABLE
        assert any(p.suppressed_reason for p in res.pairs)

    def test_dataset_revision_mismatch_is_a_confound(self):
        mans = {
            "a": RunManifest(run_id="a", policy_id="a", batch_size=32, steps=1000,
                             dataset_revision="aaa"),
            "b": RunManifest(run_id="b", policy_id="b", batch_size=32, steps=1000,
                             dataset_revision="bbb"),
        }
        assert any(c.kind == "DATA_CONFOUND" for c in find_confounds(mans, "a", "b"))


class TestRollout:
    def test_outcomeless_rollout_refused(self):
        with pytest.raises(ValueError, match="nothing to measure"):
            Rollout(run_id="r", policy_id="p", episode_idx=0)
