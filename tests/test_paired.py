"""Paired comparison: the power win, and the assumption that makes it dangerous.

Pairing removes scene difficulty from a comparison, which is a large and free improvement in
power. It is also the easiest thing here to get silently wrong: episode 7 of one run is only
comparable to episode 7 of another if those were the same scene. When the harness records no
per-episode seed - and `lerobot-eval` does not - that has to be stated by the caller, never
assumed by the tool.
"""

from __future__ import annotations

import pytest

from verdikt.compare import PairingError, compare, pair_outcomes
from verdikt.schema import Rollout, Verdict


def arm(policy: str, outcomes: list[bool], seeds: list[int] | None = None) -> list[Rollout]:
    return [
        Rollout(run_id="r", policy_id=policy, episode_idx=i, success=ok,
                seed=(seeds[i] if seeds else None))
        for i, ok in enumerate(outcomes)
    ]


class TestPairingRefusals:
    def test_unseeded_pairing_is_refused_by_default(self):
        rollouts = arm("a", [True, False]) + arm("b", [False, True])
        with pytest.raises(PairingError, match="will not assume it for you"):
            pair_outcomes(rollouts, "a", "b")

    def test_unseeded_pairing_allowed_when_the_caller_states_it(self):
        rollouts = arm("a", [True, False]) + arm("b", [False, True])
        both, only_a, only_b, neither, key = pair_outcomes(
            rollouts, "a", "b", allow_index_pairing=True)
        assert (both, only_a, only_b, neither) == (0, 1, 1, 0)
        assert key == "episode index"

    def test_seeded_pairing_needs_no_assumption(self):
        rollouts = (arm("a", [True, False], seeds=[11, 22])
                    + arm("b", [False, True], seeds=[11, 22]))
        _both, only_a, only_b, _neither, key = pair_outcomes(rollouts, "a", "b")
        assert key == "seed" and (only_a, only_b) == (1, 1)

    def test_pairs_by_seed_not_by_order(self):
        """The arms record the same two scenes in opposite order; pairing must follow the
        seed, or the discordant counts come out backwards."""
        rollouts = (arm("a", [True, False], seeds=[11, 22])
                    + arm("b", [False, True], seeds=[22, 11]))
        both, only_a, only_b, neither, _key = pair_outcomes(rollouts, "a", "b")
        # seed 11: a=True b=True -> both ; seed 22: a=False b=False -> neither
        assert (both, only_a, only_b, neither) == (1, 0, 0, 1)

    def test_disjoint_scene_sets_are_refused(self):
        rollouts = (arm("a", [True] * 10, seeds=list(range(10)))
                    + arm("b", [True] * 10, seeds=list(range(100, 110))))
        with pytest.raises(PairingError, match="no episodes are shared"):
            pair_outcomes(rollouts, "a", "b")

    def test_mostly_disjoint_sets_are_refused(self):
        rollouts = (arm("a", [True] * 20, seeds=list(range(20)))
                    + arm("b", [True] * 20, seeds=list(range(18, 38))))
        with pytest.raises(PairingError, match="same set of scenes"):
            pair_outcomes(rollouts, "a", "b")


class TestPairedPower:
    """Why anyone would bother: the same data, resolved with far fewer episodes."""

    @staticmethod
    def _correlated(n: int, hard_fraction: float = 0.6):
        """Both policies fail the hard scenes; the candidate wins on some easy ones.

        Marginal rates are close, so an unpaired test struggles - but every discordant pair
        points the same way, which is exactly the structure pairing exploits.
        """
        base, cand, seeds = [], [], []
        hard = int(n * hard_fraction)
        for i in range(n):
            seeds.append(i)
            if i < hard:                       # nobody solves these
                base.append(False)
                cand.append(False)
            elif i % 3 == 0:                   # candidate wins these
                base.append(False)
                cand.append(True)
            else:                              # both solve these
                base.append(True)
                cand.append(True)
        return arm("base", base, seeds) + arm("cand", cand, seeds)

    def test_paired_detects_what_unpaired_cannot(self):
        rollouts = self._correlated(40)
        unpaired = compare(rollouts, "base", correction="none")
        paired = compare(rollouts, "base", correction="none", paired=True)
        p_unpaired = unpaired.pairs[0].p_value
        p_paired = paired.pairs[0].p_value
        assert p_paired < p_unpaired, "pairing should be strictly more powerful here"
        assert p_paired <= 0.05 < p_unpaired

    def test_paired_verdict_names_the_test_and_the_key(self):
        result = compare(self._correlated(40), "base", correction="none", paired=True)
        assert "mcnemar" in result.pairs[0].test
        assert "seed" in result.pairs[0].test
        assert "discordant" in result.pairs[0].test

    def test_no_discordant_pairs_means_no_evidence(self):
        """Identical outcomes on every scene carry no information about a difference."""
        rollouts = (arm("base", [True, False, True], seeds=[1, 2, 3])
                    + arm("cand", [True, False, True], seeds=[1, 2, 3]))
        result = compare(rollouts, "base", correction="none", paired=True)
        assert result.pairs[0].p_value == pytest.approx(1.0)
        assert result.verdict in (Verdict.UNDERPOWERED, Verdict.BETTER)

    def test_paired_still_reports_unpaired_arm_summaries(self):
        """The table is still per-arm; only the test between them changes."""
        result = compare(self._correlated(40), "base", correction="none", paired=True)
        assert {a.policy_id for a in result.arms} == {"base", "cand"}
        assert all(a.n == 40 for a in result.arms)
