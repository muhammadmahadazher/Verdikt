"""Adapter contract, pinned against committed golden fixtures.

Adapters are the first thing to rot when an upstream format changes. These fixtures are the
early-warning system: if a future LeRobot release changes the shape, the parser must fail
loudly here rather than silently mis-map a field in someone's analysis.
"""

from pathlib import Path

import pytest

from verdikt.compare import summarise
from verdikt.ingest import autodetect, available, get
from verdikt.ingest.base import AdapterError

FIXTURES = Path(__file__).parent / "fixtures" / "adapters" / "lerobot_eval"
BETTER = FIXTURES / "eval_better.json"
WORSE = FIXTURES / "eval_worse.json"


class TestRegistry:
    def test_expected_adapters_registered(self):
        assert {"lerobot", "csv", "vla-on-a-budget"} <= set(available())

    def test_unknown_adapter_named_clearly(self):
        with pytest.raises(AdapterError, match="unknown adapter"):
            get("does-not-exist")


class TestLeRobotAdapter:
    def test_autodetects_by_content(self):
        assert autodetect(BETTER).name in ("lerobot", "vla-on-a-budget")

    def test_parses_every_episode(self):
        rollouts = get("lerobot").parse(BETTER, policy_id="better")
        assert len(rollouts) == 20
        assert all(r.policy_id == "better" for r in rollouts)
        assert all(r.label_source == "simulator" for r in rollouts)

    def test_success_count_matches_the_source(self):
        rollouts = get("lerobot").parse(BETTER, policy_id="better")
        (arm,) = summarise(rollouts)
        assert arm.successes == 10 and arm.n == 20  # 50.0% as recorded in the fixture

    def test_partial_credit_only_when_normalised(self):
        """max_reward is used as progress only when it is genuinely in [0, 1]."""
        rollouts = get("lerobot").parse(WORSE, policy_id="worse")
        progresses = [r.progress for r in rollouts if r.progress is not None]
        assert len(progresses) == 20
        assert all(0.0 <= p <= 1.0 for p in progresses)

    def test_seeds_absent_is_explicit_not_invented(self):
        """This format carries no per-episode seed; the adapter must not fabricate one."""
        rollouts = get("lerobot").parse(BETTER, policy_id="better")
        assert all(r.seed is None for r in rollouts)

    def test_garbage_fails_loudly(self, tmp_path):
        bad = tmp_path / "not_eval.json"
        bad.write_text('{"something": "else"}', encoding="utf-8")
        with pytest.raises(AdapterError, match="per_task"):
            get("lerobot").parse(bad)

    def test_end_to_end_two_arms(self):
        rollouts = (get("lerobot").parse(BETTER, policy_id="better")
                    + get("lerobot").parse(WORSE, policy_id="worse"))
        arms = {a.policy_id: a for a in summarise(rollouts)}
        assert arms["better"].rate == pytest.approx(0.50)
        assert arms["worse"].rate == pytest.approx(0.15)
        assert arms["better"].ci_low < arms["better"].rate < arms["better"].ci_high
