"""The W&B payload, tested without a network, an account or an API key.

The push itself is a thin wrapper around the client; what actually needs pinning is the shape
of what gets sent. In particular: a rate must never travel into a dashboard without its n and
its interval, because a bare number on a chart is exactly the artefact this tool exists to
prevent - and once it is in someone's dashboard it will be screenshotted into a slide deck.
"""

from __future__ import annotations

import pytest

from verdikt.compare import compare
from verdikt.integrations.wandb import build_payload, build_table_rows, parse_run_path
from verdikt.schema import Rollout, RunManifest


def arm(policy: str, n: int, k: int) -> list[Rollout]:
    return [Rollout(run_id="r", policy_id=policy, episode_idx=i, success=i < k)
            for i in range(n)]


@pytest.fixture
def result():
    return compare(arm("base", 200, 130) + arm("cand", 200, 48), "base")


class TestRunPath:
    def test_three_part_path(self):
        assert parse_run_path("acme/robots/abc123") == ("acme", "robots", "abc123")

    def test_two_part_path_defaults_entity(self):
        assert parse_run_path("robots/abc123") == (None, "robots", "abc123")

    def test_leading_slash_tolerated(self):
        assert parse_run_path("/robots/abc123") == (None, "robots", "abc123")

    def test_garbage_refused_with_a_useful_message(self):
        with pytest.raises(ValueError, match="entity/project/run_id"):
            parse_run_path("just-a-name")


class TestPayload:
    def test_verdict_is_both_name_and_code(self, result):
        p = build_payload(result, "base")
        assert p["verdikt/verdict"] in ("BETTER", "REGRESSION", "UNDERPOWERED",
                                        "NOT_COMPARABLE")
        assert p["verdikt/exit_code"] == int(result.verdict)

    def test_every_rate_travels_with_n_and_an_interval(self, result):
        p = build_payload(result, "base")
        for a in result.arms:
            key = f"verdikt/arm/{a.policy_id}"
            assert p[f"{key}/rate"] == pytest.approx(a.rate)
            assert p[f"{key}/n"] == a.n
            assert f"{key}/ci_low" in p and f"{key}/ci_high" in p, \
                "a rate must never reach a dashboard without its interval"

    def test_zero_arm_carries_its_one_sided_bound(self):
        res = compare(arm("base", 200, 100) + arm("zero", 200, 0), "base")
        p = build_payload(res, "base")
        assert p["verdikt/arm/zero/one_sided_bound"] > 0, \
            "a 0/n arm must not render as a hard zero on a chart"

    def test_ungraded_episodes_are_reported(self):
        rollouts = arm("cand", 5, 5) + [
            Rollout(run_id="r", policy_id="cand", episode_idx=100 + i, success=None,
                    progress=0.5) for i in range(20)
        ]
        p = build_payload(compare(rollouts), None)
        assert p["verdikt/arm/cand/ungraded"] == 20

    def test_confound_flag_and_samples_seen(self):
        mans = {
            "a": RunManifest(run_id="a", policy_id="a", batch_size=32, steps=50_000),
            "b": RunManifest(run_id="b", policy_id="b", batch_size=8, steps=20_000),
        }
        res = compare(arm("a", 50, 25) + arm("b", 50, 10), "a", manifests=mans)
        p = build_payload(res, "a")
        assert p["verdikt/confounded"] is True
        assert p["verdikt/arm/a/samples_seen"] == pytest.approx(1.6e6)

    def test_required_n_present_when_underpowered(self):
        res = compare(arm("base", 20, 7) + arm("cand", 20, 14), "base")
        p = build_payload(res, "base")
        assert p["verdikt/required_n"] > 20

    def test_posteriors_included(self, result):
        p = build_payload(result, "base", {"cand": 0.013})
        assert p["verdikt/posterior/P(cand>base)"] == pytest.approx(0.013)

    def test_payload_is_flat_and_json_safe(self, result):
        import json

        p = build_payload(result, "base")
        assert all(isinstance(v, (str, int, float, bool)) for v in p.values())
        json.dumps(p)  # must not raise


class TestTable:
    def test_columns_and_rows_align(self, result):
        cols, rows = build_table_rows(result)
        assert rows and all(len(r) == len(cols) for r in rows)

    def test_row_order_matches_arms(self, result):
        _cols, rows = build_table_rows(result)
        assert [r[0] for r in rows] == [a.policy_id for a in result.arms]
