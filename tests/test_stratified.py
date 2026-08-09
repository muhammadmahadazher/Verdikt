"""Multi-task comparison, and the pooled number that lies about it.

The values asserted here are not this code's own output recorded after the fact. Every
Cochran-Mantel-Haenszel and Breslow-Day figure below was produced independently by
statsmodels 0.14.6 and agreed to 1e-6; `docs/crosscheck_stratified.py` reruns that comparison
on demand. Where statsmodels declines to compute a value - it divides by (OR - 1), so a common
odds ratio of exactly 1 gives NaN - the expected number is derived by hand in the test that
uses it, rather than being taken on trust from the implementation under test.
"""

from __future__ import annotations

import pytest

from verdikt.compare import compare, posterior_table
from verdikt.schema import Rollout, RunManifest, Verdict
from verdikt.stratified import (
    Stratum,
    analyse,
    breslow_day,
    cochran_mantel_haenszel,
    strata_from_rollouts,
)


def strata(*cells: tuple[int, int, int, int]) -> list[Stratum]:
    return [Stratum(f"task{i}", *c) for i, c in enumerate(cells)]


class TestAgainstStatsmodels:
    """Frozen cross-check values. See the module docstring for provenance."""

    def test_homogeneous_real_effect(self):
        s = strata((30, 50, 15, 50), (28, 50, 14, 50), (32, 50, 16, 50))
        stat, p, odds = cochran_mantel_haenszel(s)
        assert stat == pytest.approx(25.876063, abs=1e-5)
        assert p == pytest.approx(3.640557224571239e-07, rel=1e-6)
        assert odds == pytest.approx(3.5112, abs=1e-4)
        bd_stat, bd_p, contributing = breslow_day(s, odds)
        assert bd_stat == pytest.approx(0.05745, abs=1e-4)
        assert bd_p == pytest.approx(0.971682, abs=1e-5)
        assert contributing == 3

    def test_small_counts(self):
        s = strata((3, 10, 1, 10), (4, 12, 2, 11))
        stat, p, odds = cochran_mantel_haenszel(s)
        assert stat == pytest.approx(0.938321, abs=1e-5)
        assert p == pytest.approx(0.332710, abs=1e-5)
        assert odds == pytest.approx(2.7879, abs=1e-4)
        assert breslow_day(s, odds)[1] == pytest.approx(0.736248, abs=1e-5)

    def test_uneven_suite_of_five(self):
        s = strata((9, 20, 4, 20), (14, 30, 8, 31), (2, 8, 1, 9),
                   (25, 40, 18, 42), (6, 15, 3, 14))
        stat, p, odds = cochran_mantel_haenszel(s)
        assert stat == pytest.approx(9.357970, abs=1e-5)
        assert p == pytest.approx(0.002220, abs=1e-6)
        assert odds == pytest.approx(2.4968, abs=1e-4)
        assert breslow_day(s, odds)[1] == pytest.approx(0.994740, abs=1e-5)

    def test_near_null(self):
        s = strata((10, 20, 10, 20), (15, 30, 14, 30), (7, 14, 8, 15))
        stat, p, odds = cochran_mantel_haenszel(s)
        assert stat == pytest.approx(0.007066, abs=1e-5)
        assert p == pytest.approx(0.933011, abs=1e-5)
        assert odds == pytest.approx(1.0326, abs=1e-4)


class TestWhereTheReferenceDeclines:
    """statsmodels returns NaN when the common odds ratio is exactly 1. These are hand-derived.

    A NaN reference is worse than no reference: the obvious comparison `abs(mine - ref) > tol`
    is False for NaN, so an unchecked case silently reports agreement. The cross-check script
    lists these separately and they are pinned by hand here.
    """

    def test_breslow_day_at_unit_odds_ratio(self):
        """Wins one task by as much as it loses the other: OR = 1, but wildly heterogeneous.

        By hand, with a common OR of 1 the expected count is the independence expectation:
        each stratum has row1=50, col1=50, n=100, so E[a]=25 and
        Var = 1/(4/25) = 6.25. Observed 40 and 10 give (15^2)/6.25 = 36 each, total 72 on
        1 degree of freedom.
        """
        s = strata((40, 50, 10, 50), (10, 50, 40, 50))
        _stat, _p, odds = cochran_mantel_haenszel(s)
        assert odds == pytest.approx(1.0, abs=1e-9)
        bd_stat, bd_p, contributing = breslow_day(s, odds)
        assert bd_stat == pytest.approx(72.0, abs=1e-9)
        assert bd_p < 1e-15
        assert contributing == 2

    def test_tied_on_every_task_is_perfectly_homogeneous(self):
        """Equal rates everywhere: observed equals expected, so the statistic is exactly 0."""
        s = strata((5, 10, 45, 90), (18, 90, 2, 10))
        _stat, _p, odds = cochran_mantel_haenszel(s)
        assert odds == pytest.approx(1.0, abs=1e-9)
        assert breslow_day(s, odds)[0] == pytest.approx(0.0, abs=1e-9)


class TestSimpsonsParadox:
    def test_uneven_episode_counts_manufacture_a_pooled_gap(self):
        """The case this module exists for.

        Both policies are at exactly 50% on `reach` and exactly 20% on `insert`. They differ
        only in how many episodes each task received. The pooled rate shows a 24-point gap
        that corresponds to no difference in behaviour whatsoever.
        """
        result = analyse(strata((5, 10, 45, 90), (18, 90, 2, 10)))
        assert result.pooled_a_rate == pytest.approx(0.23)
        assert result.pooled_b_rate == pytest.approx(0.47)
        assert result.simpson_reversal
        assert result.common_odds_ratio == pytest.approx(1.0, abs=1e-9)
        assert result.cmh_p > 0.5, "the stratified test must not see an effect here"
        assert "SIMPSON" in result.notes[0]
        assert "level on every task" in result.notes[0]

    def test_strict_reversal_is_caught_and_named(self):
        """Ahead on both tasks, behind once pooled - the textbook sign flip."""
        result = analyse(strata((6, 10, 45, 90), (19, 90, 2, 10)))
        assert result.pooled_a_rate < result.pooled_b_rate
        assert all(s.a_rate > s.b_rate for s in result.strata)
        assert result.simpson_reversal
        assert result.common_odds_ratio > 1.0, "stratified analysis should favour policy A"
        assert "1 of 2 tasks" in result.notes[0] or "2 of 2 tasks" in result.notes[0]

    def test_an_honest_win_raises_nothing(self):
        result = analyse(strata((30, 50, 15, 50), (28, 50, 14, 50)))
        assert not result.simpson_reversal
        assert result.homogeneous
        assert result.notes == []
        assert result.cmh_p < 0.001

    def test_a_trivial_pooled_gap_is_not_called_a_paradox(self):
        """Sub-percent wobble is not worth an alarm, even when no task strictly supports it."""
        result = analyse(strata((10, 100, 10, 100), (10, 101, 10, 100)))
        assert abs(result.pooled_a_rate - result.pooled_b_rate) < 0.02
        assert not result.simpson_reversal


class TestHomogeneity:
    def test_opposite_effects_refuse_to_pool(self):
        result = analyse(strata((40, 50, 10, 50), (10, 50, 40, 50)))
        assert not result.homogeneous
        assert result.homogeneity_testable
        note = " ".join(result.notes)
        assert "not the same across tasks" in note
        assert "per-task table" in note

    def test_one_informative_task_reports_untestable_not_homogeneous(self):
        """A suite of uninformative tasks must not read as evidence of consistency."""
        result = analyse(strata((0, 10, 0, 10), (12, 20, 5, 20)))
        assert not result.homogeneity_testable
        assert "could not be tested" in " ".join(result.notes)
        assert "this is not evidence that it is" in " ".join(result.notes)

    def test_degrees_of_freedom_ignore_tasks_that_carry_no_information(self):
        """Padding a suite with unsolvable tasks must not make it look more homogeneous.

        The two informative tasks are identical in both runs; the four added tasks were solved
        by nobody. If those counted toward the degrees of freedom the p-value would drift
        upward purely because useless tasks were appended.
        """
        informative = ((40, 50, 10, 50), (10, 50, 40, 50))
        bare = analyse(strata(*informative))
        padded = analyse(strata(*informative, (0, 10, 0, 10), (0, 10, 0, 10),
                                (0, 10, 0, 10), (0, 10, 0, 10)))
        assert padded.homogeneity_p == pytest.approx(bare.homogeneity_p, rel=1e-12)


class TestStrataConstruction:
    def test_builds_one_table_per_task(self):
        rollouts = (
            [Rollout(run_id="r", policy_id="a", episode_idx=i, success=i < 3, task="lift")
             for i in range(5)]
            + [Rollout(run_id="r", policy_id="b", episode_idx=i, success=i < 1, task="lift")
               for i in range(5)]
            + [Rollout(run_id="r", policy_id="a", episode_idx=i, success=True, task="stack")
               for i in range(4)]
            + [Rollout(run_id="r", policy_id="b", episode_idx=i, success=False, task="stack")
               for i in range(4)]
        )
        built = strata_from_rollouts(rollouts, "a", "b")
        assert [s.task for s in built] == ["lift", "stack"]
        assert (built[0].a_success, built[0].a_n) == (3, 5)
        assert (built[1].b_success, built[1].b_n) == (0, 4)

    def test_ungraded_rollouts_are_excluded_from_the_denominator(self):
        """An episode with no success label is not a failure; it is not data.

        The schema will not accept a rollout with nothing measurable at all, so the ungraded
        case here is one scored on `progress` only - a partial-credit episode that no
        success threshold has been applied to.
        """
        rollouts = [
            Rollout(run_id="r", policy_id="a", episode_idx=0, success=True, task="lift"),
            Rollout(run_id="r", policy_id="a", episode_idx=1, progress=0.6, task="lift"),
            Rollout(run_id="r", policy_id="b", episode_idx=0, success=False, task="lift"),
        ]
        built = strata_from_rollouts(rollouts, "a", "b")
        assert built[0].a_n == 1

    def test_a_task_only_one_policy_attempted_is_dropped(self):
        """Comparing against an empty arm is not a comparison."""
        rollouts = (
            [Rollout(run_id="r", policy_id="a", episode_idx=0, success=True, task="solo")]
            + [Rollout(run_id="r", policy_id="a", episode_idx=0, success=True, task="both"),
               Rollout(run_id="r", policy_id="b", episode_idx=0, success=False, task="both")]
        )
        built = strata_from_rollouts(rollouts, "a", "b")
        assert [s.task for s in built if s.usable] == ["both"]
        assert analyse(built).strata[0].task == "both"

    def test_no_shared_task_is_refused(self):
        rollouts = [
            Rollout(run_id="r", policy_id="a", episode_idx=0, success=True, task="x"),
            Rollout(run_id="r", policy_id="b", episode_idx=0, success=True, task="y"),
        ]
        with pytest.raises(ValueError, match="no task has episodes for both"):
            analyse(strata_from_rollouts(rollouts, "a", "b"))


def suite_rollouts(cells: list[tuple[str, str, int, int]]) -> list[Rollout]:
    """(task, policy, successes, n) -> rollouts, successes placed deterministically."""
    return [
        Rollout(run_id=f"{policy}-{task}", policy_id=policy, task=task,
                episode_idx=i, success=i < successes)
        for task, policy, successes, n in cells
        for i in range(n)
    ]


# The fixture scenario: act_v2 was given more episodes on the easy task, so it "wins" the
# pooled rate by 29 points while being fractionally worse on one task and identical on the
# other. See tests/fixtures/multitask_suite/make_fixture.py.
DRIFTED_SUITE = [
    ("pick_bowl", "act_v1", 14, 20), ("pick_bowl", "act_v2", 55, 80),
    ("stack_blocks", "act_v1", 16, 80), ("stack_blocks", "act_v2", 4, 20),
]


class TestCompareIntegration:
    def test_a_task_mix_reversal_suppresses_the_comparison(self):
        """The pooled gap is 29 points and the tool must refuse to report it as a win."""
        result = compare(suite_rollouts(DRIFTED_SUITE), "act_v1")
        assert result.verdict is Verdict.NOT_COMPARABLE
        assert [c.kind for c in result.confounds] == ["TASK_MIX_CONFOUND"]
        assert result.pairs[0].suppressed_reason is not None
        assert "no task supports the pooled result" in result.confounds[0].message

    def test_the_check_is_not_behind_a_flag(self):
        """`compare` takes no argument to enable this. A guard you must remember is not one."""
        import inspect
        params = inspect.signature(compare).parameters
        assert not any("task" in p or "strat" in p for p in params), \
            f"stratification must not be opt-in, found: {sorted(params)}"

    def test_the_breakdown_is_attached_for_the_reader(self):
        result = compare(suite_rollouts(DRIFTED_SUITE), "act_v1")
        assert len(result.by_task) == 1
        summary = result.by_task[0]
        assert [r.task for r in summary.rows] == ["pick_bowl", "stack_blocks"]
        assert summary.simpson_reversal
        assert summary.cmh_p > 0.5

    def test_a_single_task_run_says_nothing_about_tasks(self):
        """Stratifying one stratum is the pooled comparison with extra words."""
        result = compare(suite_rollouts([
            ("lift", "a", 30, 100), ("lift", "b", 59, 100)]), "a")
        assert result.by_task == []
        assert result.verdict is Verdict.BETTER

    def test_an_even_suite_is_left_alone(self):
        """Same task mix, a real and consistent win: no suppression, breakdown still attached."""
        result = compare(suite_rollouts([
            ("pick_bowl", "a", 14, 50), ("pick_bowl", "b", 30, 50),
            ("stack_blocks", "a", 8, 50), ("stack_blocks", "b", 21, 50)]), "a")
        assert result.verdict is Verdict.BETTER
        assert [c.kind for c in result.confounds] == []
        assert result.by_task and not result.by_task[0].simpson_reversal

    def test_untasked_rollouts_do_not_trigger_anything(self):
        """`task` defaults to "unknown"; a harness that reports no task must not be punished."""
        rollouts = ([Rollout(run_id="r", policy_id="a", episode_idx=i, success=i < 30)
                     for i in range(100)]
                    + [Rollout(run_id="r", policy_id="b", episode_idx=i, success=i < 59)
                       for i in range(100)])
        result = compare(rollouts, "a")
        assert result.by_task == []
        assert result.verdict is Verdict.BETTER


class TestTaskCoverage:
    """Episodes spent where the other arm never went.

    Distinct from an uneven split of shared tasks: those episodes have nothing to pair
    against, so no stratified test can repair them. The arms' headline rates are averages
    over different problems.
    """

    def test_disjoint_task_sets_are_refused(self):
        result = compare(suite_rollouts([("x", "a", 5, 10), ("y", "b", 5, 10)]), "a")
        assert result.verdict is Verdict.NOT_COMPARABLE
        assert "share no task" in result.confounds[0].message

    def test_partial_overlap_is_refused_when_material(self):
        result = compare(suite_rollouts([
            ("x", "a", 5, 40), ("y", "a", 10, 60),
            ("y", "b", 12, 60), ("z", "b", 20, 40)]), "a")
        assert result.verdict is Verdict.NOT_COMPARABLE
        assert [c.kind for c in result.confounds] == ["TASK_MIX_CONFOUND"]
        assert "never attempted" in result.confounds[0].message

    def test_a_stray_episode_does_not_block_the_comparison(self):
        """One episode on an extra task is a crash, not a different experiment."""
        result = compare(suite_rollouts([
            ("y", "a", 10, 50), ("y", "b", 12, 49), ("z", "a", 1, 1)]), "a")
        assert result.confounds == []
        assert result.verdict is not Verdict.NOT_COMPARABLE

    def test_matching_coverage_is_left_alone(self):
        result = compare(suite_rollouts([
            ("y", "a", 10, 50), ("z", "a", 8, 50),
            ("y", "b", 20, 50), ("z", "b", 16, 50)]), "a")
        assert result.confounds == []
        assert result.verdict is Verdict.BETTER

    def test_the_breakdown_records_what_it_excluded(self):
        """The pooled row covers shared tasks only; the counts say how much was left out."""
        result = compare(suite_rollouts([
            ("y", "a", 10, 50), ("y", "b", 12, 50), ("z", "a", 5, 20)]), "a")
        summary = result.by_task[0]
        assert summary.a_unmatched == 20
        assert summary.b_unmatched == 0
        assert [r.task for r in summary.rows] == ["y"]


class TestPosteriorIsNotAWayAroundSuppression:
    """A posterior probability of superiority is a ranking.

    Printing "P(b > a) = 1.000" under a NOT COMPARABLE verdict hands the reader exactly the
    conclusion the verdict refused to draw, in the most quotable form on the page. Present in
    releases up to 0.3.1 for every confound type; the multi-task work made it impossible to
    ignore.
    """

    def test_a_confounded_arm_gets_no_posterior(self):
        result = compare(suite_rollouts(DRIFTED_SUITE), "act_v1")
        assert result.verdict is Verdict.NOT_COMPARABLE
        assert posterior_table(result.arms, "act_v1", result.pairs) == {}

    def test_suppression_is_targeted_not_blanket(self):
        """An unconfounded arm in the same run keeps its posterior."""
        rollouts = suite_rollouts(DRIFTED_SUITE) + suite_rollouts([
            ("pick_bowl", "act_v3", 14, 20), ("stack_blocks", "act_v3", 16, 80)])
        result = compare(rollouts, "act_v1")
        post = posterior_table(result.arms, "act_v1", result.pairs)
        assert "act_v2" not in post, "the confounded arm must be withheld"
        assert "act_v3" in post, "an arm on the same task mix is still comparable"

    def test_manifest_confounds_are_withheld_too(self):
        """Not specific to task mix - any suppressed pair."""
        rollouts = suite_rollouts([("lift", "a", 30, 100), ("lift", "b", 59, 100)])
        mans = {"a": RunManifest(run_id="ra", policy_id="a", dataset_revision="v1"),
                "b": RunManifest(run_id="rb", policy_id="b", dataset_revision="v2")}
        result = compare(rollouts, "a", manifests=mans)
        assert result.verdict is Verdict.NOT_COMPARABLE
        assert posterior_table(result.arms, "a", result.pairs) == {}

    def test_a_clean_comparison_still_reports_one(self):
        rollouts = suite_rollouts([("lift", "a", 30, 100), ("lift", "b", 59, 100)])
        post = posterior_table(*_args(compare(rollouts, "a"), "a"))
        assert post["b"] > 0.99


def _args(result, baseline):
    return result.arms, baseline, result.pairs
