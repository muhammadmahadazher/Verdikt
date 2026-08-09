"""Multi-task comparison: pool tasks correctly, or refuse to pool them at all.

Robotics benchmarks are suites. LIBERO has ten task groups, Meta-World fifty. The obvious
thing - add up all the successes and compare two totals - is wrong in a specific and
embarrassing way:

    task          policy A        policy B
    reach         5/10  (50%)    45/90  (50%)     tie
    insert       18/90  (20%)     2/10  (20%)     tie
    ---------------------------------------------------
    pooled       23/100 (23%)    47/100 (47%)     B wins by 24 points

B "wins" while being exactly equal on every task, purely because it was evaluated more often
on the easy one. This is Simpson's paradox, and a pooled success rate over a task suite invites
it whenever the per-task episode counts differ - which they routinely do, because evaluations
get interrupted, tasks get added, and reruns are uneven.

The fix is standard and old: stratify. Compare within each task, then combine the within-task
comparisons with the Cochran-Mantel-Haenszel test, which never lets a difference in how often
each task was run masquerade as a difference between policies. Before pooling at all, the
Breslow-Day test asks whether the effect is even the same across tasks: if a policy wins one
task and loses another, a single pooled number is a summary of nothing, and Verdikt says so
rather than averaging the contradiction away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

# below this p-value the per-task effects are considered too different to pool
HOMOGENEITY_ALPHA = 0.05

# a pooled gap smaller than this is not worth calling a paradox even when no task supports it
MATERIAL_POOLED_GAP = 0.02

EPS = 1e-9


@dataclass
class Stratum:
    """One task's 2x2 table: successes and failures for each of two policies."""

    task: str
    a_success: int
    a_n: int
    b_success: int
    b_n: int

    @property
    def a_rate(self) -> float:
        return self.a_success / self.a_n if self.a_n else float("nan")

    @property
    def b_rate(self) -> float:
        return self.b_success / self.b_n if self.b_n else float("nan")

    @property
    def odds_ratio(self) -> float:
        """Haldane-corrected, so a zero cell does not send it to 0 or infinity."""
        a, b = self.a_success + 0.5, self.a_n - self.a_success + 0.5
        c, d = self.b_success + 0.5, self.b_n - self.b_success + 0.5
        return (a * d) / (b * c)

    @property
    def usable(self) -> bool:
        return self.a_n > 0 and self.b_n > 0


@dataclass
class StratifiedResult:
    strata: list[Stratum]
    cmh_statistic: float
    cmh_p: float
    common_odds_ratio: float
    homogeneity_p: float
    homogeneous: bool
    homogeneity_testable: bool
    pooled_a_rate: float
    pooled_b_rate: float
    simpson_reversal: bool
    notes: list[str]


def cochran_mantel_haenszel(strata: list[Stratum]) -> tuple[float, float, float]:
    """Test a common association across strata. Returns (statistic, p, common odds ratio).

    Each task contributes its own expectation and variance, so a task that was evaluated
    twice as often carries twice the weight in the variance - but cannot shift the comparison
    merely by being easier.
    """
    usable = [s for s in strata if s.usable]
    if not usable:
        raise ValueError("no task has episodes for both policies")

    observed = expected = variance = 0.0
    num = den = 0.0
    for s in usable:
        a = s.a_success
        row1, row2 = s.a_n, s.b_n
        col1 = s.a_success + s.b_success
        n = row1 + row2
        if n < 2 or col1 == 0 or col1 == n:
            continue  # a task where nobody or everybody succeeded carries no information
        observed += a
        expected += row1 * col1 / n
        variance += (row1 * row2 * col1 * (n - col1)) / (n * n * (n - 1))
        num += (a * (s.b_n - s.b_success)) / n
        den += (s.b_success * (s.a_n - a)) / n

    if variance <= 0:
        return 0.0, 1.0, float("nan")

    # continuity-corrected, matching the conventional definition
    statistic = (abs(observed - expected) - 0.5) ** 2 / variance
    p = float(chi2.sf(statistic, df=1))
    odds = float(num / den) if den > 0 else float("nan")
    return float(statistic), p, odds


def breslow_day(strata: list[Stratum], common_or: float) -> tuple[float, float, int]:
    """Are the per-task effects the same? Returns (statistic, p, contributing strata).

    A small p-value means the policies' relative performance genuinely differs by task, so a
    single pooled figure describes none of them.

    The degrees of freedom count only the strata that carry information. A task nobody solved
    contributes nothing to the statistic, and counting it in the degrees of freedom would
    quietly inflate the p-value - making a suite look more homogeneous the more useless tasks
    it contains. The third return value is that count, so a caller can tell "the effects agree"
    apart from "there was only one task to compare".
    """
    usable = [s for s in strata if s.usable]
    if len(usable) < 2 or not np.isfinite(common_or) or common_or <= 0:
        return 0.0, 1.0, 0

    statistic = 0.0
    contributing = 0
    for s in usable:
        row1, row2 = s.a_n, s.b_n
        col1 = s.a_success + s.b_success
        n = row1 + row2
        if col1 == 0 or col1 == n:
            continue
        # expected count in cell a under the common odds ratio: solve the quadratic
        coef_a = common_or - 1.0
        coef_b = -(common_or * (row1 + col1) + (row2 - col1))
        coef_c = common_or * row1 * col1
        if abs(coef_a) < 1e-12:
            exp_a = row1 * col1 / n
        else:
            disc = coef_b * coef_b - 4 * coef_a * coef_c
            if disc < 0:
                continue
            root = (-coef_b - np.sqrt(disc)) / (2 * coef_a)
            if not (0 < root < min(row1, col1)):
                root = (-coef_b + np.sqrt(disc)) / (2 * coef_a)
            exp_a = root
        if not (0 < exp_a < min(row1, col1)):
            continue
        var = 1.0 / (1.0 / exp_a + 1.0 / (row1 - exp_a) + 1.0 / (col1 - exp_a)
                     + 1.0 / (row2 - col1 + exp_a))
        if var <= 0:
            continue
        statistic += (s.a_success - exp_a) ** 2 / var
        contributing += 1

    if contributing < 2:
        # nothing to compare a second task against; homogeneity is untestable, not confirmed
        return float(statistic), 1.0, contributing
    return float(statistic), float(chi2.sf(statistic, df=contributing - 1)), contributing


def analyse(strata: list[Stratum]) -> StratifiedResult:
    """Per-task comparison, correctly combined - and a loud warning when pooling would lie."""
    usable = [s for s in strata if s.usable]
    if not usable:
        raise ValueError("no task has episodes for both policies")

    statistic, p, common_or = cochran_mantel_haenszel(strata)
    _bd_stat, bd_p, bd_contributing = breslow_day(strata, common_or)
    testable = bd_contributing >= 2
    homogeneous = bd_p >= HOMOGENEITY_ALPHA

    a_succ = sum(s.a_success for s in usable)
    a_n = sum(s.a_n for s in usable)
    b_succ = sum(s.b_success for s in usable)
    b_n = sum(s.b_n for s in usable)
    pooled_a = a_succ / a_n if a_n else float("nan")
    pooled_b = b_succ / b_n if b_n else float("nan")

    notes: list[str] = []

    # Simpson's paradox. The strict textbook form is a sign flip - ahead on every task, behind
    # once pooled - but requiring a flip misses the case that motivated this module: two
    # policies exactly tied on every task, where uneven episode counts alone manufacture a
    # 24-point pooled gap. Nothing reversed there, and the pooled number is still a fiction.
    # The condition that covers both is: the pooled winner is not the winner of a single task.
    diffs = [s.a_rate - s.b_rate for s in usable]
    pooled_diff = pooled_a - pooled_b
    reversal = False
    if abs(pooled_diff) >= MATERIAL_POOLED_GAP:
        favours_a = pooled_diff > 0
        supporting = [d for d in diffs if (d > EPS if favours_a else d < -EPS)]
        reversal = not supporting
    if reversal:
        ahead, behind = ("the first", "the second") if pooled_diff > 0 else \
                        ("the second", "the first")
        contradicting = sum(1 for d in diffs
                            if (d < -EPS if pooled_diff > 0 else d > EPS))
        detail = (f"{behind} policy is ahead on {contradicting} of {len(usable)} tasks"
                  if contradicting else "the two are level on every task")
        notes.append(
            f"SIMPSON'S PARADOX: the pooled success rate favours {ahead} policy "
            f"({pooled_a:.1%} vs {pooled_b:.1%}), but not one task supports that - "
            f"{detail}. the pooled gap is an artefact of how many episodes each task got, "
            "not a difference between the policies. report the per-task table; the "
            "stratified test below is the one that is not fooled by this."
        )

    if testable and not homogeneous:
        notes.append(
            f"the effect is not the same across tasks (Breslow-Day p={bd_p:.4g}). one policy "
            "wins some tasks and loses others, so a single combined figure summarises none of "
            "them - report the per-task table instead."
        )
    elif not testable:
        notes.append(
            f"only {bd_contributing} task carries information (the rest were solved by "
            "everyone or by no one), so whether the effect is consistent across tasks could "
            "not be tested. this is not evidence that it is."
        )

    counts = {s.task: (s.a_n, s.b_n) for s in usable}
    lopsided = [t for t, (na, nb) in counts.items() if max(na, nb) > 2 * max(1, min(na, nb))]
    if lopsided:
        notes.append(
            f"unequal episode counts on {len(lopsided)} task(s) ({', '.join(lopsided[:3])}"
            f"{'...' if len(lopsided) > 3 else ''}); this is what makes a pooled rate "
            "misleading, and why the stratified test is used here."
        )

    return StratifiedResult(
        strata=usable, cmh_statistic=statistic, cmh_p=p, common_odds_ratio=common_or,
        homogeneity_p=bd_p, homogeneous=homogeneous, homogeneity_testable=testable,
        pooled_a_rate=pooled_a, pooled_b_rate=pooled_b, simpson_reversal=reversal,
        notes=notes,
    )


def coverage_gap(strata: list[Stratum]) -> tuple[int, int, list[str]]:
    """Episodes spent on tasks the other arm never attempted. Returns (a_only, b_only, tasks).

    This is a separate failure from an uneven split of shared tasks. If one arm was evaluated
    on a task the other never ran, its headline success rate is an average over a different
    set of problems - the two numbers are not two measurements of the same quantity, and the
    stratified test cannot repair that because those episodes have nothing to pair against.
    """
    a_only = sum(s.a_n for s in strata if s.a_n > 0 and s.b_n == 0)
    b_only = sum(s.b_n for s in strata if s.b_n > 0 and s.a_n == 0)
    tasks = sorted(s.task for s in strata if not s.usable and (s.a_n or s.b_n))
    return a_only, b_only, tasks


def strata_from_rollouts(rollouts, a: str, b: str) -> list[Stratum]:
    """Build one 2x2 table per task from a rollout list."""
    tasks: dict[str, dict[str, list[int]]] = {}
    for r in rollouts:
        if r.policy_id not in (a, b) or r.success is None:
            continue
        entry = tasks.setdefault(r.task, {a: [0, 0], b: [0, 0]})
        entry[r.policy_id][0] += int(bool(r.success))
        entry[r.policy_id][1] += 1
    return [
        Stratum(task=task, a_success=v[a][0], a_n=v[a][1], b_success=v[b][0], b_n=v[b][1])
        for task, v in sorted(tasks.items())
    ]
