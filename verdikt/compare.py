"""The decision engine: rollouts (+ optional manifests) -> a verdict that survives scrutiny.

Three refusals are enforced here rather than left to the caller's discretion, because the
whole point of the tool is that they cannot be forgotten:

1. a rate is never returned without n and an interval;
2. arms whose training budgets differ materially are not ranked, they are suppressed;
3. "no significant difference" is never reported as "the same" - if the design could not
   have detected the difference in question, the verdict is UNDERPOWERED and carries the n
   that would settle it.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import (
    ArmSummary,
    ComparisonResult,
    Confound,
    PairTest,
    Plan,
    Rollout,
    RunManifest,
    Verdict,
)
from .stats import (
    bonferroni,
    compact_letters,
    holm,
    interval,
    one_sided_lower,
    one_sided_upper,
    prob_a_beats_b,
    required_n,
    unpaired_p,
)
from .stats.power import mde as mde_fn

# how far apart training budgets may be before ranking is refused
SAMPLES_SEEN_RATIO_LIMIT = 2.0

# severity order used when several candidate comparisons disagree
_SEVERITY = {Verdict.REGRESSION: 3, Verdict.NOT_COMPARABLE: 2, Verdict.UNDERPOWERED: 1,
             Verdict.BETTER: 0}


def summarise(rollouts: list[Rollout], ci_method: str = "wilson",
              conf: float = 0.95) -> list[ArmSummary]:
    """One ArmSummary per policy. Rates always travel with their evidence."""
    by_policy: dict[str, list[Rollout]] = defaultdict(list)
    for r in rollouts:
        by_policy[r.policy_id].append(r)

    arms: list[ArmSummary] = []
    for pid, rs in sorted(by_policy.items()):
        graded = [r for r in rs if r.success is not None]
        n = len(graded)
        if n == 0:
            continue
        k = sum(1 for r in graded if r.success)
        lo, hi = interval(k, n, ci_method, conf)
        progresses = [r.progress for r in rs if r.progress is not None]
        bound = None
        if k == 0:
            bound = one_sided_upper(0, n, conf)
        elif k == n:
            bound = one_sided_lower(n, n, conf)
        arms.append(
            ArmSummary(
                policy_id=pid, n=n, successes=k, rate=k / n,
                ci_low=lo, ci_high=hi, ci_method=ci_method, one_sided_bound=bound,
                mean_progress=(sum(progresses) / len(progresses)) if progresses else None,
            )
        )
    return arms


def find_confounds(manifests: dict[str, RunManifest], a: str, b: str) -> list[Confound]:
    """Reasons two arms must not be ranked. Arithmetic, not inference - and the strongest
    opinion in the tool."""
    out: list[Confound] = []
    ma, mb = manifests.get(a), manifests.get(b)
    if not ma or not mb:
        return out

    sa, sb = ma.samples_seen, mb.samples_seen
    if sa and sb:
        ratio = max(sa, sb) / min(sa, sb)
        if ratio >= SAMPLES_SEEN_RATIO_LIMIT:
            out.append(Confound(
                field="samples_seen", a_value=f"{sa:.3g}", b_value=f"{sb:.3g}", ratio=ratio,
                kind="COMPUTE_CONFOUND",
                message=(f"{a} saw {sa:.3g} samples, {b} saw {sb:.3g} ({ratio:.1f}x). "
                         "an architecture claim between these two is not supported; the "
                         "difference in budget is a sufficient alternative explanation."),
            ))

    for field, kind in (("dataset_revision", "DATA_CONFOUND"),
                        ("dataset_content_hash", "DATA_CONFOUND"),
                        ("normalization_mode", "DATA_CONFOUND")):
        va, vb = getattr(ma, field), getattr(mb, field)
        if va and vb and va != vb:
            out.append(Confound(
                field=field, a_value=str(va), b_value=str(vb), kind=kind,
                message=f"{field} differs ({va} vs {vb}); the arms were not trained on the "
                        "same data as far as this manifest can tell.",
            ))
    return out


def compare(
    rollouts: list[Rollout],
    baseline: str | None = None,
    *,
    manifests: dict[str, RunManifest] | None = None,
    test: str = "fisher",
    alpha: float = 0.05,
    correction: str = "bonferroni",
    ci_method: str = "wilson",
    conf: float = 0.95,
    min_lower_bound: float | None = None,
    noninferiority_margin: float | None = None,
    plan: Plan | None = None,
    power_target: float = 0.80,
) -> ComparisonResult:
    """Compare policy arms and issue one verdict."""
    manifests = manifests or {}
    arms = summarise(rollouts, ci_method, conf)
    if not arms:
        raise ValueError("no graded rollouts to compare")
    for arm in arms:
        m = manifests.get(arm.policy_id)
        if m:
            arm.samples_seen = m.samples_seen

    names = [a.policy_id for a in arms]
    by_name = {a.policy_id: a for a in arms}
    if baseline and baseline not in by_name:
        raise ValueError(f"baseline {baseline!r} not among arms: {names}")

    pairs_idx = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    m_tests = len(pairs_idx)
    alpha_adj = bonferroni(alpha, m_tests) if correction == "bonferroni" else alpha

    confounds: list[Confound] = []
    pairs: list[PairTest] = []
    raw_p: list[tuple[int, float]] = []

    for a, b in pairs_idx:
        found = find_confounds(manifests, a, b)
        blocking = [c for c in found if c.kind in ("COMPUTE_CONFOUND", "DATA_CONFOUND")]
        confounds.extend(found)
        A, B = by_name[a], by_name[b]
        if blocking:
            pairs.append(PairTest(
                a=a, b=b, test=test, p_value=float("nan"), alpha_adjusted=alpha_adj,
                significant=False, suppressed_reason=blocking[0].message,
            ))
            continue

        p = unpaired_p(A.successes, A.n, B.successes, B.n, test)
        alt = "barnard" if test == "fisher" else "fisher"
        p_alt = unpaired_p(A.successes, A.n, B.successes, B.n, alt)
        idx = len(pairs)
        pairs.append(PairTest(
            a=a, b=b, test=test, p_value=p, alpha_adjusted=alpha_adj,
            significant=p <= alpha_adj, alt_test=alt, alt_p_value=p_alt,
        ))
        raw_p.append((idx, p))

    if correction == "holm" and raw_p:
        flags = holm([p for _, p in raw_p], alpha)
        for (idx, _), keep in zip(raw_p, flags, strict=True):
            pairs[idx].significant = keep
            pairs[idx].alpha_adjusted = alpha

    sig_map = {(p.a, p.b): p.significant for p in pairs if p.suppressed_reason is None}
    letters = compact_letters(names, sig_map)
    for arm in arms:
        arm.letter = letters.get(arm.policy_id, "")

    verdict, reason, req_n = _decide(
        arms=arms, pairs=pairs, baseline=baseline, alpha=alpha, test=test,
        min_lower_bound=min_lower_bound, noninferiority_margin=noninferiority_margin,
        power_target=power_target,
    )

    return ComparisonResult(
        arms=arms, pairs=pairs, confounds=confounds, verdict=verdict, reason=reason,
        required_n=req_n, plan=plan,
        label_sources=sorted({r.label_source for r in rollouts}),
    )


def _decide(*, arms, pairs, baseline, alpha, test, min_lower_bound, noninferiority_margin,
            power_target) -> tuple[Verdict, str, int | None]:
    """Pick one verdict. Kept separate so the policy is auditable in one screen."""
    by_name = {a.policy_id: a for a in arms}

    if min_lower_bound is not None:
        failing = [a for a in arms if a.ci_low < min_lower_bound]
        if failing:
            worst = min(failing, key=lambda a: a.ci_low)
            return (Verdict.REGRESSION,
                    f"{worst.policy_id}: 95% lower bound {worst.ci_low:.1%} is below the "
                    f"required {min_lower_bound:.1%} (point estimate {worst.rate:.1%} is not "
                    "evidence at this n)", None)
        return (Verdict.BETTER,
                f"all arms clear the required lower bound of {min_lower_bound:.1%}", None)

    if baseline is None:
        undecided = [p for p in pairs if p.suppressed_reason is None and not p.significant]
        if any(p.suppressed_reason for p in pairs):
            return (Verdict.NOT_COMPARABLE,
                    "at least one pair is confounded; see the CONFOUND block", None)
        if undecided:
            return (Verdict.UNDERPOWERED,
                    f"{len(undecided)} of {len(pairs)} pairs are indistinguishable at this n",
                    None)
        return (Verdict.BETTER, "all pairs are separated at the corrected alpha", None)

    base = by_name[baseline]
    outcomes: list[tuple[Verdict, str, int | None]] = []

    for cand in arms:
        if cand.policy_id == baseline:
            continue
        pair = next((p for p in pairs
                     if {p.a, p.b} == {cand.policy_id, baseline}), None)
        if pair is None:
            continue
        if pair.suppressed_reason:
            outcomes.append((Verdict.NOT_COMPARABLE,
                             f"{cand.policy_id} vs {baseline}: {pair.suppressed_reason}", None))
            continue

        if noninferiority_margin is not None:
            if cand.ci_low >= base.rate - noninferiority_margin:
                outcomes.append((Verdict.BETTER,
                                 f"{cand.policy_id} is non-inferior to {baseline} at margin "
                                 f"{noninferiority_margin:.1%}", None))
            else:
                outcomes.append((Verdict.REGRESSION,
                                 f"{cand.policy_id} cannot be shown non-inferior: lower bound "
                                 f"{cand.ci_low:.1%} < {base.rate - noninferiority_margin:.1%}",
                                 None))
            continue

        if pair.significant:
            if cand.rate > base.rate:
                outcomes.append((Verdict.BETTER,
                                 f"{cand.policy_id} beats {baseline} "
                                 f"({cand.rate:.1%} vs {base.rate:.1%}, p={pair.p_value:.4g})",
                                 None))
            else:
                outcomes.append((Verdict.REGRESSION,
                                 f"{cand.policy_id} is worse than {baseline} "
                                 f"({cand.rate:.1%} vs {base.rate:.1%}, p={pair.p_value:.4g})",
                                 None))
            continue

        # not significant: is that "no difference" or "not enough episodes"?
        n = min(cand.n, base.n)
        p0 = _clamp(base.rate)
        p1 = _clamp(cand.rate)
        need = None
        if abs(p1 - p0) > 1e-9:
            need = required_n(p0, p1, power=power_target, alpha=alpha, test=test)
        detectable = mde_fn(n, p0, power=power_target, alpha=alpha, test=test)
        if need is not None and need > n:
            outcomes.append((
                Verdict.UNDERPOWERED,
                f"{cand.policy_id} vs {baseline}: observed {cand.rate:.1%} vs {base.rate:.1%} "
                f"is not resolvable at n={n}. a difference of this size needs ~{need}/arm at "
                f"{power_target:.0%} power; at n={n} the smallest resolvable difference is "
                f"{detectable:.1%}" if detectable else
                f"{cand.policy_id} vs {baseline}: not resolvable at n={n}; needs ~{need}/arm",
                need))
        else:
            outcomes.append((Verdict.BETTER,
                             f"{cand.policy_id} shows no regression against {baseline} "
                             f"(p={pair.p_value:.4g}, and the design could resolve a "
                             f"{detectable:.1%} difference)" if detectable else
                             f"{cand.policy_id} shows no regression against {baseline}", None))

    if not outcomes:
        return (Verdict.UNDERPOWERED, "nothing to compare against the baseline", None)
    return max(outcomes, key=lambda o: _SEVERITY[o[0]])


def _clamp(p: float, eps: float = 0.005) -> float:
    """Keep planning rates strictly inside (0, 1).

    An observed 0/20 is not evidence that the true rate is zero, and required-N is undefined
    there; we plan against a small non-zero rate and say so rather than refusing to help.
    """
    return min(1 - eps, max(eps, p))


def posterior_table(arms: list[ArmSummary], baseline: str) -> dict[str, float]:
    """P(arm > baseline) for each arm - what people think overlapping error bars mean."""
    base = next(a for a in arms if a.policy_id == baseline)
    return {
        a.policy_id: prob_a_beats_b(a.successes, a.n, base.successes, base.n)
        for a in arms if a.policy_id != baseline
    }
