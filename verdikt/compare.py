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
    paired_p,
    prob_a_beats_b,
    required_n,
    unpaired_p,
)
from .stats.power import mde as mde_fn

# how far apart training budgets may be before ranking is refused
SAMPLES_SEEN_RATIO_LIMIT = 2.0

# The smallest difference a reader would act on. A non-significant result only means
# "no difference" if the design could have resolved at least this much; otherwise the honest
# verdict is UNDERPOWERED. Override per call site if your task has a different bar.
PRACTICAL_DIFFERENCE = 0.10

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
                policy_id=pid, n=n, successes=k, n_ungraded=len(rs) - n, rate=k / n,
                ci_low=lo, ci_high=hi, ci_method=ci_method, one_sided_bound=bound,
                mean_progress=(sum(progresses) / len(progresses)) if progresses else None,
            )
        )
    return arms


class PairingError(ValueError):
    """Raised when a paired comparison is requested but the episodes cannot be aligned."""


def pair_outcomes(rollouts: list[Rollout], a: str, b: str,
                  allow_index_pairing: bool = False) -> tuple[int, int, int, int, str]:
    """Align two arms episode-by-episode. Returns (both, only_a, only_b, neither, key).

    Pairing is what buys power: it removes scene difficulty from the comparison, so two
    policies that both fail the hard scenes and differ only on the easy ones are separated
    with far fewer episodes than an unpaired test needs.

    It is also the easiest thing in this tool to get silently wrong. Pairing episode 7 of one
    run against episode 7 of another is only meaningful if those two episodes were the *same
    scene*. When the harness records a per-episode seed we can verify that. When it does not -
    and `lerobot-eval` does not - the alignment rests on an assumption about how the runs were
    launched, so the caller has to state it explicitly rather than have it assumed for them.
    """
    by_arm: dict[str, dict] = {a: {}, b: {}}
    for r in rollouts:
        if r.policy_id in by_arm and r.success is not None:
            key = r.seed if r.seed is not None else r.episode_idx
            by_arm[r.policy_id][key] = bool(r.success)

    seeded = all(r.seed is not None for r in rollouts
                 if r.policy_id in by_arm and r.success is not None)
    key_name = "seed" if seeded else "episode index"

    if not seeded and not allow_index_pairing:
        raise PairingError(
            "paired comparison needs episodes that are known to be the same scene, and this "
            "source records no per-episode seed. re-run the evaluation with a fixed --seed "
            "and identical batch size for both policies, then pass --assume-aligned to "
            "confirm you did - verdikt will not assume it for you."
        )

    shared = sorted(set(by_arm[a]) & set(by_arm[b]))
    if not shared:
        raise PairingError(f"no episodes are shared between {a} and {b} by {key_name}")
    missing = (len(by_arm[a]) - len(shared)) + (len(by_arm[b]) - len(shared))
    if missing > 0.1 * (len(by_arm[a]) + len(by_arm[b])):
        raise PairingError(
            f"{a} and {b} share only {len(shared)} episodes by {key_name} "
            f"({len(by_arm[a])} vs {len(by_arm[b])} recorded); they were not run over the "
            "same set of scenes, so pairing them would compare different problems"
        )

    both = sum(by_arm[a][k] and by_arm[b][k] for k in shared)
    only_a = sum(by_arm[a][k] and not by_arm[b][k] for k in shared)
    only_b = sum(by_arm[b][k] and not by_arm[a][k] for k in shared)
    neither = len(shared) - both - only_a - only_b
    return both, only_a, only_b, neither, key_name


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
    practical_difference: float = PRACTICAL_DIFFERENCE,
    paired: bool = False,
    allow_index_pairing: bool = False,
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

    # Pass 1: decide which pairs are testable at all. A suppressed pair consumes no alpha -
    # counting it in the family over-corrects every test that actually ran, which can hide a
    # real regression behind a threshold that was never needed.
    confounds: list[Confound] = []
    blocked: dict[tuple[str, str], str] = {}
    for a, b in pairs_idx:
        found = find_confounds(manifests, a, b)
        confounds.extend(found)
        blocking = [c for c in found if c.kind in ("COMPUTE_CONFOUND", "DATA_CONFOUND")]
        if blocking:
            blocked[(a, b)] = blocking[0].message

    live = [pair for pair in pairs_idx if pair not in blocked]
    m_tests = max(1, len(live))
    alpha_adj = bonferroni(alpha, m_tests) if correction == "bonferroni" else alpha

    # Pass 2: run the tests that survive.
    pairs: list[PairTest] = []
    raw_p: list[tuple[int, float]] = []
    for a, b in pairs_idx:
        if (a, b) in blocked:
            pairs.append(PairTest(
                a=a, b=b, test=test, p_value=float("nan"), alpha_adjusted=alpha_adj,
                significant=False, suppressed_reason=blocked[(a, b)],
            ))
            continue
        A, B = by_name[a], by_name[b]
        if paired:
            _both, only_a, only_b, _neither, key_name = pair_outcomes(
                rollouts, a, b, allow_index_pairing=allow_index_pairing)
            p = paired_p(only_a, only_b)
            used_test = f"mcnemar (paired by {key_name}, {only_a + only_b} discordant)"
            alt, p_alt = None, None
        else:
            p = unpaired_p(A.successes, A.n, B.successes, B.n, test)
            used_test = test
            alt = "barnard" if test == "fisher" else "fisher"
            p_alt = unpaired_p(A.successes, A.n, B.successes, B.n, alt)
        idx = len(pairs)
        pairs.append(PairTest(
            a=a, b=b, test=used_test, p_value=p, alpha_adjusted=alpha_adj,
            significant=p <= alpha_adj, alt_test=alt, alt_p_value=p_alt,
        ))
        raw_p.append((idx, p))

    if correction == "holm" and raw_p:
        flags = holm([p for _, p in raw_p], alpha)
        # Holm uses a different threshold per rank; record the one this comparison actually
        # faced, so a reader can never find p <= alpha_adjusted next to significant=False.
        order = sorted(range(len(raw_p)), key=lambda i: raw_p[i][1])
        thresholds = {}
        for rank, i in enumerate(order):
            thresholds[raw_p[i][0]] = alpha / (len(raw_p) - rank)
        for (idx, _), keep in zip(raw_p, flags, strict=True):
            pairs[idx].significant = keep
            pairs[idx].alpha_adjusted = thresholds[idx]

    sig_map = {(p.a, p.b): p.significant for p in pairs if p.suppressed_reason is None}
    letters = compact_letters(names, sig_map)
    for arm in arms:
        arm.letter = letters.get(arm.policy_id, "")

    verdict, reason, req_n = _decide(
        arms=arms, pairs=pairs, baseline=baseline, alpha=alpha, test=test,
        min_lower_bound=min_lower_bound, noninferiority_margin=noninferiority_margin,
        power_target=power_target, conf=conf, practical_difference=practical_difference,
    )

    return ComparisonResult(
        arms=arms, pairs=pairs, confounds=confounds, verdict=verdict, reason=reason,
        required_n=req_n, plan=plan,
        label_sources=sorted({r.label_source for r in rollouts}),
    )


def _decide(*, arms, pairs, baseline, alpha, test, min_lower_bound, noninferiority_margin,
            power_target, conf=0.95, practical_difference=PRACTICAL_DIFFERENCE
            ) -> tuple[Verdict, str, int | None]:
    """Pick one verdict. Kept separate so the policy is auditable in one screen."""
    by_name = {a.policy_id: a for a in arms}
    conf_pct = f"{conf:.0%}"

    # A confound outranks every other question: if two arms must not be compared, no gate or
    # threshold makes them comparable again. This runs first so that adding --min-lower-bound
    # can never launder a suppressed comparison into a pass.
    suppressed = [p for p in pairs if p.suppressed_reason]
    if suppressed and len(arms) > 1:
        return (Verdict.NOT_COMPARABLE,
                f"{len(suppressed)} of {len(pairs)} comparisons are confounded and were not "
                f"ranked: {suppressed[0].suppressed_reason}", None)

    if min_lower_bound is not None:
        # Gate the candidates, not the reference. The baseline is what you are measuring
        # against; holding it to the candidate's bar reports a regression for a candidate
        # that cleared it.
        gated = [a for a in arms if a.policy_id != baseline] if baseline else arms
        failing = [a for a in gated if a.ci_low < min_lower_bound]
        if failing:
            worst = min(failing, key=lambda a: a.ci_low)
            return (Verdict.REGRESSION,
                    f"{worst.policy_id}: {conf_pct} lower bound {worst.ci_low:.1%} is below the "
                    f"required {min_lower_bound:.1%} (the point estimate {worst.rate:.1%} is "
                    "not evidence at this n)", None)
        return (Verdict.BETTER,
                f"all candidate arms clear the required lower bound of {min_lower_bound:.1%} "
                f"at {conf_pct} confidence", None)

    if baseline is None:
        tested = [p for p in pairs if p.suppressed_reason is None]
        if not tested:
            return (Verdict.UNDERPOWERED,
                    "there is nothing to compare: a single arm carries no comparison. give "
                    "--baseline and at least two arms, or gate with --min-lower-bound", None)
        undecided = [p for p in tested if not p.significant]
        if undecided:
            return (Verdict.UNDERPOWERED,
                    f"{len(undecided)} of {len(tested)} comparisons are indistinguishable at "
                    "this n", None)
        return (Verdict.BETTER,
                f"all {len(tested)} comparisons are separated at the corrected alpha", None)

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
            threshold = base.rate - noninferiority_margin
            if threshold <= 0:
                # With a margin at or above the baseline rate the test can never fail - a
                # zero-success candidate would "pass". Refusing is the only honest answer.
                outcomes.append((
                    Verdict.NOT_COMPARABLE,
                    f"a non-inferiority margin of {noninferiority_margin:.1%} against a "
                    f"{base.rate:.1%} baseline is vacuous: the threshold falls to "
                    f"{threshold:.1%}, so every candidate passes regardless of performance. "
                    f"choose a margin below {base.rate:.1%}", None))
            elif cand.ci_low >= threshold:
                outcomes.append((Verdict.BETTER,
                                 f"{cand.policy_id} is non-inferior to {baseline} at margin "
                                 f"{noninferiority_margin:.1%} (lower bound {cand.ci_low:.1%} "
                                 f">= {threshold:.1%})", None))
            else:
                outcomes.append((Verdict.REGRESSION,
                                 f"{cand.policy_id} cannot be shown non-inferior: lower bound "
                                 f"{cand.ci_low:.1%} < {threshold:.1%}", None))
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

        # Not significant. That is only "no difference" if the design could have found one.
        n = min(cand.n, base.n)
        p0, clamped_base = _clamp(base.rate)
        p1, clamped_cand = _clamp(cand.rate)
        detectable = mde_fn(n, p0, power=power_target, alpha=alpha, test=test)
        note = ""
        if clamped_base or clamped_cand:
            note = (" (planning figures use a floor of 0.5% in place of an observed 0%, since "
                    "required-N is undefined at exactly zero)")

        # Can this design resolve a difference anyone would act on? If not, a null result is
        # uninformative no matter how close the two observed rates happen to be.
        if detectable is None or detectable > practical_difference:
            need = required_n(p0, min(0.999, p0 + practical_difference),
                              power=power_target, alpha=alpha, test=test)
            detect_txt = (f"the smallest difference it could resolve is {detectable:.1%}"
                          if detectable else
                          "it cannot resolve any difference at this size")
            outcomes.append((
                Verdict.UNDERPOWERED,
                f"{cand.policy_id} vs {baseline}: {cand.rate:.1%} vs {base.rate:.1%} is not "
                f"significant (p={pair.p_value:.4g}), but at n={n} {detect_txt} - so this is "
                f"not evidence that they are equivalent. resolving a "
                f"{practical_difference:.0%} difference needs ~{need}/arm at "
                f"{power_target:.0%} power{note}", need))
        else:
            outcomes.append((
                Verdict.BETTER,
                f"{cand.policy_id} shows no regression against {baseline} "
                f"(p={pair.p_value:.4g}); the design could have resolved a "
                f"{detectable:.1%} difference, so this null result is informative", None))

    if not outcomes:
        return (Verdict.UNDERPOWERED, "nothing to compare against the baseline", None)
    return max(outcomes, key=lambda o: _SEVERITY[o[0]])


def _clamp(p: float, eps: float = 0.005) -> tuple[float, bool]:
    """Keep planning rates strictly inside (0, 1), and report whether it was necessary.

    An observed 0/20 is not evidence that the true rate is zero, and required-N is undefined
    there; we plan against a small non-zero rate. The caller discloses that substitution in
    the verdict text rather than letting a silently invented number drive the advice.
    """
    clamped = min(1 - eps, max(eps, p))
    return clamped, abs(clamped - p) > 1e-12


def posterior_table(arms: list[ArmSummary], baseline: str) -> dict[str, float]:
    """P(arm > baseline) for each arm - what people think overlapping error bars mean."""
    base = next(a for a in arms if a.policy_id == baseline)
    return {
        a.policy_id: prob_a_beats_b(a.successes, a.n, base.successes, base.n)
        for a in arms if a.policy_id != baseline
    }
