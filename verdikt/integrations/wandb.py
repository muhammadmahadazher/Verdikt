"""Write verdicts back into Weights & Biases.

Verdikt should show up where a team already looks, not ask them to open one more tool. W&B
stores and plots the numbers; it has no opinion about whether a difference is real. This
module puts the opinion next to the numbers.

The payload is built by a pure function so it can be tested without a network, an account or
an API key - `build_payload` is covered by unit tests, and `push` is the thin, obvious wrapper
that hands it to the client.
"""

from __future__ import annotations

from typing import Any

from ..schema import ComparisonResult

SUMMARY_PREFIX = "verdikt"


def build_payload(result: ComparisonResult, baseline: str | None = None,
                  posteriors: dict[str, float] | None = None) -> dict[str, Any]:
    """Flat summary keys for a W&B run.

    Deliberate choices:
      - the verdict is written as both a name and an exit code, so a dashboard can filter on
        either without parsing strings;
      - every rate is accompanied by its interval bounds and n, because a bare rate in a
        dashboard is exactly the artefact this tool exists to prevent;
      - `0/n` arms carry their one-sided bound, so a panel never renders a hard zero.
    """
    name = {0: "BETTER", 1: "REGRESSION", 2: "UNDERPOWERED", 3: "NOT_COMPARABLE"}[
        int(result.verdict)]
    payload: dict[str, Any] = {
        f"{SUMMARY_PREFIX}/verdict": name,
        f"{SUMMARY_PREFIX}/exit_code": int(result.verdict),
        f"{SUMMARY_PREFIX}/reason": result.reason,
        f"{SUMMARY_PREFIX}/n_arms": len(result.arms),
        f"{SUMMARY_PREFIX}/confounded": bool(result.confounds),
    }
    if baseline:
        payload[f"{SUMMARY_PREFIX}/baseline"] = baseline
    if result.required_n:
        payload[f"{SUMMARY_PREFIX}/required_n"] = int(result.required_n)
    if result.pairs:
        payload[f"{SUMMARY_PREFIX}/test"] = result.pairs[0].test
        payload[f"{SUMMARY_PREFIX}/alpha_adjusted"] = result.pairs[0].alpha_adjusted

    for arm in result.arms:
        key = f"{SUMMARY_PREFIX}/arm/{arm.policy_id}"
        payload[f"{key}/n"] = arm.n
        payload[f"{key}/successes"] = arm.successes
        payload[f"{key}/rate"] = arm.rate
        payload[f"{key}/ci_low"] = arm.ci_low
        payload[f"{key}/ci_high"] = arm.ci_high
        payload[f"{key}/group"] = arm.letter or "-"
        if arm.n_ungraded:
            payload[f"{key}/ungraded"] = arm.n_ungraded
        if arm.one_sided_bound is not None:
            payload[f"{key}/one_sided_bound"] = arm.one_sided_bound
        if arm.samples_seen:
            payload[f"{key}/samples_seen"] = arm.samples_seen

    for pid, prob in (posteriors or {}).items():
        payload[f"{SUMMARY_PREFIX}/posterior/P({pid}>{baseline})"] = prob

    return payload


def build_table_rows(result: ComparisonResult) -> tuple[list[str], list[list[Any]]]:
    """Rows for a wandb.Table, in the same shape as the terminal and HTML output."""
    columns = ["policy", "n", "successes", "rate", "ci_low", "ci_high", "group",
               "one_sided_bound", "ungraded"]
    rows = [[
        a.policy_id, a.n, a.successes, round(a.rate, 4), round(a.ci_low, 4),
        round(a.ci_high, 4), a.letter or "-",
        None if a.one_sided_bound is None else round(a.one_sided_bound, 4),
        a.n_ungraded,
    ] for a in result.arms]
    return columns, rows


def parse_run_path(path: str) -> tuple[str | None, str, str]:
    """Accept entity/project/run_id or project/run_id."""
    parts = [p for p in path.split("/") if p]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    raise ValueError(
        f"cannot parse W&B run path {path!r}; expected entity/project/run_id or project/run_id")


def push(result: ComparisonResult, run_path: str, *, baseline: str | None = None,
         posteriors: dict[str, float] | None = None, html_report: str | None = None,
         model_card: str | None = None) -> dict[str, Any]:
    """Attach the verdict, the arm table and the report to an existing W&B run.

    Uses `resume="allow"` so this works on a finished training run - the usual case, since the
    evaluation happens after training ends.
    """
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
        raise RuntimeError(
            "W&B write-back needs the wandb package: pip install 'verdikt[wandb]'"
        ) from exc

    entity, project, run_id = parse_run_path(run_path)
    payload = build_payload(result, baseline, posteriors)
    columns, rows = build_table_rows(result)

    run = wandb.init(entity=entity, project=project, id=run_id, resume="allow",
                     job_type="verdikt-eval", reinit=True)
    try:
        run.summary.update(payload)
        run.log({f"{SUMMARY_PREFIX}/arms": wandb.Table(columns=columns, data=rows)})

        if html_report or model_card:
            artifact = wandb.Artifact(
                name=f"verdikt-report-{run_id}", type="evaluation-report",
                description=f"Verdikt {payload[f'{SUMMARY_PREFIX}/verdict']}: {result.reason}",
                metadata=payload,
            )
            if html_report:
                artifact.add_file(html_report, name="report.html")
            if model_card:
                artifact.add_file(model_card, name="MODEL_CARD.md")
            run.log_artifact(artifact)
    finally:
        run.finish()

    return payload
