"""Verdikt command line.

Output rules, enforced here so they cannot be bypassed by a caller in a hurry:
  - a success rate is never printed without n and an interval;
  - 0/n and n/n print their exact one-sided bound, because "0%" is not what happened;
  - the test that produced a p-value is always named, and the alternative test's value is
    shown whenever the two would disagree about significance.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .compare import compare as run_compare
from .compare import posterior_table
from .ingest import autodetect, available, get
from .ingest.base import AdapterError
from .schema import Plan, Rollout, RunManifest, Verdict
from .stats import mde, normal_approx_n, power_exact, required_n
from .theme import VERDICT

console = Console()


def _echo_verdict(code: Verdict, reason: str) -> None:
    name, _hex, style = VERDICT[int(code)]
    console.print()
    console.print(f"[{style}]VERDICT   {name}[/]")
    for line in _wrap(reason, 88):
        console.print(f"          [dim]{line}[/]")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _load(paths: tuple[str, ...], adapter: str | None, policy_id: str | None,
          mapping: dict[str, str] | None = None) -> list[Rollout]:
    expanded: list[Path] = []
    for p in paths:
        hits = [Path(h) for h in glob.glob(p)] or ([Path(p)] if Path(p).exists() else [])
        if not hits:
            raise click.ClickException(f"no files matched {p!r}")
        expanded.extend(hits)

    rollouts: list[Rollout] = []
    for path in expanded:
        if path.suffix.lower() == ".parquet":
            rollouts.extend(_load_parquet(path))
            continue
        ad = get(adapter) if adapter else autodetect(path)
        try:
            if ad.name == "csv":
                rollouts.extend(ad.parse(path, policy_id, mapping or {}))
            else:
                rollouts.extend(ad.parse(path, policy_id))
        except AdapterError as exc:
            raise click.ClickException(str(exc)) from exc
    return rollouts


def _load_parquet(path: Path) -> list[Rollout]:
    """Re-read a canonical table produced by `verdikt ingest`."""
    import pandas as pd

    df = pd.read_parquet(path)
    out: list[Rollout] = []
    for rec in df.to_dict(orient="records"):
        clean = {k: (None if _isna(v) else v) for k, v in rec.items()}
        out.append(Rollout(**clean))
    return out


def _isna(v) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="verdikt")
def main() -> None:
    """Verdikt - refuse to draw a conclusion the data does not support.

    Reads the eval JSON, dataset files and run configs you already have. Trains nothing,
    runs nothing on a GPU, needs no robot.
    """


# ------------------------------------------------------------------ ingest
@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--adapter", type=click.Choice(available()), default=None,
              help="force an adapter instead of detecting by content")
@click.option("--policy-id", default=None, help="override the inferred policy name")
@click.option("--map", "mapping", multiple=True, metavar="DEST=SRC",
              help="csv adapter only: map your column names onto canonical fields")
@click.option("--out", type=click.Path(), default="rollouts.parquet", show_default=True)
def ingest(paths, adapter, policy_id, mapping, out):
    """Convert any harness's eval output into one canonical rollout table."""
    parsed_map = dict(m.split("=", 1) for m in mapping) if mapping else {}
    rollouts = _load(paths, adapter, policy_id, parsed_map)

    import pandas as pd

    df = pd.DataFrame([r.model_dump() for r in rollouts])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    by_policy = df.groupby("policy_id").size().to_dict()
    console.print(f"[green]wrote[/] {out}  [dim]{len(df)} rollouts[/]")
    for pid, n in sorted(by_policy.items()):
        console.print(f"  [dim]{pid:24s} {n:5d} episodes[/]")
    if df["seed"].isna().all():
        console.print("[yellow]note[/] no per-episode seeds in this source: paired tests "
                      "(McNemar) are unavailable. see `verdikt doctor --check-seeds`.")


# -------------------------------------------------------------------- plan
@main.command()
@click.option("--p0", type=float, required=True, help="baseline success rate you expect")
@click.option("--mde", "mde_arg", type=float, default=None,
              help="difference you need to detect, in absolute percentage points (e.g. 0.15)")
@click.option("--budget", type=int, default=None,
              help="inverse mode: episodes per arm you can afford")
@click.option("--power", type=float, default=0.80, show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("--test", type=click.Choice(["fisher", "barnard", "boschloo"]), default="fisher",
              show_default=True)
@click.option("--hypothesis", default="", help="what you are testing, recorded in the plan")
@click.option("--out", type=click.Path(), default="plan.json", show_default=True)
@click.option("--fast", is_flag=True, help="also show the normal approximation, for contrast")
def plan(p0, mde_arg, budget, power, alpha, test, hypothesis, out, fast):
    """How many episodes do you actually need - computed through the test that will decide.

    The normal approximation under-recommends: it claims 31/arm for 35% vs 70%, where the
    exact test delivers only 0.749 power. Verdikt plans with the test that issues the verdict.
    """
    if mde_arg is None and budget is None:
        raise click.ClickException("give either --mde (how big an effect) or --budget (how many episodes)")

    console.print()
    if mde_arg is not None:
        p1 = min(0.999, p0 + mde_arg)
        with console.status(f"computing exact power through {test}..."):
            n = required_n(p0, p1, power=power, alpha=alpha, test=test)
        if n is None:
            console.print("[red]no n below 2000/arm reaches this power[/]")
            raise SystemExit(2)
        console.print(f"[bold]required N per arm[/]  [dim](exact, test={test}, "
                      f"alpha={alpha} two-sided, power={power:.0%})[/]")
        console.print(f"  {p0:.0%} vs {p1:.0%}  ->  [bold cyan]{n}[/] per arm")
        if fast:
            na = normal_approx_n(p0, p1, power, alpha)
            realised = power_exact(na, p0, p1, alpha, test)
            console.print(f"  [dim]normal approximation would have said {na}/arm -> "
                          f"realised power {realised:.3f}[/]")
        planned_n = n
    else:
        with console.status(f"computing detectable effect at n={budget}..."):
            d = mde(budget, p0, power=power, alpha=alpha, test=test)
        if d is None:
            console.print(f"[yellow]at n={budget}/arm no effect below 100pp reaches "
                          f"{power:.0%} power against a {p0:.0%} baseline.[/]")
            console.print("[dim]this budget cannot answer a comparison question.[/]")
        else:
            console.print(f"[bold]at your budget of n={budget}/arm[/]")
            console.print(f"  smallest detectable difference vs {p0:.0%}: "
                          f"[bold cyan]{d * 100:.1f} pp[/]")
        planned_n = budget

    commitment = {"test": test, "alpha": alpha, "alternative": "two-sided",
                  "planned_n": planned_n, "hypothesis": hypothesis}
    h = hashlib.blake2b(json.dumps(commitment, sort_keys=True).encode(), digest_size=8).hexdigest()
    p = Plan(test=test, alpha=alpha, alternative="two-sided", baseline_rate=p0, mde=mde_arg,
             power=power, planned_n=planned_n, hypothesis=hypothesis, commitment_hash=h)
    Path(out).write_text(p.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\n[green]wrote[/] {out}  [dim]pre-registration hash {h}[/]")
    console.print("[dim]pin this before you look at results; `compare --plan` verifies it.[/]")


# ----------------------------------------------------------------- compare
@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--baseline", default=None, help="the arm everything is judged against")
@click.option("--adapter", type=click.Choice(available()), default=None)
@click.option("--manifests", multiple=True, help="run manifest json (glob ok) for confound checks")
@click.option("--test", type=click.Choice(["fisher", "barnard", "boschloo"]), default="fisher",
              show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("--correction", type=click.Choice(["bonferroni", "holm", "none"]),
              default="bonferroni", show_default=True)
@click.option("--interval", "ci_method",
              type=click.Choice(["wilson", "jeffreys", "clopper-pearson"]), default="wilson",
              show_default=True)
@click.option("--min-lower-bound", type=float, default=None,
              help="gate on the CI LOWER bound (there is deliberately no --min-success)")
@click.option("--noninferiority", "noninf", type=float, default=None, metavar="MARGIN",
              help="pass if the candidate is no worse than baseline by more than MARGIN")
@click.option("--plan", "plan_path", type=click.Path(exists=True), default=None,
              help="verify the pre-registered analysis commitment")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "markdown"]),
              default="text", show_default=True)
def compare(paths, baseline, adapter, manifests, test, alpha, correction, ci_method,
            min_lower_bound, noninf, plan_path, fmt):
    """Is checkpoint B actually better than A - or did you just not run enough episodes?"""
    rollouts = _load(paths, adapter, None)

    mans: dict[str, RunManifest] = {}
    for pattern in manifests:
        for hit in glob.glob(pattern) or [pattern]:
            m = RunManifest.model_validate_json(Path(hit).read_text(encoding="utf-8"))
            mans[m.policy_id] = m

    plan_obj = None
    if plan_path:
        plan_obj = Plan.model_validate_json(Path(plan_path).read_text(encoding="utf-8"))
        if plan_obj.test != test:
            raise click.ClickException(
                f"pre-registered test is {plan_obj.test!r} but you passed --test {test!r}. "
                "changing the test after seeing data is test-shopping; rerun with the "
                "registered test, or start a new plan and say so."
            )
        test, alpha = plan_obj.test, plan_obj.alpha

    result = run_compare(
        rollouts, baseline, manifests=mans, test=test, alpha=alpha,
        correction=("none" if correction == "none" else correction),
        ci_method=ci_method, min_lower_bound=min_lower_bound,
        noninferiority_margin=noninf, plan=plan_obj,
    )

    if fmt == "json":
        click.echo(result.model_dump_json(indent=2))
        raise SystemExit(int(result.verdict))
    if fmt == "markdown":
        _print_markdown(result, baseline)
        raise SystemExit(int(result.verdict))

    _print_text(result, baseline, plan_obj)
    raise SystemExit(int(result.verdict))


def _print_text(result, baseline, plan_obj) -> None:
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("policy")
    table.add_column("n", justify="right")
    table.add_column("success", justify="right")
    table.add_column(f"95% CI ({result.arms[0].ci_method})", justify="right")
    table.add_column("grp", justify="center")
    table.add_column("samples", justify="right")

    for a in result.arms:
        table.add_row(
            a.policy_id, str(a.n), f"{a.successes}/{a.n}  {a.rate:6.1%}",
            f"[{a.ci_low * 100:5.1f}, {a.ci_high * 100:5.1f}]",
            a.letter or "-",
            f"{a.samples_seen:.2e}" if a.samples_seen else "-",
        )
    console.print()
    console.print(table)
    console.print("[dim]arms sharing a group letter are not distinguishable at this n[/]")

    # An episode that carries no success label is not a success. Dropping it silently turns
    # a broken grader into a perfect score, so it is always reported.
    ungraded = [a for a in result.arms if a.n_ungraded]
    if ungraded:
        console.print()
        for a in ungraded:
            total = a.n + a.n_ungraded
            console.print(f"  [yellow]{a.policy_id}: {a.n_ungraded} of {total} rollouts carry "
                          f"no success label[/] and are excluded from n.")
        console.print("  [dim]the rate above describes only the graded episodes. if your "
                      "grader failed, this rate is not the number you think it is.[/]")

    for a in result.arms:
        if a.successes == 0 and a.one_sided_bound is not None:
            console.print(f"\n  [yellow]{a.successes}/{a.n} does not mean zero.[/] "
                          f"one-sided 95% upper bound: [bold]{a.one_sided_bound:.1%}[/]")
        elif a.successes == a.n and a.one_sided_bound is not None:
            console.print(f"\n  [yellow]{a.n}/{a.n} does not mean certainty.[/] "
                          f"one-sided 95% lower bound: [bold]{a.one_sided_bound:.1%}[/]")

    if plan_obj:
        console.print(f"\n[dim]pre-registered:[/] test={plan_obj.test} alpha={plan_obj.alpha} "
                      f"planned_n={plan_obj.planned_n} [green]hash {plan_obj.commitment_hash} "
                      "VERIFIED[/]")

    shown = [p for p in result.pairs]
    if shown:
        m = len([p for p in shown if p.suppressed_reason is None])
        console.print(f"\n[bold]pairwise[/] [dim](test={shown[0].test}, "
                      f"m={m}, corrected alpha={shown[0].alpha_adjusted:.5f})[/]")
        for p in shown:
            if p.suppressed_reason:
                console.print(f"  {p.a} vs {p.b}   [magenta]SUPPRESSED[/] [dim](confounded)[/]")
                continue
            mark = "[green]SIGNIFICANT[/]" if p.significant else "[dim]not significant[/]"
            line = f"  {p.a} vs {p.b}   p={p.p_value:.4g}  {mark}"
            if p.alt_p_value is not None:
                flips = (p.alt_p_value <= p.alpha_adjusted) != p.significant
                if flips:
                    line += (f"  [yellow]<- {p.alt_test} would say p={p.alt_p_value:.4g} "
                             "and disagree[/]")
            console.print(line)

    if baseline:
        post = posterior_table(result.arms, baseline)
        if post:
            console.print("\n[bold]posterior[/] [dim](uniform prior)[/]")
            for pid, prob in post.items():
                console.print(f"  P({pid} > {baseline}) = {prob:.3f}")

    seen = set()
    for c in result.confounds:
        key = (c.field, c.a_value, c.b_value)
        if key in seen:
            continue
        seen.add(key)
        console.print(f"\n[magenta]CONFOUND[/]  [dim]{c.kind}[/]")
        for line in _wrap(c.message, 84):
            console.print(f"          {line}")

    if result.label_sources and result.label_sources != ["simulator"]:
        console.print(f"\n[dim]label sources present: {', '.join(result.label_sources)}[/]")

    _echo_verdict(result.verdict, result.reason)


# ------------------------------------------------------------------ doctor
@main.command()
@click.option("--train-config", type=click.Path(exists=True), default=None,
              help="a LeRobot train_config.json to inspect")
@click.option("--dataset-meta", type=click.Path(exists=True), default=None,
              help="a dataset meta/ directory, to check normalisation and rename_map")
@click.option("--fail-on", type=click.Choice(["error", "warning", "never"]), default="error",
              show_default=True)
def doctor(train_config, dataset_meta, fail_on):
    """Preflight the stack for silent failures - the bugs that never raise."""
    from .doctor import run_all

    findings = run_all(train_config, dataset_meta)
    console.print()
    icon = {"error": "[red]ERROR  [/]", "warning": "[yellow]WARN   [/]", "info": "[dim]ok     [/]"}
    for f in findings:
        console.print(f"{icon[f.severity]} [bold]{f.rule_id}[/]  {f.message}")
        for line in _wrap(f.detail, 78):
            console.print(f"          [dim]{line}[/]")
        if f.fix:
            console.print(f"          [cyan]fix:[/] {f.fix}")
        if f.citation:
            console.print(f"          [dim]seen in: {f.citation}[/]")

    errors = sum(f.severity == "error" for f in findings)
    warnings = sum(f.severity == "warning" for f in findings)
    console.print(f"\n{errors} error(s), {warnings} warning(s), "
                  f"{len(findings) - errors - warnings} ok")
    if fail_on == "error" and errors:
        raise SystemExit(2)
    if fail_on == "warning" and (errors or warnings):
        raise SystemExit(2)


# -------------------------------------------------------------------- lint
@main.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--train-config", type=click.Path(exists=True), default=None,
              help="check normalisation feasibility against what your policy requests")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "sarif"]), default="text",
              show_default=True)
@click.option("--fail-on", type=click.Choice(["error", "warning", "never"]), default="error",
              show_default=True)
def lint(dataset, train_config, fmt, fail_on):
    """Check a LeRobot dataset for the silent misconfigurations that cost GPU-hours.

    Never imports lerobot: it reads meta/*.json and the parquet files directly, so it still
    works when the training stack itself is broken.
    """
    from .lint import run_all, to_sarif

    cfg = None
    if train_config:
        cfg = json.loads(Path(train_config).read_text(encoding="utf-8"))
    findings = run_all(dataset, cfg)

    if fmt == "sarif":
        click.echo(json.dumps(to_sarif(findings, __version__), indent=2))
    elif fmt == "json":
        click.echo(json.dumps([f.model_dump() for f in findings], indent=2))
    else:
        console.print()
        console.print(f"[bold]{dataset}[/]")
        icon = {"error": "[red]ERROR[/]", "warning": "[yellow]WARN [/]", "info": "[dim]ok   [/]"}
        for f in findings:
            console.print(f"{icon[f.severity]} [bold]{f.rule_id}[/]  {f.message}")
            for line in _wrap(f.detail, 76):
                console.print(f"        [dim]{line}[/]")
            if f.fix:
                console.print(f"        [cyan]fix:[/] {f.fix}")
            if f.citation:
                console.print(f"        [dim]{f.citation}[/]")
        errors = sum(f.severity == "error" for f in findings)
        warnings = sum(f.severity == "warning" for f in findings)
        console.print(f"\n{errors} error(s), {warnings} warning(s), "
                      f"{len(findings) - errors - warnings} passed")

    errors = sum(f.severity == "error" for f in findings)
    warnings = sum(f.severity == "warning" for f in findings)
    if fail_on == "error" and errors:
        raise SystemExit(2)
    if fail_on == "warning" and (errors or warnings):
        raise SystemExit(2)


# ---------------------------------------------------------------- manifest
@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--policy-id", default=None, help="name this arm as it appears in eval output")
@click.option("--out", type=click.Path(), default=None, help="default: <run_dir>/manifest.json")
def manifest(run_dir, policy_id, out):
    """Capture run provenance, including the samples_seen that decides comparability."""
    from .manifest import capture

    m = capture(run_dir, policy_id)
    target = Path(out) if out else Path(run_dir) / "manifest.json"
    target.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]wrote[/] {target}")
    console.print(f"  policy_id     {m.policy_id}")
    console.print(f"  batch x steps {m.batch_size} x {m.steps}")
    console.print(f"  [bold]samples_seen  {m.samples_seen:.3g}[/]" if m.samples_seen
                  else "  [yellow]samples_seen  unknown (batch_size or steps missing)[/]")


@main.command()
@click.argument("manifest_a", type=click.Path(exists=True))
@click.argument("manifest_b", type=click.Path(exists=True))
def diff(manifest_a, manifest_b):
    """Are these two runs comparable at all? Arithmetic, not opinion."""
    from .manifest import comparable
    from .manifest import diff as diff_fn

    a = RunManifest.model_validate_json(Path(manifest_a).read_text(encoding="utf-8"))
    b = RunManifest.model_validate_json(Path(manifest_b).read_text(encoding="utf-8"))
    rows = diff_fn(a, b)

    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("field", no_wrap=True)
    table.add_column(a.policy_id, overflow="fold")
    table.add_column(b.policy_id, overflow="fold")
    table.add_column("class", no_wrap=True)
    style = {"ok": "dim", "EXPECTED": "dim", "CAUSE": "yellow",
             "COMPUTE_CONFOUND": "magenta", "DATA_CONFOUND": "magenta", "unknown": "yellow"}
    for r in rows:
        note = f"  {r['note']}" if r.get("note") else ""
        table.add_row(r["field"], r["a"], r["b"],
                      f"[{style.get(r['class'], '')}]{r['class']}{note}[/]")
    console.print()
    console.print(table)

    # spell out exactly what differs: a flagged row the reader cannot decipher is useless
    flagged = [r for r in rows if r["class"] not in ("ok", "EXPECTED")]
    if flagged:
        console.print("\n[bold]what differs[/]")
        for r in flagged:
            console.print(f"  [bold]{r['field']}[/]")
            for token in _field_delta(r["a"], r["b"]):
                console.print(f"    {token}")

    if comparable(rows):
        console.print("\n[green]these runs are comparable[/]")
        raise SystemExit(0)
    console.print(f"\n[magenta]=> {a.policy_id} and {b.policy_id} are NOT comparable "
                  "as an architecture result[/]")
    console.print("[dim]the flagged difference is a sufficient alternative explanation for "
                  "any gap you measure between them.[/]")
    raise SystemExit(3)


# ------------------------------------------------------------------ report
@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--baseline", default=None)
@click.option("--adapter", type=click.Choice(available()), default=None)
@click.option("--manifests", multiple=True)
@click.option("--dataset", type=click.Path(exists=True), default=None,
              help="also lint this dataset and include the findings")
@click.option("--test", type=click.Choice(["fisher", "barnard", "boschloo"]), default="fisher",
              show_default=True)
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("-o", "--out", type=click.Path(), default="report.html", show_default=True)
@click.option("--modelcard", type=click.Path(), default=None,
              help="also write a LeRobot-format model card")
@click.option("--task", default="unknown", help="task name for the model card")
def report(paths, baseline, adapter, manifests, dataset, test, alpha, out, modelcard, task):
    """One self-contained HTML file you can hand to your lead, plus a model card."""
    from .lint import run_all as lint_all
    from .report import render_html, render_model_card

    rollouts = _load(paths, adapter, None)
    mans: dict[str, RunManifest] = {}
    for pattern in manifests:
        for hit in glob.glob(pattern) or [pattern]:
            m = RunManifest.model_validate_json(Path(hit).read_text(encoding="utf-8"))
            mans[m.policy_id] = m

    result = run_compare(rollouts, baseline, manifests=mans, test=test, alpha=alpha)
    posteriors = posterior_table(result.arms, baseline) if baseline else {}
    findings = lint_all(dataset) if dataset else []

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        render_html(result, baseline=baseline, posteriors=posteriors, lint_findings=findings),
        encoding="utf-8")
    console.print(f"[green]wrote[/] {out}  [dim]self-contained, no external assets[/]")

    if modelcard:
        Path(modelcard).write_text(
            render_model_card(result, task=task,
                              eval_command=f"verdikt compare {' '.join(paths)}"),
            encoding="utf-8")
        console.print(f"[green]wrote[/] {modelcard}")

    _echo_verdict(result.verdict, result.reason)
    raise SystemExit(int(result.verdict))


def _field_delta(a: str, b: str) -> list[str]:
    """Show only the parts that actually differ, for comma-separated composite fields."""
    if "=" in a and "=" in b and "," in (a + b):
        da = dict(p.split("=", 1) for p in a.split(",") if "=" in p)
        db = dict(p.split("=", 1) for p in b.split(",") if "=" in p)
        out = []
        for key in sorted(set(da) | set(db)):
            va, vb = da.get(key, "-"), db.get(key, "-")
            if va != vb:
                out.append(f"[yellow]{key}[/]: {va}  ->  {vb}")
        if out:
            return out
    return [f"{a}  ->  {b}"]


def _print_markdown(result, baseline) -> None:
    click.echo("| policy | n | success | 95% CI | grp |")
    click.echo("|---|---|---|---|---|")
    for a in result.arms:
        click.echo(f"| `{a.policy_id}` | {a.n} | {a.successes}/{a.n} ({a.rate:.1%}) | "
                   f"[{a.ci_low:.1%}, {a.ci_high:.1%}] | {a.letter} |")
    name = VERDICT[int(result.verdict)][0]
    click.echo(f"\n**VERDICT: {name}** - {result.reason}")


if __name__ == "__main__":
    main()
