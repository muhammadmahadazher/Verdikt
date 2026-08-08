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


def _expand_braces(pattern: str) -> list[str]:
    """Expand shell-style {a,b} alternatives.

    A quoted pattern never reaches the shell, so `"runs/{act,upstream}.json"` arrives here
    literally and Python's glob - which has no brace syntax - silently matches nothing. Users
    reasonably expect it to work because it does when unquoted, so we expand it ourselves.
    """
    start = pattern.find("{")
    if start == -1:
        return [pattern]
    depth = 0
    for i in range(start, len(pattern)):
        if pattern[i] == "{":
            depth += 1
        elif pattern[i] == "}":
            depth -= 1
            if depth == 0:
                head, body, tail = pattern[:start], pattern[start + 1:i], pattern[i + 1:]
                parts, level, current = [], 0, ""
                for ch in body:
                    if ch == "," and level == 0:
                        parts.append(current)
                        current = ""
                        continue
                    level += (ch == "{") - (ch == "}")
                    current += ch
                parts.append(current)
                out = []
                for part in parts:
                    out.extend(_expand_braces(f"{head}{part}{tail}"))
                return out
    return [pattern]  # unbalanced brace: leave it alone rather than guess


def _load(paths: tuple[str, ...], adapter: str | None, policy_id: str | None,
          mapping: dict[str, str] | None = None) -> list[Rollout]:
    expanded: list[Path] = []
    for raw in paths:
        hits: list[Path] = []
        for p in _expand_braces(raw):
            hits.extend(Path(h) for h in glob.glob(p))
            if not hits and Path(p).exists():
                hits.append(Path(p))
        if not hits:
            raise click.ClickException(
                f"no files matched {raw!r}. note that brace patterns are expanded by verdikt, "
                "so check the paths themselves exist."
            )
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
@click.option("--paired", is_flag=True,
              help="McNemar on episodes that are the same scene in both arms; removes scene "
                   "difficulty from the comparison and needs far fewer episodes")
@click.option("--assume-aligned", is_flag=True,
              help="with --paired, confirm both arms were evaluated over the same scenes when "
                   "the source records no per-episode seed")
@click.option("--min-lower-bound", type=float, default=None,
              help="gate on the CI LOWER bound (there is deliberately no --min-success)")
@click.option("--noninferiority", "noninf", type=float, default=None, metavar="MARGIN",
              help="pass if the candidate is no worse than baseline by more than MARGIN")
@click.option("--plan", "plan_path", type=click.Path(exists=True), default=None,
              help="verify the pre-registered analysis commitment")
@click.option("--format", "fmt", type=click.Choice(["text", "json", "markdown"]),
              default="text", show_default=True)
def compare(paths, baseline, adapter, manifests, test, alpha, correction, ci_method,
            paired, assume_aligned, min_lower_bound, noninf, plan_path, fmt):
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

    from .compare import PairingError

    try:
        result = run_compare(
            rollouts, baseline, manifests=mans, test=test, alpha=alpha,
            correction=("none" if correction == "none" else correction),
            ci_method=ci_method, min_lower_bound=min_lower_bound,
            noninferiority_margin=noninf, plan=plan_obj,
            paired=paired, allow_index_pairing=assume_aligned,
        )
    except PairingError as exc:
        raise click.ClickException(str(exc)) from exc

    if paired and assume_aligned and fmt == "text":
        console.print("\n[yellow]assumed[/] episodes with the same index are the same scene. "
                      "that holds only if\n        both evaluations ran with the same --seed "
                      "AND the same batch size.")

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


# ----------------------------------------------------------------- profile
@main.command()
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--experimental", is_flag=True,
              help="required: this metric is provisional and reports a bound, not a prediction")
@click.option("--k", type=int, default=24, show_default=True, help="neighbours per anchor")
@click.option("--block-radius", type=int, default=15, show_default=True,
              help="frames of the same episode excluded from a neighbourhood")
@click.option("--sample", type=int, default=600, show_default=True,
              help="anchors sampled (cost is roughly linear in this)")
@click.option("--permutations", type=int, default=99, show_default=True)
def profile(dataset, experimental, k, block_radius, sample, permutations):
    """[EXPERIMENTAL] How much action variance can a deterministic policy not explain?

    Reports a BOUND under the embeddings tested - never a success-rate prediction and never
    an architecture recommendation. Requires agreement across at least two embeddings, and
    refuses to answer when they disagree.
    """
    if not experimental:
        raise click.ClickException(
            "verdikt profile is provisional and must be run with --experimental.\n"
            "it reports a bound under the embedding you give it, not a property of the "
            "dataset, and it has not been validated against downstream policy performance.\n"
            "false-positive calibration: docs/calibrate_profile.py"
        )

    import numpy as np
    import pyarrow.parquet as pq

    from .lint import load as load_dataset
    from .profile import profile as run_profile

    view = load_dataset(dataset)
    if view.errors or not view.data_files:
        raise click.ClickException(f"cannot read dataset: {'; '.join(view.errors) or 'no data'}")

    try:
        import pyarrow as pa

        cols = ["observation.state", "action", "episode_index"]
        parts = [pq.read_table(p, columns=cols) for p in view.data_files]
        table = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
    except Exception as exc:
        raise click.ClickException(f"dataset lacks observation.state/action: {exc}") from exc

    def matrix(col):
        vals = table[col].to_pylist()
        arr = np.asarray(vals, dtype=float)
        return arr if arr.ndim == 2 else arr.reshape(-1, 1)

    state, actions = matrix("observation.state"), matrix("action")
    episodes = np.asarray(table["episode_index"])

    # Two genuinely different views of the same frames. If the bound depends on which one you
    # look through, that is a fact about the embedding, not the dataset - and verdikt says so.
    velocity = np.vstack([np.zeros((1, state.shape[1])), np.diff(state, axis=0)])
    velocity[np.concatenate([[True], episodes[1:] != episodes[:-1]])] = 0.0
    embeddings = {
        "observation.state": state,
        "state + velocity": np.hstack([state, velocity]),
    }

    console.print()
    console.print(f"[bold]{dataset}[/]")
    console.print(f"[dim]{len(state)} frames · k={k} · block radius {block_radius} frames · "
                  f"{permutations} permutations[/]")
    with console.status("profiling embeddings..."):
        results, verdict, explanation = run_profile(
            embeddings, actions, episodes, k=k, block_radius=block_radius,
            sample=sample, permutations=permutations)

    table_out = Table(box=None, pad_edge=False, header_style="bold")
    table_out.add_column("embedding")
    table_out.add_column("anchors", justify="right")
    table_out.add_column("AMR (L2)", justify="right")
    table_out.add_column("MAD (L1)", justify="right")
    table_out.add_column("multimodal", justify="right")
    table_out.add_column("eff. dim", justify="right")
    for r in results:
        table_out.add_row(r.embedding_name, str(r.n_samples), f"{r.amr_l2:.3f}",
                          f"{r.mad_l1:.3f}", f"{r.multimodal_fraction:.1%}",
                          f"{r.participation_ratio:.1f}")
    console.print()
    console.print(table_out)
    console.print("[dim]AMR bounds an L2 objective; MAD bounds an L1 objective (what ACT "
                  "minimises). they are not interchangeable.[/]")

    for r in results:
        for note in r.notes:
            console.print(f"  [yellow]{r.embedding_name}[/]: {note}")

    style = "yellow" if verdict == "INSUFFICIENT EVIDENCE" else "cyan"
    console.print(f"\n[{style}]{verdict}[/]")
    for line in _wrap(explanation, 84):
        console.print(f"  [dim]{line}[/]")
    console.print("\n[dim]\\[provisional] this metric is not calibrated against downstream "
                  "policy success. it is a bound under the embeddings shown, nothing more.[/]")


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


# ------------------------------------------------------------------- watch
@main.command()
@click.argument("paths", nargs=-1, required=True)
@click.option("--adapter", type=click.Choice(available()), default=None)
@click.option("--alpha", type=float, default=0.05, show_default=True)
@click.option("--partial", is_flag=True,
              help="wager on partial credit (progress/coverage) instead of binary success")
@click.option("--replay", is_flag=True,
              help="replay the recorded episodes in many random orders and report the "
                   "measured saving instead of a single run")
@click.option("--trials", type=int, default=2000, show_default=True)
def watch(paths, adapter, alpha, partial, replay, trials):
    """Stop an evaluation the moment the answer is in - valid no matter how often you look.

    A fixed-sample test is only valid at the n you committed to; peeking until p drops below
    0.05 inflates false positives badly. This uses a test martingale instead, so Ville's
    inequality bounds the error at alpha across every possible stopping time.
    """
    from .sequential import replay_savings, run

    rollouts = _load(paths, adapter, None)
    by_policy: dict[str, list] = {}
    for r in sorted(rollouts, key=lambda r: r.episode_idx):
        by_policy.setdefault(r.policy_id, []).append(r)
    if len(by_policy) != 2:
        raise click.ClickException(
            f"watch compares exactly two arms; found {len(by_policy)}: {sorted(by_policy)}")

    (name_a, rs_a), (name_b, rs_b) = sorted(by_policy.items())

    def outcomes(rs):
        vals = []
        for r in rs:
            v = r.progress if partial else (1.0 if r.success else 0.0)
            if v is None:
                continue
            vals.append(float(v))
        return vals

    a, b = outcomes(rs_a), outcomes(rs_b)
    if not a or not b:
        raise click.ClickException(
            "no usable outcomes: --partial needs a progress column, and the default needs "
            "success labels")

    signal = "partial credit (progress)" if partial else "binary success"
    console.print()
    console.print(f"[bold]sequential test[/]  {name_a} vs {name_b}")
    console.print(f"[dim]signal: {signal} · alpha={alpha} · capital threshold "
                  f"{1 / alpha:.0f}x[/]")
    if partial:
        console.print("[yellow]note[/] partial credit answers a different question than "
                      "success:\n      two policies can differ in coverage while their "
                      "success rates do not.")

    if replay:
        res = replay_savings(a, b, alpha=alpha, trials=trials)
        console.print(f"\n[bold]replayed {res['trials']} random orderings of "
                      f"{res['n_available']} episodes[/]")
        console.print(f"  reached a verdict in   [bold]{res['stop_rate']:.0%}[/] of orderings")
        if res["stop_rate"] > 0:
            console.print(f"  median stopping point  [bold cyan]{res['median_stop']}[/] episodes "
                          f"(90th percentile {res['p90_stop']})")
            console.print(f"  median saving          [bold green]"
                          f"{res['median_saving']:.0%}[/] of the episodes you ran")
        else:
            console.print("  [dim]never crossed the threshold: at this effect size the "
                          "sequential test correctly declines to reject.[/]")
        raise SystemExit(0)

    state = run(a, b, alpha=alpha)
    console.print()
    if state.stopped_at:
        saving = 1 - state.stopped_at / min(len(a), len(b))
        console.print(f"[green]STOP[/]  a verdict was available after "
                      f"[bold]{state.stopped_at}[/] episodes per arm")
        console.print(f"      {state.evidence}")
        console.print(f"      [dim]you ran {min(len(a), len(b))}; "
                      f"{saving:.0%} of them were not needed[/]")
        raise SystemExit(0)

    console.print(f"[yellow]CONTINUE[/]  no verdict after {state.steps} episodes per arm")
    console.print(f"          {state.evidence}")
    console.print(f"          [dim]peak capital reached {state.peak:.2f}x. this is not "
                  "evidence of equivalence - it is an absence of evidence either way.[/]")
    raise SystemExit(2)


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
@click.option("--wandb", "wandb_run", default=None, metavar="ENTITY/PROJECT/RUN_ID",
              help="attach the verdict, arm table and report to an existing W&B run")
@click.option("--wandb-dry-run", is_flag=True,
              help="print exactly what would be sent to W&B, and send nothing")
def report(paths, baseline, adapter, manifests, dataset, test, alpha, out, modelcard, task,
           wandb_run, wandb_dry_run):
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

    if wandb_run or wandb_dry_run:
        from .integrations.wandb import build_payload, build_table_rows, parse_run_path

        payload = build_payload(result, baseline, posteriors)
        if wandb_dry_run:
            target = wandb_run or "<no run given>"
            console.print(f"\n[bold]W&B dry run[/] [dim]-> {target}[/]")
            console.print(f"[dim]{len(payload)} summary keys, "
                          f"{len(build_table_rows(result)[1])} table rows, "
                          f"{'report + ' if out else ''}"
                          f"{'model card' if modelcard else 'no model card'} as artifact[/]")
            for k, v in sorted(payload.items()):
                console.print(f"  [cyan]{k}[/] = {v}")
        else:
            try:
                parse_run_path(wandb_run)
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            from .integrations.wandb import push

            try:
                push(result, wandb_run, baseline=baseline, posteriors=posteriors,
                     html_report=out, model_card=modelcard)
            except RuntimeError as exc:
                raise click.ClickException(str(exc)) from exc
            console.print(f"[green]pushed to W&B[/] {wandb_run}  [dim]{len(payload)} summary "
                          "keys + arm table + report artifact[/]")

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
