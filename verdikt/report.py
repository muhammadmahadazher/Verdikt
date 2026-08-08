"""Self-contained HTML report and a LeRobot-format model card.

One file, no CDN, no external fonts, no JavaScript dependencies - it must open from a USB
stick on a machine with no network, because that is how eval results actually get shared.

Every metric carries a provenance tag. An engineer's trust dies the first time an uncalibrated
number is spoken in the same voice as a validated one, so `[validated]`, `[prior-art]` and
`[provisional]` are rendered inline rather than buried in documentation.
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone

from jinja2 import Template

from . import __version__, theme
from .schema import ComparisonResult, Finding, Verdict


def forest_svg(result: ComparisonResult) -> str:
    """Interval plot as inline SVG. Overlapping bars are the whole point of the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(theme.mpl_rc()):
        n = len(result.arms)
        fig, ax = plt.subplots(figsize=(9, max(2.0, 0.62 * n + 1.0)))
        for i, arm in enumerate(result.arms):
            y = n - i
            colour = theme.SERIES[i % len(theme.SERIES)]
            ax.plot([arm.ci_low * 100, arm.ci_high * 100], [y, y], color=colour, lw=3.2,
                    solid_capstyle="round")
            ax.scatter([arm.rate * 100], [y], color=colour, s=64, zorder=5)
            ax.text(-2, y, arm.policy_id, ha="right", va="center", color=theme.TEXT, fontsize=10)
            label = f"{arm.successes}/{arm.n}"
            if arm.one_sided_bound is not None:
                sign = "<=" if arm.successes == 0 else ">="
                label += f"  ({sign} {arm.one_sided_bound * 100:.1f}% one-sided)"
            ax.text(103, y, label, ha="left", va="center", color=theme.TEXT_DIM, fontsize=8.5)
        ax.set_xlim(-42, 132)
        ax.set_ylim(0.3, n + 0.7)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(f"success rate (%), {result.arms[0].ci_method} 95% interval")
        ax.spines["left"].set_visible(False)
        fig.tight_layout()
        buf = io.StringIO()
        fig.savefig(buf, format="svg", transparent=True)
        plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


_TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>verdikt report - {{ generated }}</title>
<style>
  :root {
    --bg:{{ t.BG }}; --surface:{{ t.SURFACE }}; --surface2:{{ t.SURFACE_2 }};
    --line:{{ t.HAIRLINE }}; --text:{{ t.TEXT }}; --dim:{{ t.TEXT_DIM }};
    --faint:{{ t.TEXT_FAINT }}; --accent:{{ t.ACCENT }}; --verdict:{{ verdict_colour }};
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:14px/1.65 {{ t.FONT_STACK }};padding:40px 20px}
  .wrap{max-width:940px;margin:0 auto}
  header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px}
  h1{font-size:20px;font-weight:500;margin:0 0 4px;letter-spacing:.02em}
  .sub{color:var(--faint);font-size:12px}
  .banner{border:1px solid var(--verdict);border-left:3px solid var(--verdict);
          background:color-mix(in srgb, var(--verdict) 10%, transparent);
          padding:16px 20px;margin:24px 0}
  .banner .state{color:var(--verdict);font-size:16px;font-weight:500;letter-spacing:.08em}
  .banner .reason{color:var(--dim);margin-top:6px;font-size:13px}
  h2{font-size:13px;font-weight:500;color:var(--faint);text-transform:uppercase;
     letter-spacing:.14em;margin:34px 0 12px;border-bottom:1px solid var(--line);
     padding-bottom:8px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;color:var(--faint);font-weight:500;padding:8px 10px;
     background:var(--surface2);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
  td{padding:9px 10px;border-bottom:1px solid var(--line)}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  .grp{display:inline-block;min-width:22px;text-align:center;padding:1px 6px;
       border:1px solid var(--line);color:var(--dim);font-size:11px}
  .note{background:var(--surface);border-left:2px solid {{ t.WARN }};padding:12px 16px;
        margin:14px 0;color:var(--dim);font-size:13px}
  .confound{background:var(--surface);border-left:2px solid {{ t.CONFOUND }};
            padding:12px 16px;margin:12px 0;font-size:13px}
  .confound b{color:{{ t.CONFOUND }};font-weight:500}
  .tag{font-size:10px;letter-spacing:.06em;padding:1px 6px;border:1px solid currentColor;
       margin-left:8px;vertical-align:1px}
  .validated{color:{{ t.OK }}} .provisional{color:{{ t.WARN }}} .priorart{color:{{ t.ACCENT }}}
  .sig{color:{{ t.OK }}} .nosig{color:var(--faint)} .suppressed{color:{{ t.CONFOUND }}}
  svg{max-width:100%;height:auto;display:block;margin:8px 0}
  footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);
         color:var(--faint);font-size:11px}
  code{background:var(--surface2);padding:1px 5px;font-size:12px}
</style></head><body><div class="wrap">

<header>
  <h1>verdikt report</h1>
  <div class="sub">{{ generated }} &nbsp;·&nbsp; verdikt {{ version }}
  {%- if label_sources %} &nbsp;·&nbsp; labels: {{ label_sources|join(", ") }}{% endif %}</div>
</header>

<div class="banner">
  <div class="state">{{ verdict_name }} &nbsp;<span style="color:var(--faint);font-size:12px">
    exit {{ verdict_code }}</span></div>
  <div class="reason">{{ reason }}</div>
</div>

<h2>Arms <span class="tag validated">validated</span></h2>
<table>
  <tr><th>policy</th><th>n</th><th>successes</th><th>rate</th>
      <th>95% CI ({{ ci_method }})</th><th>grp</th><th>samples seen</th></tr>
  {% for a in arms %}
  <tr>
    <td>{{ a.policy_id }}</td>
    <td class="num">{{ a.n }}</td>
    <td class="num">{{ a.successes }}</td>
    <td class="num">{{ "%.1f"|format(a.rate * 100) }}%</td>
    <td class="num">[{{ "%.1f"|format(a.ci_low * 100) }}, {{ "%.1f"|format(a.ci_high * 100) }}]</td>
    <td><span class="grp">{{ a.letter or "-" }}</span></td>
    <td class="num">{{ "%.2e"|format(a.samples_seen) if a.samples_seen else "-" }}</td>
  </tr>
  {% endfor %}
</table>
<div class="sub" style="margin-top:8px">arms sharing a group letter are not distinguishable at
this n</div>

{% for a in arms if a.one_sided_bound is not none %}
<div class="note">
  <b>{{ a.successes }}/{{ a.n }} does not mean {{ "0%" if a.successes == 0 else "certainty" }}.</b>
  exact one-sided 95% bound:
  {{ "<=" if a.successes == 0 else ">=" }} {{ "%.1f"|format(a.one_sided_bound * 100) }}%
</div>
{% endfor %}

{{ forest }}

<h2>Pairwise tests <span class="tag validated">validated</span></h2>
<table>
  <tr><th>comparison</th><th>test</th><th>p</th><th>corrected alpha</th><th>result</th></tr>
  {% for p in pairs %}
  <tr>
    <td>{{ p.a }} vs {{ p.b }}</td>
    <td>{{ p.test }}</td>
    <td class="num">{{ "-" if p.suppressed_reason else "%.4g"|format(p.p_value) }}</td>
    <td class="num">{{ "%.5f"|format(p.alpha_adjusted) }}</td>
    <td>
      {%- if p.suppressed_reason %}<span class="suppressed">SUPPRESSED</span>
      {%- elif p.significant %}<span class="sig">significant</span>
      {%- else %}<span class="nosig">not significant</span>{% endif -%}
      {%- if p.alt_p_value is not none and not p.suppressed_reason
             and ((p.alt_p_value <= p.alpha_adjusted) != p.significant) %}
      <span style="color:{{ t.WARN }}"> &nbsp;{{ p.alt_test }} would disagree
      (p={{ "%.4g"|format(p.alt_p_value) }})</span>{% endif -%}
    </td>
  </tr>
  {% endfor %}
</table>

{% if posteriors %}
<h2>Posterior probabilities <span class="tag validated">validated</span></h2>
<table>
  <tr><th>statement</th><th>probability</th></tr>
  {% for name, prob in posteriors.items() %}
  <tr><td>P({{ name }} &gt; {{ baseline }})</td>
      <td class="num">{{ "%.3f"|format(prob) }}</td></tr>
  {% endfor %}
</table>
<div class="sub" style="margin-top:8px">uniform priors; answers the question people actually
ask when they compare two error bars</div>
{% endif %}

{% if confounds %}
<h2>Confounds</h2>
{% for c in confounds %}
<div class="confound"><b>{{ c.kind }}</b> &nbsp; <code>{{ c.field }}</code>:
  {{ c.a_value }} vs {{ c.b_value }}{% if c.ratio %} ({{ "%.1f"|format(c.ratio) }}x){% endif %}
  <div style="color:var(--dim);margin-top:6px">{{ c.message }}</div>
</div>
{% endfor %}
{% endif %}

{% if lint_findings %}
<h2>Dataset findings</h2>
<table>
  <tr><th>rule</th><th>severity</th><th>message</th></tr>
  {% for f in lint_findings %}
  <tr><td>{{ f.rule_id }}</td>
      <td style="color:{{ t.BAD if f.severity == 'error' else t.WARN }}">{{ f.severity }}</td>
      <td>{{ f.message }}</td></tr>
  {% endfor %}
</table>
{% endif %}

<footer>
  generated by verdikt {{ version }} — statistics are computed with scipy/statsmodels
  reference implementations; power is computed by exact enumeration through the test that
  issued this verdict. this report is self-contained: no network requests, no external assets.
</footer>
</div></body></html>""")


def render_html(result: ComparisonResult, *, baseline: str | None = None,
                posteriors: dict[str, float] | None = None,
                lint_findings: list[Finding] | None = None) -> str:
    name, colour, _style = theme.VERDICT[int(result.verdict)]
    return _TEMPLATE.render(
        t=theme,
        version=__version__,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        verdict_name=name,
        verdict_code=int(result.verdict),
        verdict_colour=colour,
        reason=result.reason,
        arms=result.arms,
        pairs=result.pairs,
        confounds=_dedupe(result.confounds),
        ci_method=result.arms[0].ci_method if result.arms else "wilson",
        forest=forest_svg(result) if result.arms else "",
        posteriors=posteriors or {},
        baseline=baseline,
        lint_findings=[f for f in (lint_findings or []) if f.severity != "info"],
        label_sources=result.label_sources,
    )


def render_model_card(result: ComparisonResult, *, task: str = "unknown",
                      eval_command: str = "", hardware: str = "") -> str:
    """Markdown model card in the shape LeRobot's reporting guidance asks for."""
    lines = [
        "# Model card - evaluation results",
        "",
        f"_Generated by [verdikt](https://github.com/muhammadmahadazher/Verdikt) "
        f"{__version__} on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}._",
        "",
        "## Results",
        "",
        "| policy | task | n_episodes | success rate | 95% CI | group |",
        "|---|---|---:|---:|---|:---:|",
    ]
    for a in result.arms:
        ci = f"[{a.ci_low:.1%}, {a.ci_high:.1%}]"
        lines.append(f"| `{a.policy_id}` | {task} | {a.n} | {a.successes}/{a.n} "
                     f"({a.rate:.1%}) | {ci} | {a.letter or '-'} |")

    lines += ["", f"**Verdict: {theme.VERDICT[int(result.verdict)][0]}** - {result.reason}", ""]

    bounded = [a for a in result.arms if a.one_sided_bound is not None]
    if bounded:
        lines.append("### Exact bounds")
        lines.append("")
        for a in bounded:
            sign = "below" if a.successes == 0 else "above"
            lines.append(f"- `{a.policy_id}` scored {a.successes}/{a.n}; the true rate is "
                         f"{sign} **{a.one_sided_bound:.1%}** at 95% one-sided confidence.")
        lines.append("")

    if result.confounds:
        lines += ["### Comparability", ""]
        for c in _dedupe(result.confounds):
            lines.append(f"- **{c.kind}** on `{c.field}`: {c.message}")
        lines.append("")

    if result.verdict == Verdict.UNDERPOWERED and result.required_n:
        lines += [f"> This evaluation is underpowered. Resolving a difference of the observed "
                  f"size needs about **{result.required_n} episodes per arm**.", ""]

    if eval_command:
        lines += ["### Reproduce", "", "```bash", eval_command, "```", ""]
    if hardware:
        lines += [f"**Hardware:** {hardware}", ""]

    lines += ["### Statistical method", "",
              "- Intervals: "
              f"{result.arms[0].ci_method if result.arms else 'wilson'} (Wald not used)",
              f"- Tests: {result.pairs[0].test if result.pairs else 'n/a'}, "
              "multiplicity-corrected",
              "- Power and required-N: exact enumeration through the test that issued the "
              "verdict, not the normal approximation",
              ""]
    return "\n".join(lines)


def _dedupe(confounds):
    seen, out = set(), []
    for c in confounds:
        key = (c.field, c.a_value, c.b_value)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _esc(s: str) -> str:
    return html.escape(str(s))
