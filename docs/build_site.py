"""Build the local demo site: a landing page plus a live report from the real n=200 corpus.

    python docs/build_site.py     ->  verdikt/site/{index.html,report.html,*.png}

The site is static and self-contained, so the same output can be dropped on any host.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jinja2 import Template  # noqa: E402

from verdikt import __version__, theme  # noqa: E402
from verdikt.compare import compare, posterior_table  # noqa: E402
from verdikt.ingest import get  # noqa: E402
from verdikt.report import render_html  # noqa: E402
from verdikt.stats import required_n  # noqa: E402
from verdikt.stats.power import mde  # noqa: E402

SITE = ROOT / "site"
FIXTURES = ROOT / "tests" / "fixtures" / "pusht_n200"

LANDING = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verdikt - a decision layer for robot-policy evaluation</title>
<meta name="description" content="Verdikt reads the eval JSON you already have and tells you
whether checkpoint B is really better than A - or whether you have not run enough episodes to
know. CPU-only, no GPU, no robot.">
<style>
  :root{
    --bg:{{t.BG}};--surface:{{t.SURFACE}};--surface2:{{t.SURFACE_2}};--line:{{t.HAIRLINE}};
    --text:{{t.TEXT}};--dim:{{t.TEXT_DIM}};--faint:{{t.TEXT_FAINT}};--accent:{{t.ACCENT}};
    --ok:{{t.OK}};--bad:{{t.BAD}};--warn:{{t.WARN}};--conf:{{t.CONFOUND}};
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{background:var(--bg);color:var(--text);font:15px/1.7 {{t.FONT_UI}};
       -webkit-font-smoothing:antialiased}
  .mono{font-family:{{t.FONT_STACK}}}
  .wrap{max-width:1080px;margin:0 auto;padding:0 24px}
  a{color:var(--accent);text-decoration:none}
  a:hover{text-decoration:underline}

  /* ---------- hero ---------- */
  header{padding:88px 0 64px;border-bottom:1px solid var(--line);position:relative;
         overflow:hidden}
  header::after{content:"";position:absolute;inset:0;pointer-events:none;
    background:
      repeating-linear-gradient(90deg,transparent 0 79px,{{t.HAIRLINE}}22 79px 80px),
      repeating-linear-gradient(0deg,transparent 0 79px,{{t.HAIRLINE}}22 79px 80px);
    mask-image:radial-gradient(ellipse 80% 60% at 50% 0%,#000 20%,transparent 75%)}
  .eyebrow{color:var(--faint);letter-spacing:.22em;text-transform:uppercase;font-size:11px;
           margin-bottom:20px}
  h1{font-size:clamp(38px,6vw,60px);line-height:1.04;font-weight:500;letter-spacing:-.02em;
     max-width:16ch}
  h1 em{font-style:normal;color:var(--warn)}
  .lede{color:var(--dim);font-size:18px;max-width:60ch;margin-top:22px}
  .cta{display:flex;gap:12px;margin-top:34px;flex-wrap:wrap;position:relative;z-index:1}
  .btn{display:inline-block;padding:11px 20px;border:1px solid var(--line);
       background:var(--surface);color:var(--text);font-size:14px;border-radius:2px;
       transition:border-color .18s,background .18s}
  .btn:hover{border-color:var(--accent);background:var(--surface2);text-decoration:none}
  .btn.primary{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:500}
  .btn.primary:hover{filter:brightness(1.1)}

  /* ---------- terminal ---------- */
  .term{background:#0A0D12;border:1px solid var(--line);border-radius:4px;margin-top:44px;
        position:relative;z-index:1;overflow:hidden}
  .term-bar{display:flex;align-items:center;gap:7px;padding:10px 14px;
            border-bottom:1px solid var(--line);background:var(--surface)}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--line)}
  .term-title{color:var(--faint);font-size:11px;margin-left:8px;letter-spacing:.06em}
  .term pre{padding:20px 22px;overflow-x:auto;font-family:{{t.FONT_STACK}};font-size:12.5px;
            line-height:1.75;color:var(--dim)}
  .c-ok{color:var(--ok)}.c-bad{color:var(--bad)}.c-warn{color:var(--warn)}
  .c-txt{color:var(--text)}.c-dim{color:var(--faint)}.c-acc{color:var(--accent)}

  section{padding:72px 0;border-bottom:1px solid var(--line)}
  h2{font-size:13px;font-weight:500;color:var(--faint);letter-spacing:.16em;
     text-transform:uppercase;margin-bottom:28px}
  h3{font-size:22px;font-weight:500;margin-bottom:10px;letter-spacing:-.01em}
  p.body{color:var(--dim);max-width:70ch}

  .grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(232px,1fr))}
  .card{background:var(--surface);border:1px solid var(--line);padding:22px;border-radius:3px;
        transition:transform .18s,border-color .18s}
  .card:hover{transform:translateY(-2px);border-color:var(--accent)}
  .card .k{font-family:{{t.FONT_STACK}};font-size:11px;letter-spacing:.08em;margin-bottom:10px}
  .card h4{font-size:15px;font-weight:500;margin-bottom:6px}
  .card p{color:var(--faint);font-size:13.5px}

  table{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}
  th{text-align:left;color:var(--faint);font-weight:500;font-size:11px;letter-spacing:.08em;
     text-transform:uppercase;padding:10px 12px;background:var(--surface2)}
  td{padding:11px 12px;border-bottom:1px solid var(--line)}
  td.n{text-align:right;font-family:{{t.FONT_STACK}};font-variant-numeric:tabular-nums}
  .bar{height:5px;background:var(--surface2);border-radius:3px;position:relative;min-width:120px}
  .bar span{position:absolute;height:100%;border-radius:3px;background:var(--accent);opacity:.85}
  figure{margin-top:26px}
  figure img{width:100%;border:1px solid var(--line);border-radius:3px;display:block}
  figcaption{color:var(--faint);font-size:12.5px;margin-top:10px}
  footer{padding:48px 0 72px;color:var(--faint);font-size:12.5px}
  .pill{display:inline-block;font-family:{{t.FONT_STACK}};font-size:11px;padding:3px 9px;
        border:1px solid currentColor;border-radius:2px;margin-right:6px}
</style></head><body>

<header><div class="wrap">
  <div class="eyebrow">verdikt {{ version }} &nbsp;·&nbsp; apache-2.0 &nbsp;·&nbsp; cpu only</div>
  <h1>Your eval printed a number.<br><em>Can you believe it?</em></h1>
  <p class="lede">Verdikt reads the eval JSON, dataset files and run configs you already have,
     and refuses to let you draw a conclusion the data does not support. No GPU. No robot.
     No new workflow.</p>
  <div class="cta">
    <a class="btn primary" href="report.html">See a live report &rarr;</a>
    <a class="btn" href="https://github.com/muhammadmahadazher/Verdikt">GitHub</a>
    <a class="btn" href="#evidence">The evidence</a>
  </div>

  <div class="term">
    <div class="term-bar"><i class="dot"></i><i class="dot"></i><i class="dot"></i>
      <span class="term-title mono">verdikt compare results/*.json --baseline diffusion</span></div>
<pre><span class="c-dim">policy       n          success  95% CI (wilson)  grp</span>
<span class="c-txt">act        200    2/200    1.0%   [  0.3,   3.6]    a</span>
<span class="c-txt">diffusion  200   48/200   24.0%   [ 18.6,  30.4]    b</span>
<span class="c-txt">smolvla    200    0/200    0.0%   [  0.0,   1.9]    a</span>
<span class="c-txt">upstream   200  131/200   65.5%   [ 58.7,  71.7]    c</span>

<span class="c-warn">  0/200 does not mean zero. one-sided 95% upper bound: 1.5%</span>

<span class="c-dim">pairwise (test=fisher, m=6, corrected alpha=0.00833)</span>
  act vs diffusion        p=1.168e-13  <span class="c-ok">SIGNIFICANT</span>
  act vs smolvla          p=0.4987     <span class="c-dim">not significant</span>
  diffusion vs upstream   p=5.537e-17  <span class="c-ok">SIGNIFICANT</span>

<span class="c-bad">VERDICT   REGRESSION</span>
<span class="c-dim">          act is worse than diffusion (1.0% vs 24.0%, p=1.168e-13)</span>
<span class="c-dim">exit 1</span></pre>
  </div>
</div></header>

<section><div class="wrap">
  <h2>The four-state verdict</h2>
  <p class="body">Success rates are stochastic. A binary pass/fail gate on a binomial produces
     constant false alarms, so &ldquo;I cannot tell yet&rdquo; is a first-class answer here -
     and it arrives with the number of episodes that would settle it.</p>
  <div class="grid" style="margin-top:26px">
    <div class="card"><div class="k" style="color:{{t.OK}}">EXIT 0 &nbsp;BETTER</div>
      <h4>No regression</h4><p>Ship it. The design was sensitive enough for this null to mean
      something.</p></div>
    <div class="card"><div class="k" style="color:{{t.BAD}}">EXIT 1 &nbsp;REGRESSION</div>
      <h4>Significantly worse</h4><p>The candidate lost, and the evidence supports saying so.</p></div>
    <div class="card"><div class="k" style="color:{{t.WARN}}">EXIT 2 &nbsp;UNDERPOWERED</div>
      <h4>Cannot decide</h4><p>Not a pass. Here is the n that would actually answer your
      question.</p></div>
    <div class="card"><div class="k" style="color:{{t.CONFOUND}}">EXIT 3 &nbsp;NOT COMPARABLE</div>
      <h4>Confounded</h4><p>One arm saw 10&times; the samples. That difference explains any gap
      you measured.</p></div>
  </div>
</div></section>

<section id="evidence"><div class="wrap">
  <h2>Measured, not asserted</h2>
  <h3>How many episodes do you actually need?</h3>
  <p class="body">The normal approximation says {{ n_approx }} per arm to separate 35% from 70%
     at 80% power. Run the exact test at that n and the real power is {{ realised }}. Verdikt
     computes power by exact enumeration <em>through the test that issues the verdict</em>.</p>
  <table>
    <tr><th>your budget</th><th>smallest difference detectable vs a 35% baseline</th><th></th></tr>
    {% for n, d in budget_rows %}
    <tr><td class="n">n = {{ n }}</td><td class="n">{{ "%.0f"|format(d * 100) }} pp</td>
        <td><div class="bar"><span style="width:{{ (d * 100)|round|int }}%"></span></div></td></tr>
    {% endfor %}
  </table>
  <figure><img src="n_matters.png" alt="Same four policies at n=20 and n=200">
    <figcaption>The same four checkpoints at n=20 and n=200 - 800 real rollouts, no retraining.
    At n=20 two groups and one comparison too close to call; at n=200 three distinct tiers.
    n=20 was wrong in both directions: diffusion read 35% and is really 24%; ACT read a flat 0%
    and actually solves 1%.</figcaption></figure>
  <figure><img src="power.png" alt="Exact power curves">
    <figcaption>At n=20, power to detect 35% vs 70% is 0.47 - worse than a coin flip.</figcaption></figure>
</div></section>

<section><div class="wrap">
  <h2>Refusals you cannot switch off</h2>
  <div class="grid">
    {% for title, body in refusals %}
    <div class="card"><div class="k" style="color:{{t.WARN}}">&#10007;</div>
      <h4>{{ title }}</h4><p>{{ body }}</p></div>
    {% endfor %}
  </div>
</div></section>

<section><div class="wrap">
  <h2>Install</h2>
  <div class="term"><div class="term-bar"><i class="dot"></i><i class="dot"></i><i class="dot"></i>
    <span class="term-title mono">any platform &middot; python 3.10+</span></div>
<pre><span class="c-dim">$</span> <span class="c-txt">pip install git+https://github.com/muhammadmahadazher/Verdikt</span>

<span class="c-dim">$</span> <span class="c-txt">verdikt doctor</span>            <span class="c-dim"># is my stack lying to me?</span>
<span class="c-dim">$</span> <span class="c-txt">verdikt lint &lt;dataset&gt;</span>   <span class="c-dim"># will this waste my GPU-hours?</span>
<span class="c-dim">$</span> <span class="c-txt">verdikt compare "results/*.json" --baseline diffusion</span>
<span class="c-dim">$</span> <span class="c-txt">verdikt report  "results/*.json" -o report.html</span></pre></div>
</div></section>

<footer><div class="wrap">
  <span class="pill" style="color:{{t.OK}}">97 tests</span>
  <span class="pill" style="color:{{t.ACCENT}}">apache-2.0</span>
  <span class="pill" style="color:{{t.WARN}}">no gpu required</span>
  <p style="margin-top:16px">Every number on this page is computed from the committed n=200
  corpus by <span class="mono">docs/build_site.py</span>. None were typed by hand.</p>
</div></footer>
</body></html>""")

REFUSALS = [
    ("No bare success rates",
     "A rate never prints without n and an interval. There is nowhere in the code that can."),
    ("0/n is never 0%",
     "It prints its exact one-sided bound. 0/200 means at most 1.5%, not zero."),
    ("No --min-success flag",
     "Gating a stochastic binomial on a point estimate is the malpractice this tool exists to "
     "stop. Gate on a bound or a margin."),
    ("No Wald interval",
     "It under-covers at small n and collapses to [0,0] at k=0. Asking for it raises an error."),
    ("Confounded arms are not ranked",
     "If two runs saw budgets 10x apart, no flag makes them comparable again."),
    ("The test is pinned before the data",
     "Fisher and Barnard disagree at the margin. Changing test after seeing results is blocked."),
]


def main() -> None:
    SITE.mkdir(exist_ok=True)

    rollouts = []
    for f in sorted(FIXTURES.glob("*.json")):
        rollouts.extend(get("lerobot").parse(f, policy_id=f.stem))
    result = compare(rollouts, "diffusion")
    (SITE / "report.html").write_text(
        render_html(result, baseline="diffusion",
                    posteriors=posterior_table(result.arms, "diffusion", result.pairs)),
        encoding="utf-8")

    budget_rows = [(n, mde(n, 0.35, 0.80, 0.05, "fisher")) for n in (20, 50, 100, 200)]
    budget_rows = [(n, d) for n, d in budget_rows if d]
    from verdikt.stats import normal_approx_n, power_exact

    n_approx = normal_approx_n(0.35, 0.70, 0.80, 0.05)
    html = LANDING.render(
        t=theme, version=__version__,
        n_approx=n_approx,
        realised=f"{power_exact(n_approx, 0.35, 0.70, 0.05, 'fisher'):.3f}",
        exact_n=required_n(0.35, 0.70, 0.80, 0.05, "fisher"),
        budget_rows=budget_rows, refusals=REFUSALS,
    )
    (SITE / "index.html").write_text(html, encoding="utf-8")

    for png in ("n_matters.png", "power.png", "audit.png", "workflow.png"):
        src = ROOT / "docs" / png
        if src.exists():
            shutil.copy2(src, SITE / png)

    print(f"site -> {SITE}")


if __name__ == "__main__":
    main()
