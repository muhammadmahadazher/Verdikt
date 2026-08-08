<div align="center">

<h1>Verdikt</h1>

**Your eval printed a success rate. Verdikt tells you whether you're allowed to believe it.**

A CPU-only decision layer for robot-policy evaluation. It reads the eval JSON, dataset files
and run configs you already have — and refuses to let you draw a conclusion the data does not
support.

[![PyPI](https://img.shields.io/badge/pip-verdikt-3775A9?logo=pypi&logoColor=white)](https://github.com/muhammadmahadazher/Verdikt)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![No GPU](https://img.shields.io/badge/GPU-not_required-76B900?logo=nvidia&logoColor=white)](#)
[![Tests](https://img.shields.io/badge/tests-97_passing-3FB950)](tests/)
[![Works with](https://img.shields.io/badge/works_with-LeRobot-FF9D00)](https://github.com/huggingface/lerobot)

<img src="docs/workflow.png" width="96%" alt="Verdikt architecture: every command reads a file that already exists"/>

</div>

---

## The problem, in one picture

Two policies. Twenty episodes each. One scored 35%, the other 70%. Ship the winner?

<div align="center">
<img src="docs/audit.png" width="92%" alt="Four policy arms with Wilson confidence intervals showing substantial overlap"/>
</div>

**No.** At n=20 that comparison is `p = 0.056` — the intervals overlap, and the difference is
not significant. Meanwhile "0/20 = 0%" actually means *"below 13.9%"*, and the two 0/20 arms
are statistically indistinguishable from each other (`p = 1.000`).

Every number above was produced by the command below, from files that already existed.

```bash
verdikt compare results/*.json --baseline diffusion_50k
```

---

## What Verdikt does

| Command | Question it answers | Exit |
|---|---|---|
| `verdikt doctor` | Is my training stack about to fail *silently*? | 0 / 2 |
| `verdikt lint` | Is this dataset going to waste my GPU-hours? | 0 / 2 |
| `verdikt ingest` | Turn any harness's eval output into one canonical table | 0 |
| `verdikt plan` | How many episodes do I actually need? | 0 |
| `verdikt manifest` / `diff` | Are these two runs even comparable? | 0 / 3 |
| `verdikt compare` | Is checkpoint B **really** better than A? | 0 / 1 / 2 / 3 |

### The four-state verdict

Most gates are binary: pass or fail. Success rates are stochastic, so a binary gate on a
binomial produces constant false alarms. Verdikt returns four states, so "I can't tell yet"
is a first-class answer instead of a silent pass:

| Exit | State | Meaning |
|:---:|---|---|
| `0` | **BETTER** | no regression — ship it |
| `1` | **REGRESSION** | candidate is significantly worse |
| `2` | **UNDERPOWERED** | cannot decide at this n — *here is the n that would* |
| `3` | **NOT COMPARABLE** | a confound makes the comparison meaningless |

---

## Install

> **Requirements:** Python 3.10+. No GPU. No robot. No simulator. Nothing to configure.

<details open>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install git+https://github.com/muhammadmahadazher/Verdikt
```
</details>

<details>
<summary><b>🐧 Linux</b></summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/muhammadmahadazher/Verdikt
```
</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/muhammadmahadazher/Verdikt
```
*Identical on macOS — Verdikt is pure CPU Python, so Apple Silicon and Intel both just work.*
</details>

**Optional extras:** `pip install "verdikt[hub,gpu,wandb]"` — HuggingFace dataset resolution,
NVIDIA device detection for `doctor`, and Weights & Biases write-back.

**From source (for development):**
```bash
git clone https://github.com/muhammadmahadazher/Verdikt && cd Verdikt
pip install -e ".[dev]" && pytest -q
```

---

## Quickstart — 60 seconds, on results you already have

```bash
# 1. is my stack lying to me?
verdikt doctor --train-config runs/act/checkpoints/last/pretrained_model/train_config.json

# 2. what do my existing eval files actually say?
verdikt compare "results/*.json" --baseline diffusion

# 3. how many episodes would settle it?
verdikt plan --p0 0.35 --mde 0.35 --power 0.80

# 4. were these two runs ever comparable?
verdikt manifest runs/act --policy-id act && verdikt manifest runs/smolvla --policy-id smolvla
verdikt diff runs/act/manifest.json runs/smolvla/manifest.json
```

Verdikt reads `lerobot-eval` output natively. For any other harness, describe your columns
once and everything downstream works:

```bash
verdikt ingest my_evals.csv --adapter csv --map success=passed,policy_id=model
```

---

## How many episodes do you actually need?

<div align="center">
<img src="docs/power.png" width="96%" alt="Exact power curves versus the normal approximation"/>
</div>

This is the design decision that makes Verdikt worth installing.

`statsmodels` will tell you **31 episodes per arm** are enough to separate 35% from 70% at 80%
power. Run Fisher's exact test at n=31 and the realised power is **0.741**. A planner that
plans with the normal approximation and decides with an exact test under-recommends rollouts —
the exact opposite of its purpose.

Verdikt computes power by **exact enumeration through the test that will issue the verdict**.
No simulation error, no approximation gap.

| Comparison | Verdikt (exact) | Normal approximation | Realised power at that n |
|---|---:|---:|---:|
| 35% vs 70% | **37**/arm | 31/arm | 0.741 ❌ |
| 35% vs 50% | **183**/arm | 170/arm | 0.778 ❌ |

And the number nobody wants to hear:

| Your budget | Smallest difference you can detect vs a 35% baseline |
|---:|---:|
| n = 20 | **47 percentage points** |
| n = 50 | 30 pp |
| n = 100 | 20 pp |
| n = 200 | 15 pp |

At n=20, power to detect 35%-vs-70% is **0.468** — worse than a coin flip.

---

## Structural refusals

These are enforced in the formatter, not left to the caller's discipline. They cannot be
forgotten in a hurry:

- 🚫 **A success rate never prints without `n` and an interval.**
- 🚫 **`0/n` never prints as "0%".** It prints its exact one-sided bound (`0/20 → ≤ 13.9%`).
- 🚫 **There is no `--min-success` flag.** Gating a stochastic binomial on a point estimate is
  the malpractice this tool exists to stop. Only `--min-lower-bound` and
  `--noninferiority --margin` exist.
- 🚫 **The Wald interval is not implemented.** It under-covers at small n and collapses to
  `[0, 0]` at k=0. Asking for it raises an error explaining why.
- 🚫 **Confounded arms are suppressed, not ranked.**
- 🚫 **Changing the test after seeing data is blocked** when a pre-registered `plan.json` is
  supplied — that's test-shopping, and at the margin it flips verdicts.

That last one is not hypothetical. On real data from the case study below:

```
act vs diffusion   Fisher p = 0.008316   SIGNIFICANT
                 Barnard p = 0.009984   not significant
      Bonferroni-corrected alpha = 0.008333
```

Two defensible exact tests, opposite verdicts, same data. Verdikt always names the test that
produced the number and flags when the alternative would disagree.

---

## Case study: auditing my own published benchmark

Verdikt was validated by pointing it at
[**vla-on-a-budget**](https://github.com/muhammadmahadazher/vla-on-a-budget) — my own
published study comparing ACT, Diffusion Policy and SmolVLA — and correcting its headline.

| Claim as published | What Verdikt returns |
|---|---|
| "Diffusion 35% vs upstream 70%" | `p = 0.056` — **not significant** at n=20 |
| "ACT 0%" | honestly: **≤ 13.9%** (one-sided 95%) |
| "ACT vs SmolVLA" | `p = 1.000` — **indistinguishable** |
| SmolVLA ranked beside the others | **`NOT COMPARABLE`** — 10× fewer samples seen |

That last row is arithmetic, not inference:

```
verdikt diff runs/act/manifest.json runs/smolvla/manifest.json

samples_seen    1.6e+06    1.6e+05    COMPUTE_CONFOUND  10.0x
what differs
  normalization_mode
    VISUAL: MEAN_STD  ->  IDENTITY
=> act and smolvla are NOT comparable as an architecture result   (exit 3)
```

The study's **qualitative** finding survives — generative action heads beat deterministic
regression on multimodal demonstrations. The **precision of the headline** does not. That
correction is now published in the study itself.

### Then we re-ran the evaluation at n=200 and watched the fog clear

<div align="center">
<img src="docs/n_matters.png" width="96%" alt="The same four policies at n=20 and at n=200"/>
</div>

Same four checkpoints, same task, ten times the episodes — 800 rollouts of inference, no
retraining. At n=20 the tool could only say *"two groups, and one comparison is too close to
call."* At n=200 every pair separates and **three** distinct performance tiers appear:

| policy | n=20 | n=200 | 95% CI at n=200 | group |
|---|---:|---:|---|:---:|
| upstream diffusion | 70% | **65.5%** | [58.7, 71.7] | c |
| diffusion 50k | 35% | **24.0%** | [18.6, 30.4] | b |
| act 50k | 0% | **1.0%** | [0.3, 3.6] | a |
| smolvla 20k | 0% | **0.0%** (≤1.5%) | [0.0, 1.9] | a |

Note what n=20 got *wrong in both directions*: diffusion looked like 35% and is really 24%;
ACT looked like a flat 0% and actually solves 1% of episodes. Neither error was detectable
from the smaller sample — which is the entire argument for computing required-N before you
run the eval rather than after.

This corpus ships in the repo as `tests/fixtures/pusht_n200/`, so every statistical feature
is demonstrated on real robot-policy rollouts rather than synthetic numbers.

---

## FAQ

<details>
<summary><b>Isn't this just a few scipy calls?</b></summary>

The math is commodity, and pretending otherwise would be dishonest. The value is that the
correct math is *always* applied: version-pinned adapters, opinionated defaults that make
malpractice structurally impossible, a four-state exit code, and named tests with disagreement
warnings. The moat is being correct and finished, not being clever.
</details>

<details>
<summary><b>Does it work with my eval harness?</b></summary>

Natively with `lerobot-eval` output. Anything else works through the generic CSV/JSON mapper
in one line. Adapters declare the schema version they parse and **fail loudly** on unknown
versions rather than silently mis-mapping a field.
</details>

<details>
<summary><b>Why not just use Weights & Biases?</b></summary>

W&B stores and plots your numbers; it does not tell you whether a difference is real. Verdikt
is the decision layer on top and writes back into it.
</details>

<details>
<summary><b>Are rollouts really independent Bernoulli trials?</b></summary>

Often not. Session drift, reused object placements and operator fatigue induce correlation
that makes naive intervals *narrower* than the truth. This is a correctness ceiling, stated
here rather than buried: when seeds or session IDs are available Verdikt detects intra-run
correlation and widens rather than pretends. Single-seed results always carry a warning.
</details>

<details>
<summary><b>Does it need a GPU, a robot, or a simulator?</b></summary>

None of the three. It reads files. Every command runs on a laptop in seconds.
</details>

---

## What Verdikt deliberately does **not** do

Scope discipline is a feature. Verdikt will never:

- run policies, train, or execute rollouts — every eval runner is an **input**, not a competitor
- define a new dataset format, benchmark suite, or simulator
- cluster ~20 failed rollouts into "failure modes" (numerology at that sample size)
- guess your VRAM ceiling (a false *GO* is worse than no answer)
- host a leaderboard
- extrapolate an "iso-sample projection" between a pretrained VLA and a from-scratch policy —
  refusal is defensible, extrapolation is not

---

## Roadmap

| Status | Feature |
|:---:|---|
| ✅ | `ingest` · `plan` · `compare` · `doctor` · `manifest` / `diff` |
| 🔜 | `lint` — five dataset rules, each with a deliberately-corrupted test fixture |
| 🔜 | `report` — self-contained HTML + LeRobot-format model card |
| 🔜 | `watch` — anytime-valid sequential stopping (*ends evals early*), gated on a 20,000-run false-positive calibration suite |
| 🔬 | `profile` — dataset multimodality bound, `--experimental` only, gated on a published calibration experiment |

---

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
The highest-value contribution is **an adapter for your eval harness**: drop a golden fixture
in `tests/fixtures/adapters/`, and the parser is usually 30 minutes of work.

Verdikt is released under the **[Apache License 2.0](LICENSE)** — permissive, patent-granting,
and usable inside a commercial pipeline. That is deliberate: the closest prior art on rigorous
robot-policy evaluation ships under a non-commercial license, which keeps it out of exactly the
CI pipelines that need it most.

## Citing

```bibtex
@software{verdikt2026,
  author = {Azher, Muhammad Mahad},
  title  = {Verdikt: a decision layer for robot-policy evaluation},
  year   = {2026},
  url    = {https://github.com/muhammadmahadazher/Verdikt},
  license = {Apache-2.0}
}
```

## Related projects

- [**vla-on-a-budget**](https://github.com/muhammadmahadazher/vla-on-a-budget) — the benchmark study Verdikt audits
- [**OpenVocab-4D**](https://github.com/muhammadmahadazher/openvocab-4D) — open-vocabulary 4D scene understanding on an 8 GB laptop
- [LeRobot](https://github.com/huggingface/lerobot) — the robotics library Verdikt reads

<div align="center">
<sub>Every number in this README is computed by <code>docs/make_figures.py</code> and asserted in <code>tests/</code>. None of them were typed by hand.</sub>
</div>
