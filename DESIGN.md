# Verdikt

**A CPU-only decision layer for robot-policy evaluation: it reads the eval JSON, dataset files, and run configs you already have, and refuses to let you draw a conclusion the data does not support.**

Apache-2.0 · pip-installable · no GPU · no robot · no new workflow

---

## 0. TL;DR for the impatient reviewer

Six subcommands. Every one reads a file that already exists on disk and writes a file someone already asks for. Nothing trains, nothing runs a policy, nothing decodes video, nothing tails a training log.

```
verdikt doctor      # silent-failure preflight for the training stack        (exit 0/1/2)
verdikt lint        # LeRobotDataset integrity + normalization provenance     (exit 0/1/2)
verdikt ingest      # any eval harness's JSON -> one canonical rollout table
verdikt plan        # required-N / MDE, computed under the exact test used
verdikt compare     # BETTER / NOT DIFFERENT / UNDERPOWERED / NOT COMPARABLE  (exit 0/1/2/3)
verdikt report      # self-contained HTML + LeRobot-format model card
```

The launch artifact is the author running Verdikt against his own published benchmark and correcting his own headline: `vla-on-a-budget`'s "Diffusion 35% vs upstream 70%" is not significant at n=20, its "ACT 0%" honestly reads "below 13.9% at 95% one-sided confidence", its ACT-vs-SmolVLA comparison is p=1.000, and its SmolVLA arm saw 10× fewer samples than the arms it is being ranked against.

Everything else — the dataset-multimodality verdict, sequential stopping, perception, planners, leaderboards, VRAM probes — is either deferred behind a validation experiment or killed outright, with reasons below.

---

## 1. Why this and not the alternatives

Three judges independently converged on the same shape. The statistics-and-artifacts layer is (a) the loudest documented gap, (b) the only part whose claims are defensible **today**, and (c) the only part a solo builder on an 8 GB Windows laptop can ship without a robot.

The seam is precise:

| Layer | Owner in 2026 | Verdikt's relation |
|---|---|---|
| Simulators / benchmarks | robosuite, LIBERO, RoboCasa365, ManiSkill3, Meta-World | never touched |
| Eval runners | `lerobot-eval`, allenai/vla-evaluation-harness, robocurve/inspect-robots, robomimic | **input** — distribution channels, not competitors |
| Dataset curation / annotation | FiftyOne, Encord, ARES, score_lerobot_episodes | adjacent; hygiene tier deliberately not reimplemented |
| Experiment tracking | W&B, MLflow | writes back into, does not replace |
| **The decision** | **nobody** | **this** |

The only prior art aimed at the same target is TRI's STEP — correct methodology, released as four Colab notebooks under a **non-commercial license**, integrated into nothing. The differentiator is unglamorous and decisive: the same math as an Apache-2.0 CLI with a non-zero exit code, so it can go inside a company's pipeline.

**The honest counter to "it's just scipy calls."** It is. The moat is not the math — it is version-pinned adapters with golden fixtures, deliberately-corrupted dataset fixtures per lint rule, opinionated defaults that make malpractice structurally impossible (no bare rate, Wald unavailable, no fixed-sample verdict on incrementally-collected data), the four-state exit code, and a Monte-Carlo FPR suite that proves the sequential test is not silently broken. Budget the fixture suite as the product.

---

## 2. Target user and the recurring moment

A 1–8 person robot-learning team or an individual engineer, LeRobot or openpi in the stack, one or two consumer GPUs, 0–2 arms. Two moments that recur every week:

1. **Thursday 9pm.** Two terminals show `18/50` and `22/50`. Do I ship B? What do I tell my lead? Today: vibes and a Slack thread.
2. **Monday 9am.** A vendor or teammate handed me a converted LeRobotDataset and I am about to spend 40 GPU-hours on it. Today: `lerobot-train` and hope.

The CI-gate framing is kept (exit codes cost nothing, and teams with nightly evals will find them) but is **not** the headline. Most manipulation teams do not retrain in CI. The pitch is the nightly-eval verdict and the report you hand to your lead.

---

## 3. MVP — what ships first and stands alone

### 3.1 `verdikt ingest`

```
verdikt ingest results/*.json --adapter lerobot --out rollouts.parquet
verdikt ingest my_evals.csv --adapter csv --map success=passed,seed=init_seed
```

Writes one canonical Parquet table. One row per rollout:

| column | type | notes |
|---|---|---|
| `run_id` | str | ties to a manifest |
| `policy_id` | str | |
| `task`, `suite` | str | |
| `episode_idx` | int32 | |
| `seed` | int64 \| null | null ⇒ paired tests unavailable, printed loudly |
| `success` | bool \| null | |
| `progress` | float32 \| null | partial credit in [0,1] if the harness logged it |
| `steps`, `wall_clock_s` | int32/float32 | |
| `label_source` | enum | `human` \| `simulator` \| `scripted` — printed on every downstream line |
| `manifest_id` | str \| null | |

**Adapter policy (a deliberate scope cut).** Ship exactly **two** real adapters: `vla-on-a-budget`'s own `results/*.json`, and `lerobot-eval` output. Plus a documented **generic CSV/JSON schema mapper** as a permanent escape hatch. Every other harness (inspect-robots, robomimic, SimplerEnv, vla-evaluation-harness) gets a golden-fixture template in `tests/fixtures/adapters/` and a CONTRIBUTING section — because *you cannot write or test an adapter for output you have never generated*, and generating it means MuJoCo, Docker, and Linux-only stacks on an 8 GB Windows laptop. Adapters declare the upstream schema version they parse and **fail loudly on unknown versions** rather than silently mis-mapping.

### 3.2 `verdikt plan` — required-N under the test you will actually use

```
verdikt plan --p0 0.35 --mde 0.15 --power 0.80 --alpha 0.05 --test barnard
verdikt plan --budget 20 --p0 0.35            # inverse: what can I detect?
```

```
required N per arm (simulated, 20,000 draws, test=barnard, two-sided a=0.05)
  35% vs 50%   ->  188/arm   (normal-approx would have said 170)
  35% vs 70%   ->   37/arm   (normal-approx would have said 31 -> power 0.749)
at your budget of n=20/arm the minimum detectable effect vs 35% is 39.4 pp
  -> this run cannot answer your question. see `verdikt plan --explain`
wrote plan.json  (sha256 3f9c… pinned into the manifest)
```

**Non-negotiable design point, from the rigor verdict.** `statsmodels.stats.power.NormalIndPower` says 31/arm for 35%-vs-70%; Fisher's exact at n=31 delivers **0.749** power, not 0.80. A tool that plans with the normal approximation and decides with an exact test under-recommends rollouts by 15–30%, which is the exact opposite of its stated purpose. Verdikt computes required-N by **Monte-Carlo simulation through the exact test that will issue the verdict**. It costs seconds. `--fast` exposes the normal approximation, clearly labeled as an approximation.

`plan` also emits the **pre-registration commitment**: `blake2b(test, alpha, alternative, planned_N, hypothesis)` written into `verdikt.toml` and the manifest.

### 3.3 `verdikt compare` — the core daily command

```
verdikt compare rollouts.parquet --by policy_id --baseline act \
    --manifests runs/*/manifest.json --format text
```

```
policy      n   success        95% CI (Wilson)   letter   samples_seen
act        20   0/20    0.0%   [ 0.0, 16.1]        a         1.60e6
smolvla    20   0/20    0.0%   [ 0.0, 16.1]        a         1.60e5
diffusion  20   7/20   35.0%   [18.1, 56.7]        b         1.60e6
upstream   20  14/20   70.0%   [48.1, 85.5]        b            n/a

  0/20 does not mean zero. One-sided 95% upper bound: 13.9%.
  (Two-sided Clopper-Pearson upper is 16.8% - reported separately, not as "the" bound.)

pre-registered test: barnard, alpha=0.05 two-sided, planned N=20  [hash 3f9c… VERIFIED]
pairwise (Bonferroni m=6, family-wise alpha=0.05):
  diffusion vs act        p=0.00998  NOT SIGNIFICANT at a/m=0.00833   <- Fisher would say 0.0083
  diffusion vs smolvla    -- SUPPRESSED, see CONFOUND
  diffusion vs upstream   p=0.0333   NOT SIGNIFICANT at a/m=0.00833
  act vs smolvla          p=1.000    INDISTINGUISHABLE
bayes: P(diffusion > upstream) = 0.014      P(act > smolvla) = 0.500

CONFOUND  smolvla saw 1.60e5 samples; act/diffusion saw 1.60e6 (10.0x).
          architecture claims involving smolvla are SUPPRESSED, not ranked.
          see `verdikt diff runs/act/manifest.json runs/smolvla/manifest.json`

VERDICT   UNDERPOWERED
          n=20/arm cannot separate 35% from 70% at 80% power; you need 37/arm.
exit 2
```

Note three things a naive implementation would get wrong, all caught by the rigor judge and fixed here:

- **The test and the p-value must agree.** Barnard on 7/20 vs 0/20 is 0.00998; Fisher is 0.0083. Under Bonferroni m=6 (α/m = 0.00833) those give *opposite* verdicts. The tool prints which test ran, prints the alternative test's value as a footnote, and never lets the reader think the choice was free.
- **"Barnard flips your result to significant" is not a selling point.** It is test-shopping. That is exactly why the test is pinned *before* the data and the hash is verified at verdict time.
- **`0/20 → below 16.8%`** is the two-sided Clopper-Pearson upper mislabeled as one-sided. The one-sided 95% figure is **13.9%**. Both are printed, correctly labeled.

Flags:

```
--test {barnard,boschloo,fisher,mcnemar}   pinned by verdikt.toml; CLI override requires --break-preregistration
--interval {wilson,jeffreys,clopper-pearson}   (wald is not implemented, on purpose)
--paired          requires seed column on both arms; hard-errors with a fix hint if absent
--partial         use the `progress` column; prints label_source on every line
--min-lower-bound 0.50   gate on the LOWER CI bound, never on p-hat
--noninferiority --margin 0.05
--format {text,json,markdown,github}
```

**Exit codes** (the signature design decision, from rollcall):

| code | meaning |
|---|---|
| 0 | BETTER or NOT-WORSE — no regression |
| 1 | REGRESSION — candidate is worse, significantly |
| 2 | UNDERPOWERED — cannot decide at this n; here is the required n |
| 3 | NOT COMPARABLE — manifest confound (samples-seen ratio, dataset revision, normalization mode) |

Gating on a stochastic binomial with a two-state code produces constant false alarms. The four-state code is ~20 lines and no incumbent has it.

**Structural refusals**, enforced at the formatter level so they cannot be forgotten:
- a success rate cannot be rendered without `n` and an interval;
- `0/N` and `N/N` render as exact one-sided bounds;
- `--min-success` on a point estimate **does not exist**; only `--min-lower-bound` and `--noninferiority --margin`;
- a fixed-sample verdict on data whose realized N ≠ planned N is refused with a pointer to the sequential test (Phase 2).

### 3.4 `verdikt manifest` / `verdikt diff` — provenance and the compute confound

```
verdikt manifest runs/diffusion --out runs/diffusion/manifest.json
verdikt diff runs/act/manifest.json runs/smolvla/manifest.json
```

Captures: lerobot version + git sha, dataset repo id + resolved HF revision sha + blake2b content hash over `meta/`, resolved train config, seed(s), batch size, grad-accum, gradient steps, **derived `samples_seen = batch × grad_accum × steps`**, torch / CUDA / driver versions, GPU name and peak VRAM, wall clock, normalization mode, and the `plan.json` hash.

```
verdikt diff act smolvla
  field                 act              smolvla          class
  policy.type           act              smolvla          EXPECTED
  batch_size            32               8                CAUSE
  steps                 50000            20000            CAUSE
  samples_seen          1.60e6           1.60e5           COMPUTE_CONFOUND (10.0x > 2.0x)
  dataset.revision      9f2c1ab          9f2c1ab          ok
  normalization         MEAN_STD         MEAN_STD         ok
=> these two runs are NOT comparable as an architecture result.   exit 3
```

This is arithmetic — one day of work, no GPU, no calibration, no validation study — and it is the single most defensible strong opinion in the whole product. It converts the author's own SmolVLA result from an embarrassment into a demonstration.

**Deliberately NOT included: the iso-sample *projection*.** There is no single fair normalization between a pretrained 450M VLA and a from-scratch 52M regressor — the VLA's data exposure includes pretraining. Verdikt reports the frontier (success vs samples-seen, with intervals) and **declines the verdict**. Refusal is defensible; extrapolation is not. Note also that grad-accum is never offered as a remedy for a samples-seen deficit: it leaves samples-seen exactly unchanged.

### 3.5 `verdikt lint` — dataset integrity and normalization provenance

Pure `pyarrow` + `json`. **Never imports `lerobot`**, so it still runs on a broken install — which is precisely when you need it.

```
verdikt lint hf://lerobot/pusht --train-config configs/act.yaml --format sarif --fail-on error
```

```
dataset  lerobot/pusht  rev 9f2c1ab  codebase_version v2.1  206 ep  25,650 frames
DS001 ok    fps 30 (int), matches meta/info.json type
DS005 ERROR meta/stats.json observation.state.mean computed over 206 episodes,
            but --dataset.episodes selects 180. delta = 0.71 sigma on action[1].
            every prediction will shrink toward zero.
            upstream: the 256/45-of-301 field case; see docs/rules/DS005.md
            fix: verdikt lint --rewrite-stats --episodes 0:180 --out meta/stats.fixed.json
DS006 ERROR config requests normalization_mode=QUANTILES but meta/stats.json
            has no q01/q99 for observation.image -> silent identity fallback
            upstream: huggingface/lerobot#821
DS008 WARN  state/action cross-correlation peaks at lag -1 (expected 0)
            probable uncompensated teleop latency or off-by-one
2 errors, 1 warning, 8 passed.  exit 2
```

**MVP rule set — five rules with real corrupted fixtures, not eleven aspirational ones.** Scope discipline: a rule without a fixture is a liability.

| ID | Check | Provenance |
|---|---|---|
| DS001 | `fps` type/value mismatch (30 vs 30.0) between `info.json` and metadata check | lerobot#1999 |
| DS003 | `dataset_from_index` / `dataset_to_index` monotonicity + coverage vs actual parquet row counts (the scrambling that makes `{key}_is_pad` true past episode ~3) | lerobot#1919, #2401 |
| DS005 | Recompute per-feature mean/std/q01/q99 over the **selected train split** (streaming Welford + t-digest) and diff against `meta/stats.json`, reported in σ units | 256/45-of-301 field case; openpi#711 |
| DS006 | Normalization feasibility: declared mode has the stats it requires; any feature type absent from the norm map that would hit the silent identity fallback | lerobot#821; LeRobot docs name this "the most common reproducibility pain point" |
| DS008 | State/action temporal alignment: cross-correlate `d(state)/dt` against `action` over lags −5..+5, flag argmax ≠ 0 | teleop-latency field reports |

Phase 2 adds DS002 (rollover chunk/file index), DS004 (missing/orphaned shards), DS007 (timestamp monotonicity, fps drift), DS009 (dead/jittery joints), DS010 (action saturation), DS011 (camera-key consistency). Each ships with its own corrupted fixture or it does not ship.

Every rule prints its threshold, its upstream issue link, and has a deliberately-corrupted mini-dataset in `tests/fixtures/datasets/DS00N_broken/`. **The fixtures are the moat**, not the checks. They are what let a skeptical engineer verify a threshold instead of trusting it.

Parsing is defensive off `info.json.codebase_version`, and rules are data-driven YAML so a format bump is a config edit, not a refactor.

### 3.6 `verdikt doctor` — four checks, no cosmetics

```
verdikt doctor --train-config configs/act.yaml
```

Scoped to the non-obvious and expensive only. Everything cosmetic was cut.

1. **CPU-wheel shadowing.** `torch.version.cuda is None` while `pynvml` enumerates an NVIDIA device ⇒ `pip install lerobot` replaced your CUDA build. Prints the exact `uv pip install --index-url` fix.
2. **`rename_map` keys that validate but never reach the batch.** Cross-check config keys against `meta/info.json` features. Passes config validation in 0.5.x and is silently never applied — blocks cross-embodiment finetune from `smolvla_base`.
3. **Normalization stats vs the actual train split**, diffed in σ units (shares the DS005 engine, run against the resolved config rather than the dataset alone).
4. **QUANTILES declared without q01/q99 present** (shares DS006).

Plus two advisory warnings that cost nothing: `torchcodec` unavailable ⇒ silent `pyav` fallback (Windows / macOS-Intel / Linux-ARM with torch<2.8), and laptop-dGPU-on-battery power-gating (`GetSystemPowerStatus` / sysfs).

Each check is worth weeks and none of them raises an exception today.

### 3.7 `verdikt report`

```
verdikt report rollouts.parquet --manifests runs/*/manifest.json \
    -o report.html --modelcard MODEL_CARD.md
```

One self-contained HTML file (jinja2 + inline matplotlib SVG, no CDN, no external fonts): the comparison table with letters, forest plot of intervals, Beta-posterior violins (because overlapping CIs are systematically misread), the manifest diff with confound classes, the lint findings, and the four-state verdict banner. Plus `MODEL_CARD.md` matching LeRobot's `bring_your_own_policies` reporting standard — suite/success-rate/n_episodes table, exact eval command, hardware string, checkpoint link — with the intervals and the verdict inserted.

**Per-metric provenance labels in the output itself**, three classes, printed inline:

- `[validated]` — Wilson/Clopper-Pearson/Jeffreys, Barnard/Boschloo/Fisher/McNemar, Beta posteriors, simulated power
- `[prior-art]` — anything reimplemented from an existing tool, named and cited
- `[provisional]` — anything not yet calibrated; carries its calibration status inline

An engineer's trust dies the first time an unvalidated claim is spoken in the same voice as a validated one. This costs nothing and prevents exactly that.

---

## 4. What the MVP is worth on its own

Without a single line of Phase 2+, a user gets: a dataset linted for the failure modes that cost weeks and never crash; a stack checked for the CPU-wheel and rename_map traps; an honest table with intervals for any eval they already ran; a required-N number computed under the right test; a manifest diff that catches the confound that silently invalidates most architecture comparisons; and a model card + HTML report they can hand to a lead. That is a complete, coherent product: *measure your dataset before you spend the GPU, measure your result before you believe it.*

---

## 5. Phased extensions

### Phase 2 — sequential stopping and the power-recovery layer (~1 week)

The MVP takes claims away. Phase 2 gives budget back, which is the sequencing that makes the tool welcome rather than scolding.

**`verdikt watch`** — anytime-valid stopping.

```
verdikt watch runs/eval_live/ --baseline results/diffusion.json --alpha 0.05 --partial
```

Test supermartingale `X_{n+1} = (1 + ξ_n·(r_B,n − r_A,n))·X_n` over per-episode rewards in [0,1], stopping at `X ≥ 1/α` by Ville's inequality; ξ tuned online by a mixture over a grid, not fixed. Two supermartingales at α/2 for two-sided use. Accepts binary success or the `progress` column.

**Non-negotiable release gate:** a CI job runs **20,000 null simulations per shipped configuration** and asserts empirical FPR ≤ α at *every* stopping time. Any variant that does not pass does not ship. A subtly broken e-value construction silently inflates false positives and is strictly worse than shipping no sequential test.

**The claimed savings are not printed until measured.** "25–70% fewer rollouts" and "expected stop ~14/arm" are borrowed from papers with different effect sizes. Verdikt prints only savings measured on the author's own gym-pusht re-run (§8, Experiment C). Until then the output says `expected remaining: unknown (uncalibrated)`.

**`verdikt seeds`** — paired evaluation.

```
verdikt seeds --n 200 --env gym_pusht/PushT-v0 --out seeds.json
```

Emits a deterministic seed list; both policies see byte-identical initial states, enabling exact McNemar. **Week-one prerequisite (§8, Experiment B):** verify that `lerobot-eval` actually threads a per-episode seed through to `env.reset(seed=...)` and logs it. If it does not, this headline quietly stops applying and the roadmap must know before anything is built on it. When seeds are absent, `compare --paired` hard-errors with the fix rather than silently degrading.

**`verdikt gate`** + shipped GitHub Action — the same four-state logic pinned against a `verdikt.lock` baseline, with Holm correction across tasks and a Markdown job summary / PR comment writer.

**W&B write-back** — `wandb.Api()` pull of run history; verdict, intervals, and report HTML pushed back as a run summary panel and artifact.

**Single-seed warning.** When `n_seeds == 1`, print one line: *"this interval is per-rollout binomial uncertainty for one training run; documented seed-to-seed spreads on identical configs reach 22% vs ≥98%."* Nothing more — the full beta-binomial hierarchical model is **cut** (see §6), because the author has one seed per policy and cannot dogfood it.

### Phase 3 — `verdikt profile`, behind `--experimental` and behind a validation gate (~1 week, gated)

The dataset-multimodality diagnostic is the most interesting idea in the whole design space and the one most likely to end the project's credibility. It ships **only** if Experiments D and E (§8) pass, **only** behind `--experimental`, **only** as a bound, and **never** as a success-rate prediction.

Three rigor objections must be answered in code before it exists at all:

1. **The Gaussian null is miscalibrated on real data.** Reimplemented as specified, the nominal 5% FPR became 0.117 / 0.225 / 0.468 under t(8)/t(5)/t(3) action noise, and trajectory sampling alone doubled it to ~0.11 because k-NN neighbours in an episodic dataset are *consecutive frames of the same trajectory*. **Fix:** exclude neighbours within ±H frames of the same episode, and replace the isotropic-Gaussian threshold table with a **permutation null resampled from the user's own residuals** under a unimodality-preserving transform (sign-flip / random rotation about the neighbourhood mean). Ship only if the recalibrated FPR holds ≤0.07 across t(3)/t(5)/t(8)/Gaussian noise and across trajectory-blocked sampling.
2. **The bound does not bind the policy it indicts.** `lerobot/pusht`'s `observation.state` is 2-D agent position; the T-block pose lives in the image, which both ACT and Diffusion Policy consume. AMR computed on proprio alone is the loss floor for a policy *that cannot see the block*. Demonstrated magnitude: dropping half the state raised AMR from 0.153 to 0.653 on identical unimodal data. **Fix:** compute under **≥2 embeddings** and **refuse to issue a verdict when they disagree**; print the embedding identity and its PCA participation ratio on every line; warn above effective dim 20, refuse above 40.
3. **The identity is L2 and ACT trains L1 + CVAE KL.** `argmin E‖a−f(s)‖² = E[a|s]` with irreducible loss `tr Cov(a|s)` is an L2 statement. ACT's minimizer is the conditional median with irreducible loss `E[MAD(a|s)]`. **Fix:** compute and report **both** the L2 ratio and the L1/MAD ratio, and state which loss each bounds. Never print a bound in raw MSE units (`tr Cov` is not invariant to per-dimension action scaling — a binary gripper alongside radians makes the trace whatever the normalizer says); report the dimensionless ratio with the normalization convention printed.

Output shape, if it ships:

```
verdikt profile hf://lerobot/pusht --embedding observation.state --chunk 16 --experimental
```

```
[provisional] embedding observation.state (2d)  PCA participation ratio 2.0  OK
  AMR (L2 loss-floor ratio, Bessel-corrected, k=24)   0.61  [0.58, 0.64]
  MAD ratio (L1 loss floor, bounds ACT's objective)   0.54  [0.51, 0.57]
  multimodal fraction (permutation null, episode-blocked kNN)  0.78
BOUND    a deterministic L2 head cannot reduce residual action variance below
         61% of total, GIVEN THIS EMBEDDING.
CAVEAT   this embedding excludes object pose. ACT/DP consume the image.
         a second embedding is required before any verdict is issued.
VERDICT  INSUFFICIENT EVIDENCE - bound only. no architecture recommendation.
```

A first-class `INSUFFICIENT EVIDENCE` verdict, and a calibration ledger (append-only JSONL of prediction → measured outcome) that is **logged but never printed as a confidence grade** until n reaches the dozens. Printing "confidence: MEDIUM, rho=0.68" from four points is decoration wearing the costume of rigor — the exact sin the tool exists to punish.

### Phase 4 — reach (~ongoing)

Remaining lint rules with fixtures; community adapters against the golden-fixture template; docs site; upstream engagement (offer DS003 as a check against the open v2.1→v3.0 conversion issues — that is how a solo tool gets discovered).

---

## 6. What this deliberately does NOT do

Every item here was proposed and is cut on purpose. The reasons matter as much as the cuts.

**Killed outright — will not be built:**

- **Any perception backend (VGGT / SAM3 / DINOv2 grounding).** The builder's own `CVPR_ex01/PIPELINE_PLAN.md` states VGGT is static-scene biased and that VGGT and SAM3 cannot be co-resident on 8 GB. Robot eval footage is a fixed camera observing a dynamic foreground — the inverse of VGGT's operating assumption — and PushT is 96×96 synthetic top-down renders with no recoverable 3D structure. Two GPU-bound weeks to extract a state PushT already publishes, gated behind a κ≥0.8 check that can retroactively void the investment. Worse in principle: routing perception-derived binary labels into a confidence interval **launders a vision error into a statistical claim**. Verdikt accepts a `progress` column from any source and prints `label_source` on every line; it will never manufacture one.
- **Failure clustering (HDBSCAN over ≤20 failed rollouts).** Clustering ~12–20 points in a hand-built feature space is numerology; a human watches 20 videos in six minutes and gets a better taxonomy. It is the feature most likely to produce a confident wrong story that someone repeats in a design review.
- **STAC/MMD chunk-consistency scoring.** Requires loading checkpoints and running policy inference, breaking the reads-artifacts-only rule. It also degenerates for deterministic policies (ACT at z=0), where the point-distance variant is documented to perform *worse* than baselines — i.e. it silently fails on exactly the policy class the tool would be telling you to avoid.
- **Convex-hull eval-reset coverage.** Hull volume in ≥3-D from a few hundred initial states is severely downward-biased and unstable to one outlier. "Your evals cover 38% of the training hull" is not a reproducible statistic. (The energy-distance permutation test on initial states is valid and may return later; the hull number never will.)
- **Empirical VRAM bisection / `fit` / batch-size planner.** Every practitioner has a 30-line try/except OOM loop. A synthetic-batch probe cannot rule out the mid-run OOM from allocator fragmentation and real video-decode buffers, and a **false GO is worse than no answer**. The torch.compile / bitsandbytes / gradient-checkpointing sweep is additionally the most Windows-hostile surface anyone proposed.
- **Scaling priors / projected success curves.** The anchor table cannot be built: public model cards rarely report samples-seen or eval n, and documented seed variance spans 22%→98% on one config. Any band tight enough to be useful is lying, and its failure would poison the measured features that are solid.
- **Live training `watch` TUI and the loss-floor tripwire.** Mapping a dataset-space AMR ratio onto a live policy's normalized, chunk-weighted, possibly-L1 loss after the v0.5+ processor pipeline is not computable as specified. A tripwire that spuriously aborts a 7-hour run destroys trust permanently. It also requires tailing a log format LeRobot has already broken twice.
- **Leaderboard / `board` / MMRV sim-vs-real.** MMRV needs paired sim and real results (no robot). A leaderboard populated by one person has no authority, implies hosting and curation, and collides head-on with LeRobot's 0.7.0 roadmap, RoboArena, and Robocurve.
- **DemInf-style mutual-information tier.** 300–360 minutes per task for ~300 demos, four biases the authors themselves document, and an output not actionable differently from a 10-second spectral metric.
- **Multi-seed beta-binomial hierarchical model / numpyro backend.** Correct in principle, undogfoodable in practice — one training seed per policy. Replaced by a one-line warning.
- **Gradio review UI.** Competes with Rerun, HF's visualizer, and FiftyOne simultaneously.
- **`--min-success` threshold gating on a point estimate.** Gating a stochastic binomial on p̂ is the exact practice this tool exists to stop. Only `--min-lower-bound` and non-inferiority with an explicit margin exist.
- **Iso-sample *projection*.** Refusal is defensible; extrapolation between a pretrained VLA and a from-scratch regressor is not.

**Out of scope structurally:**

- Running policies, training, or executing rollouts. Every eval runner is an input.
- Any new dataset format, converter, benchmark suite, or simulator.
- Autonomous real-world evaluation (needs a robot cell, a reset policy, a success classifier).
- Distributed/crowdsourced evaluation (RoboArena needed seven institutions).
- Episode visualization and dataset curation UIs.
- The demonstration-hygiene tier (spectral smoothness, jerk, idle fraction, gripper chatter). This duplicates `score_lerobot_episodes` and HF's visualizer, adds surface area, and claims no novelty. If it ever returns it will be labeled `[prior-art]` with citations.

---

## 7. Relationship to `vla-on-a-budget`

**Recommendation: keep both repos. `vla-on-a-budget` stays as the study; Verdikt is a new repo that cites it, consumes it, and corrects it.**

Do not absorb, do not rename, do not deprecate.

**Why.** A benchmark study and a tool are different artifacts with different audiences and different maintenance curves. The study is finished and citable; the tool is alive and versioned. Folding a finished study into a moving tool loses the study's most valuable property — that it is a fixed, dated, reproducible measurement. And the tool's single strongest credibility asset is precisely that it audits an *independently published* result, not one that lives in its own README.

**Concretely:**

- New repo: `github.com/muhammadmahadazher/verdikt`. Package name `verdikt` on PyPI, console entry point `verdikt`. Apache-2.0.
- `vla-on-a-budget` gains one new section, **"Audited by Verdikt"**, plus `results/AUDIT.md` and a `verdikt-report.html`, with a one-line amendment at the top of the README: *"The headline comparison in this study is not statistically significant at n=20. See AUDIT.md. The qualitative finding — generative heads beat deterministic regression on multimodal demonstrations — survives; the precision of the headline does not."* This makes the study **stronger and more honest simultaneously**, and it costs zero new experiments.
- `vla-on-a-budget/results/*.json` are vendored into `verdikt/tests/fixtures/adapters/vla_on_a_budget/` as the first golden fixture, with a pinned upstream commit sha.
- Verdikt's README links to the study as the motivating case; the study links to Verdikt as the tool that audited it. Two artifacts, one narrative, bidirectional.
- `OpenVocab-4D` stays entirely separate. Its packaging quality (pip package, entry points, installers, COLMAP benchmark) is the evidence that this author finishes and ships — cited in the README as prior work, not integrated.

---

## 8. Validation plan

Five experiments. Every one runs on the RTX 4060 laptop or on CPU, using public LeRobot datasets and checkpoints that already exist. Experiments A–C are **prerequisites for the MVP**; D–E are **gates on Phase 3** and the tool ships without `profile` if they fail.

### Experiment A — regenerate the fixture at usable n (2 days, GPU, no training)

**The highest-leverage non-coding move available.** Re-run PushT evaluation to **n=200 per policy** for the three existing checkpoints plus upstream `diffusion_pusht`, with per-episode seeds logged and the already-available max-reward partial-credit signal recorded.

- No training. The checkpoints exist. This is inference on `gym_pusht`, hours not days.
- **Why it is a prerequisite:** at n=20 *every* statistical feature in this design is undemoable. Wilson vs Clopper-Pearson is indistinguishable, paired McNemar has no discordant pairs to speak of, sequential stopping has nothing to stop, and the power curve is a single point. At n=200 all of it becomes a real, measured demo.
- **Output:** `fixtures/pusht_n200/{act,diffusion,smolvla,upstream}.json` — the canonical test corpus for the whole project.

### Experiment B — does the paired design actually apply? (1 day, week one)

Verify that `lerobot-eval` (and one of LIBERO / `gym_pusht` directly) threads a deterministic per-episode seed to `env.reset(seed=...)` **and logs it**. Snapshot and hash the MuJoCo `qpos` / gym initial observation to confirm byte-identical scenes across two policies.

- **Pass:** paired McNemar ships as a Phase-2 headline, and Experiment C measures the real saving.
- **Fail:** the paired design is removed from the roadmap in week one rather than week three, `compare --paired` becomes a hard error with a documented reason, and the power-recovery story rests solely on partial credit and sequential stopping.

### Experiment C — measure the savings instead of citing them (2 days, CPU, uses Experiment A output)

Replay the n=200 corpus through the sequential test 10,000 times in randomized episode order and report the **measured** distribution of stopping times, for binary success and for partial credit, for each policy pair.

- Also run the mandatory **20,000-null-simulation FPR suite** and report empirical FPR at every stopping time.
- **Output:** a table of measured savings that replaces every borrowed "25–70%" claim. If the measured saving on PushT is 12%, the README says 12%.
- **Pass condition for shipping `watch`:** empirical FPR ≤ α at all stopping times, across all shipped configurations.

### Experiment D — recalibrate the multimodality null (3 days, CPU) — *Phase 3 gate*

Reimplement the mode-separation test with (i) episode-blocked kNN excluding neighbours within ±H frames of the same episode, and (ii) a permutation null resampled from the user's own residuals rather than an isotropic Gaussian. Then sweep FPR on data that is **unimodal by construction**:

- noise: Gaussian, t(8), t(5), t(3)
- sampling: i.i.d. vs trajectory-blocked
- k ∈ {8,16,24,48,96}, N ∈ {1k,3k,10k,25k}, obs-dim ∈ {2,4,20,68,260}

**Ship condition:** empirical FPR ≤ 0.07 in *every* cell with the recalibrated null. Baseline to beat, from the reproductions already on disk at `C:\Users\mahad\AppData\Local\Temp\claude\J--My-Drive-Claude-Experiments\e1f09459-d2fd-4121-80ea-e1f3e0da14a1\scratchpad\mm.py` and `mm2.py`: the Gaussian-null version reached 0.117 / 0.225 / 0.468 under t(8)/t(5)/t(3) and 0.109 under trajectory sampling. If the recalibrated version cannot clear 0.07, `profile` does not ship — and that negative result is published as a notebook, which is itself a contribution.

### Experiment E — does the bound survive the embedding? (3 days, CPU + downloads) — *Phase 3 gate*

Run `profile` across every public LeRobot dataset that has (a) usable proprioceptive state, (b) an image stream, and (c) published head-to-head results for a deterministic and a generative head. Compute AMR and multimodal fraction under **at least two embeddings** — proprio-only, and a PCA-reduced vision embedding — and report:

- the count of datasets where the two embeddings **agree** on the quadrant;
- Spearman ρ between multimodal fraction and the published deterministic-minus-generative success gap, **with its confidence interval**;
- the count of datasets the tool **declined** to score.

**Honest expectation, stated up front:** matched-budget public ACT-vs-DP pairs are rare precisely because VRAM forces budgets apart. A realistic count is 3–6, not 14, and a ρ from n=5 has a CI spanning most of [−1,1]. **This is why `profile` ships behind `--experimental` with no confidence grade and an `INSUFFICIENT EVIDENCE` default.** If the two embeddings disagree on most datasets, the correct outcome is to publish the finding and not ship the verdict.

**What cannot be bought with effort, and is stated in the README:** real-robot evidence that a dataset statistic predicts architecture choice. Verdikt substitutes published sim results and says so.

---

## 9. Module layout

```
verdikt/
  pyproject.toml            # Apache-2.0; core deps: numpy scipy statsmodels pyarrow
  README.md                 #   pandas click rich jinja2 matplotlib pydantic>=2
  verdikt/                  # optional extras: [hub]=huggingface_hub  [wandb]=wandb
    __init__.py             #                  [gpu]=pynvml
    cli.py                  # click group; every command returns an int exit code
    schema.py               # pydantic v2 models: Rollout, RunManifest, Plan, Finding
    ingest/
      base.py               # Adapter protocol + version-pinned registry
      lerobot_eval.py       # adapter #1
      vla_on_a_budget.py    # adapter #2 (also the primary golden fixture)
      generic_csv.py        # permanent escape hatch: --map col=col
    stats/
      intervals.py          # wilson, jeffreys, clopper_pearson, exact one-sided bounds
                            #   (no wald - deliberately not implemented)
      tests.py              # barnard, boschloo, fisher, mcnemar; Bonferroni/Holm; CLD
      bayes.py              # Beta posteriors, P(A>B), HDI on lift
      power.py              # simulate_required_n() through the exact test; MDE inverse
      sequential.py         # [P2] e-value supermartingale, Ville stopping, mixture xi
      _null_fpr.py          # [P2] 20k-null Monte-Carlo harness, run in CI
    lint/
      rules/                # DS001.yaml .. DS011.yaml - thresholds + citations as data
      engine.py             # pyarrow-only reader; NEVER imports lerobot
      stats_recompute.py    # streaming Welford + t-digest over the selected split
      sarif.py
    doctor/
      checks.py             # 4 hard checks + 2 advisories
    manifest/
      capture.py            # samples_seen = batch * grad_accum * steps
      diff.py               # field diff -> {EXPECTED, CAUSE, COMPUTE_CONFOUND}
    report/
      templates/            # jinja2: report.html.j2, modelcard.md.j2
      render.py             # inline SVG only; no CDN, no external fonts
    profile/                # [P3, --experimental, gated on Experiments D+E]
      amr.py                # L2 (tr Cov) and L1 (MAD) loss-floor ratios
      modes.py              # episode-blocked kNN + permutation null
      embedding.py          # participation-ratio gate; warn>20, refuse>40
      ledger.py             # append-only JSONL; logged, not printed as a grade
  tests/
    fixtures/
      datasets/DS00N_broken/   # one deliberately-corrupted mini-dataset PER RULE
      adapters/               # golden JSON per harness + expected canonical table
      pusht_n200/             # Experiment A output - the project's test corpus
    test_fpr_calibration.py   # release gate for anything sequential
    test_formatter_refusals.py# asserts a bare rate is unrenderable
  notebooks/
    01_null_calibration.ipynb # Experiment D, published
    02_embedding_sweep.ipynb  # Experiment E, published
    03_sequential_savings.ipynb # Experiment C, published
  .github/workflows/
    ci.yml                  # matrix over the two most recent LeRobot releases
    action.yml              # [P2] shipped GitHub Action
```

**Library choices, and why:** `scipy.stats.barnard_exact` / `boschloo_exact` and `statsmodels.stats.proportion_confint` / `contingency_tables.mcnemar` are the reference implementations — do not reimplement. `pyarrow` (not `datasets`, not `lerobot`) for all dataset reads so a broken install stays diagnosable. `pydantic` v2 for schema so adapter version mismatches fail loudly. `click` + `rich` for output. `jinja2` + matplotlib's SVG backend for a single-file report. No PyMC, no numpyro, no torch in the core.

---

## 10. Risks, honestly

| Risk | Severity | Mitigation |
|---|---|---|
| **The message is unwelcome.** Core output is "your claim is weaker than you wrote" and "run 37, not 20." Users may install, see one uncomfortable verdict, and uninstall. | High — most likely failure mode | Sequence the pitch **save-then-scold**: lead the README with `doctor` and `lint` (pure time savings, zero emotional cost), let `compare` be found on day three, and ship `watch` in Phase 2 as the sweetener that ends evals *early*. Never editorialize in output — state the number and the required n, nothing more. |
| **Rollouts are not i.i.d. Bernoulli.** Session drift, reused object placements, operator fatigue, and a shared training-run fixed effect make naive intervals anti-conservative — *narrower than the truth*. A rigor tool that certifies a difference that isn't there is fatal. | High | State it **in the first screen of the README, not a risks appendix**. Detect intra-run correlation via ICC when seeds/session ids are available and widen rather than pretend. Print the single-seed warning always. This is a correctness ceiling, not a solved problem, and it is labeled as one. |
| **The statistical core is an afternoon of scipy** — thin as a portfolio artifact and easy to clone. | Medium | The moat is the fixture suite, the version-pinned adapters, the structural refusals, the four-state exit code, and the FPR calibration gate. Budget the fixtures as the product. Accept that the math is commodity and compete on being *correct and finished*. |
| **Adapter and format rot.** LeRobot broke dataset formats three times and renamed import paths and CLI flags across two releases. | Medium-High | Two real adapters only; generic CSV mapper as permanent escape hatch; strict schema-version pinning that fails loudly; data-driven YAML lint rules keyed off `codebase_version`; CI matrix over the two most recent LeRobot releases; never import `lerobot`. |
| **`profile` never earns its verdict.** Experiments D or E fail, or matched-budget public pairs turn out to number three. | Medium (contained) | It is Phase 3, behind `--experimental`, and the MVP does not depend on it. If it fails, the negative result is published as a notebook — which is a genuine contribution, since the naive version of this metric is currently being proposed by multiple people. |
| **A wrong lint threshold deletes good data or passes a corrupt dataset.** Trust dies permanently and publicly. | High | Every rule ships a deliberately-corrupted fixture; every threshold prints its value and its upstream citation; `lint` never mutates a dataset (`--rewrite-stats` writes to `--out`, never in place); five rules with fixtures beat eleven without. |
| **Sequential test silently broken** ⇒ inflated false positives, strictly worse than shipping nothing. | High (contained) | 20,000-null FPR simulation per configuration as a **non-negotiable CI release gate**; two supermartingales at α/2 for two-sided; no variant ships without a passing calibration run. |
| **Incumbent absorption.** LeRobot's 0.7.0 roadmap names a leaderboard and hub-native remote eval; an industry forecast expects policy evaluation to become a product category in 2027. | Medium | Compete on being the thing that works *across* runners rather than the one that owns any; keep the statistical core dependency-free and importable as a library; Apache-2.0 against STEP's non-commercial license is a durable structural advantage. |
| **The author's own repo is the only real fixture.** | Medium | Experiment A fixes this cheaply (n=200, no training). Public LeRobot datasets supply lint fixtures. Accept that harness-diversity comes from community contributions, and build the golden-fixture template that makes contributing a 30-minute job. |

---

## 11. Schedule

| Week | Deliverable | GPU needed |
|---|---|---|
| **0** (2 days) | Experiment A: PushT eval to n=200/policy with logged seeds and partial credit. Experiment B: verify seed exposure in `lerobot-eval`. **No product code.** | yes, inference only |
| **1** | `ingest` (2 adapters + CSV), `plan` (simulated required-N), `compare` (intervals, exact tests, Bonferroni+CLD, Beta posteriors, four-state exit), `doctor` (4 checks). PyPI 0.1.0. **Launch writeup: the author corrects his own published README.** | no |
| **2** | `lint` (DS001/003/005/006/008 + fixtures, SARIF), `manifest`/`diff` with COMPUTE_CONFOUND, `report` (HTML + model card). 0.2.0. | no |
| **3** | Experiment C. `watch` sequential stopping + FPR calibration gate. `seeds` (if Experiment B passed). `gate` + GitHub Action. W&B write-back. 0.3.0. | no |
| **4+** | Experiments D and E. `profile` behind `--experimental` **only if both pass**; the null-calibration and embedding-sweep notebooks published either way. Remaining lint rules with fixtures. Upstream engagement on the v2.1→v3.0 conversion issues. | no |

Something genuinely useful and publishable exists at the end of week 1. Nothing on the critical path requires a GPU after week 0, a robot ever, or a dependency that has broken on Windows.

---

## 12. The one paragraph that sells it

Every eval harness in robotics stops at printing a success rate. Nothing consumes that number. Verdikt is the thing that consumes it: it reads the JSON your harness already wrote, tells you whether checkpoint B is actually better than A at the n you actually ran, tells you how many more rollouts would settle it, refuses to rank two policies whose sample budgets differ by 10×, and hands you an HTML report and a LeRobot-format model card. Before that, it lints the dataset for the five silent misconfigurations that train and eval without erroring and cost weeks. It is Apache-2.0, runs on CPU in seconds, needs no robot, and the first thing its author did with it was point it at his own published benchmark and discover the headline was p=0.056.