# Changelog

All notable changes to Verdikt are recorded here. Versions follow
[semantic versioning](https://semver.org/); the statistical behaviour of a command is treated
as part of its public API, so a change that would alter a verdict is a breaking change.

## [0.4.0] — 2026-08-09

Multi-task evaluation. A suite of tasks is not one number, and pooling one can reverse the
answer.

### Added

- **Per-task comparison, always on.** Any run spanning more than one task is now compared
  within each task as well as pooled. There is no flag to enable this; `--by-task` only
  controls whether the breakdown prints. A check the caller has to remember is not a check.
- **Cochran-Mantel-Haenszel** to combine per-task comparisons without letting the number of
  episodes each task received leak into the result, and **Breslow-Day** to test first whether
  the effect is even the same across tasks. Both agree with statsmodels 0.14.6 to 1e-6 across
  seven configurations (`docs/crosscheck_stratified.py`). Where statsmodels returns NaN — it
  divides by (OR − 1), so a common odds ratio of exactly 1 is undefined for it — the expected
  values are derived by hand in the tests that pin them.
- **`compare --by-task`** prints the per-task table, the stratified test and the homogeneity
  test. It is printed unconditionally when a contradiction was found, since the breakdown is
  the evidence for the refusal.

### Changed

- **A pooled result that no task supports is now `NOT COMPARABLE`.** On a two-task suite where
  the arms received different episode counts, one policy can lead the pooled rate by 29 points
  while tying one task and losing the other. That pair is suppressed rather than ranked,
  exactly as a compute or data confound is. This changes verdicts on multi-task inputs, hence
  the minor bump.
- **Task coverage gaps are reported.** Episodes run on a task the other arm never attempted
  have nothing to pair against and no stratified test can repair them. Above 5% of an arm's
  episodes this suppresses the comparison; below it, a stray crashed episode is tolerated.
- Degrees of freedom for Breslow-Day count only tasks that carry information. Padding a suite
  with tasks nobody solved would otherwise make it look progressively more homogeneous.
- Wrapped output now adapts to the terminal width. At 80 columns — where redirected output and
  CI logs land — long confound messages were being wrapped twice and broken mid-clause.

### Fixed

- **The Bayesian posterior no longer contradicts a `NOT COMPARABLE` verdict.** Up to and
  including 0.3.1, a suppressed pair still printed `P(b > a) = 1.000` directly beneath the
  refusal to rank it — the exact conclusion the verdict withheld, in the most quotable form on
  the page. It is now withheld for suppressed pairs specifically; unconfounded arms in the same
  run keep theirs. Affects `compare`, `report` and the demo site.
- `pairwise` no longer prints a corrected alpha when no comparison was testable; `m=0` beside a
  threshold invited checking a p-value that does not exist.

## [0.3.1] — 2026-08-08

### Fixed

- **`--paired` no longer implies that pairing is always stronger.** McNemar spends only the
  discordant pairs while an unpaired test uses both full margins, so when two arms share few
  successes there is nothing for pairing to cancel and the unpaired test wins. Measured on 50
  real PushT episodes (ACT 0/50 vs diffusion 13/50): Fisher p=0.00010, McNemar p=0.00024.
  Verdikt now reports the unpaired p-value alongside the paired one and prints a `PAIRING`
  note when the contingency table shows pairing did not pay off. The 0.3.0 README claimed
  pairing made the comparison "much sharper" unconditionally; that was wrong and is corrected.

## [0.3.0] — 2026-08-08

### Added

- **`compare --paired`** — McNemar on episodes that are the same scene in both arms. Pairing
  discards the scenes nobody solves and the scenes everybody solves, so it separates policies
  an unpaired test cannot at the same n.
- **`--assume-aligned`** — required to pair by episode index when the source records no
  per-episode seed. Verdikt pairs on the seed when one exists (including when the arms logged
  the same scenes in a different order) and otherwise refuses rather than guessing.
- **`docs/pairing_evidence.md`** — the measurement behind that assumption. Two `lerobot-eval`
  runs of the same policy at the same seed and batch size agree to a median of 5e-4 in
  per-episode reward, which is the signature of the same scene replayed; different scenes
  would differ by O(0.1). Rollouts are not bit-reproducible on GPU, and one episode in fifty
  diverged substantially — noise that is real but does not bias McNemar.

## [0.2.0] — 2026-08-08

Four new dataset rules, bringing `lint` to ten. Each ships with a deliberately-corrupted
fixture, and all ten stay silent on the real 206-episode `lerobot/pusht` dataset.

### Added

- **DS004 — shard integrity.** Every shard an episode points at must exist, and every shard
  must be pointed at. A dangling reference fails partway through an epoch, once the loader
  reaches it; an orphaned shard is quieter and worse — the frames sit on disk and are
  silently never trained on.
- **DS007 — timestamp monotonicity and frame rate.** Catches frames concatenated out of
  order, and datasets whose real spacing disagrees with the declared `fps` (which silently
  scales every velocity and action-chunk duration).
- **DS009 — dead dimensions.** An action or state dimension that never changes: a broken
  sensor, a gripper that was never actuated, or a dimension that does not belong in the space.
- **DS010 — action saturation.** Commands pinned at their own recorded extremes, the
  signature of a clipped action space. Skips dimensions with few distinct values, because a
  binary gripper is legitimately always at an extreme and flagging it would be a false alarm.

### Changed

- Rules that read frame data now read **every shard**, not the first. A fault confined to
  shard three is exactly the kind that survives a spot check and then costs a training run.

## [0.1.0] — 2026-08-08

Published to PyPI as `verdikt-eval`: `pip install verdikt-eval`.

First release. Nine commands, all reading files that already exist on disk.

### Commands

- **`doctor`** — preflight for silent failures: a CPU-only torch build on a CUDA machine, a
  `rename_map` that validates but never reaches the batch, quantile normalisation without
  quantile statistics, laptop dGPU power-gating, and a missing `torchcodec`.
- **`lint`** — six dataset rules (fps typing, codebase version, episode-index integrity,
  statistics drift in σ units, per-feature normalisation feasibility, state/action alignment),
  each with a deliberately-corrupted fixture. Emits SARIF. Never imports `lerobot`, so it
  still works when the training stack is broken.
- **`ingest`** — any harness's eval output into one canonical rollout table.
- **`plan`** — required-N and minimum detectable effect, computed by exact enumeration
  *through the test that will issue the verdict*.
- **`compare`** — intervals, exact tests, multiplicity correction, compact letter display,
  Bayesian posteriors, confound suppression, and a four-state verdict.
- **`manifest` / `diff`** — run provenance and the `samples_seen` comparability check.
- **`watch`** — anytime-valid sequential stopping via a test martingale.
- **`report`** — self-contained HTML plus a LeRobot-format model card, with optional W&B
  write-back.
- **`profile`** *(experimental)* — a bound on the action variance a deterministic policy
  cannot explain.

### Structural refusals

- A success rate is never rendered without `n` and an interval.
- `0/n` and `n/n` render as exact one-sided bounds.
- No `--min-success`; gating is on a confidence bound or a non-inferiority margin.
- The Wald interval is not implemented, and asking for it explains why.
- Confounded arms are suppressed rather than ranked, and no flag overrides that.
- A pre-registered plan blocks changing the test after seeing the data.

### Verification

- **Power**: the boundary-sweep optimisation is proven identical to brute-force enumeration
  across 140 configurations; unconditional tests are routed to full enumeration because their
  two-sided p-values are not monotone.
- **Sequential stopping**: 20 000 null simulations per configuration across four base rates,
  two α levels and runs to 600 episodes. Realised false-positive rate 0.0014–0.0336.
- **Multimodality profile**: 16 calibration cells on unimodal-by-construction data across
  Gaussian/t(8)/t(5)/t(3) noise. Worst false-positive rate 0.067 against a 0.07 ship
  threshold.
- Ships an 800-rollout PushT corpus so these are demonstrated on real robot policies.

### Known limitations

- Rollouts are treated as independent Bernoulli trials. Session drift and reused object
  placements induce correlation that makes naive intervals narrower than the truth. Stated in
  the README rather than buried here.
- `profile` is provisional: it is not calibrated against downstream policy success, and it
  reports a bound under the embeddings supplied, never an architecture recommendation.
- The W&B `push()` network call is thin and untested against the live API; the payload it
  sends is covered by unit tests.
- Two eval adapters ship (`lerobot-eval` and this project's own results), plus a generic
  CSV/JSON mapper. Adapters for other harnesses need a real output file to be written safely.
