# Changelog

All notable changes to Verdikt are recorded here. Versions follow
[semantic versioning](https://semver.org/); the statistical behaviour of a command is treated
as part of its public API, so a change that would alter a verdict is a breaking change.

## [0.1.0] — unreleased

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
