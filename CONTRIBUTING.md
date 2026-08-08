# Contributing to Verdikt

Thanks for considering it. Verdikt is small on purpose, so contributions that keep it small
are the most valuable ones.

## The highest-value contribution: an adapter

Verdikt ships adapters for `lerobot-eval` output and a generic CSV/JSON mapper. It cannot ship
adapters for harnesses whose output nobody here has ever generated — an adapter written against
imagined output is a liability, not a feature.

If you use a harness Verdikt does not read yet:

1. Run your eval once and keep the output file.
2. Drop it in `tests/fixtures/adapters/<harness>/` together with the canonical table you expect.
3. Write the parser in `verdikt/ingest/<harness>.py` following `lerobot_eval.py` — usually about
   30 minutes.
4. Declare the upstream schema version it parses, and **fail loudly** on anything else.

Silently mis-mapping a field is worse than refusing to parse. Adapters must never guess.

## Rules that are not up for negotiation

These exist because the tool's only value is that it cannot be talked into a weak claim:

- **No point-estimate gates.** No `--min-success`. Ever. Gate on a bound or a margin.
- **No Wald intervals**, no fallbacks to approximations that under-cover.
- **Every printed rate carries `n` and an interval.**
- **Every p-value names its test.**
- **A lint rule ships with a deliberately-corrupted fixture, or it does not ship.** A threshold
  a user cannot verify is a threshold they have to trust, and this project does not ask for
  trust.
- **Anything uncalibrated is labelled `[provisional]`** and stays behind `--experimental` until
  a published calibration says otherwise.

## Development

```bash
git clone https://github.com/muhammadmahadazher/Verdikt && cd Verdikt
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
ruff check verdikt tests
python docs/make_figures.py      # regenerates every figure in the README
```

Tests are the specification. `tests/test_stats.py::TestPower::test_normal_approximation_overpromises`
is the one that would invalidate the project's central claim if it ever passed trivially — do
not weaken it.

## Reporting a statistical error

If you believe Verdikt computes something incorrectly, that is the most important issue you can
file. Please include the numbers, the command, and the value you expected with its source.
Statistical bugs are triaged ahead of everything else, including crashes.

## Scope

Before proposing a feature, check the "What Verdikt deliberately does not do" section of the
README. Several attractive ideas — failure clustering, VRAM planning, leaderboards, projected
success curves — were considered and rejected with reasons. If you want to reopen one, argue
against the recorded reason rather than around it.
