"""Anytime-valid sequential testing: stop the evaluation the moment the answer is in.

Every other command in Verdikt takes claims away. This one gives budget back - it is why the
tool is worth installing rather than merely worth agreeing with.

THE GUARANTEE. A fixed-sample test is only valid if you look once, at the n you committed to.
Peeking at a p-value as episodes arrive and stopping when it dips below 0.05 inflates the
false-positive rate badly - this is the single most common way robotics evaluations mislead
themselves. Here we use a *test martingale* instead: capital starts at 1 and is wagered on
each episode pair. Under the null the capital process is a martingale, so Ville's inequality
bounds the probability that it EVER reaches 1/alpha by alpha - no matter how often you look,
and no matter when you decide to stop.

    P( there exists t : X_t >= 1/alpha )  <=  alpha        (Ville, 1939)

THE BET. On each step we observe one outcome from each arm and wager a fraction lambda of
capital on their difference d = Y_b - Y_a, which lies in [-1, 1]:

    X_{t+1} = X_t * (1 + lambda * d_t)

Under H0 the arms have equal means, so E[d] = 0 and E[X_{t+1} | past] = X_t. Keeping
|lambda| < 1 keeps capital strictly positive. Rather than guess one lambda, we hold a grid of
them and track the average capital: a mixture of martingales is still a martingale, so the
guarantee survives while the grid adapts to whatever effect size actually shows up.

NOTHING HERE IS TRUSTED WITHOUT CALIBRATION. `tests/test_sequential.py` runs the whole
procedure under the null tens of thousands of times and asserts the realised false-positive
rate stays at or below alpha. If that test fails, this module must not ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Wagering grid. Symmetric, so the mixture detects a difference in either direction, and
# bounded away from 1 so capital can never be wiped out by a single observation.
DEFAULT_LAMBDAS = tuple(np.round(np.linspace(-0.85, 0.85, 35), 4))


@dataclass
class SequentialState:
    """The running capital process. Serialisable, so a watch can resume across sessions."""

    alpha: float = 0.05
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    log_capital: np.ndarray = field(default=None, repr=False)
    steps: int = 0
    peak: float = 1.0
    stopped_at: int | None = None

    def __post_init__(self):
        if self.log_capital is None:
            self.log_capital = np.zeros(len(self.lambdas))

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    @property
    def capital(self) -> float:
        """Mixture capital: the average over the wagering grid, computed in log space."""
        m = float(np.max(self.log_capital))
        return float(np.exp(m) * np.mean(np.exp(self.log_capital - m)))

    @property
    def rejected(self) -> bool:
        return self.capital >= self.threshold

    @property
    def evidence(self) -> str:
        """Capital reads directly as evidence: 20x capital is a 0.05-level rejection."""
        c = self.capital
        if c >= self.threshold:
            return f"{c:.1f}x capital - reject at alpha={self.alpha}"
        return f"{c:.2f}x capital - {self.threshold / max(c, 1e-9):.1f}x short of the threshold"


def update(state: SequentialState, y_a: float, y_b: float) -> SequentialState:
    """Observe one outcome from each arm and wager. Outcomes must lie in [0, 1].

    Binary success is the usual case; partial credit (coverage, progress) works too and
    carries strictly more information per episode, which is what shortens the run.
    """
    if not (0.0 <= y_a <= 1.0 and 0.0 <= y_b <= 1.0):
        raise ValueError(f"outcomes must be in [0,1], got {y_a} and {y_b}")
    if state.stopped_at is not None:
        return state

    d = y_b - y_a
    lam = np.asarray(state.lambdas)
    # log1p keeps the product stable over thousands of steps
    state.log_capital = state.log_capital + np.log1p(lam * d)
    state.steps += 1
    state.peak = max(state.peak, state.capital)
    if state.capital >= state.threshold:
        state.stopped_at = state.steps
    return state


def run(stream_a, stream_b, alpha: float = 0.05,
        lambdas: tuple[float, ...] = DEFAULT_LAMBDAS) -> SequentialState:
    """Run the test over two aligned outcome streams, stopping as soon as it can."""
    state = SequentialState(alpha=alpha, lambdas=lambdas)
    for y_a, y_b in zip(stream_a, stream_b, strict=False):
        update(state, float(y_a), float(y_b))
        if state.stopped_at is not None:
            break
    return state


def replay_savings(outcomes_a, outcomes_b, alpha: float = 0.05, trials: int = 2000,
                   seed: int = 0) -> dict:
    """How much of a finished evaluation was actually needed?

    Replays the recorded episodes in many random orders - order matters to a sequential test,
    and a single ordering would be an anecdote. Reports the measured distribution of stopping
    times, never a figure borrowed from a paper with a different effect size.
    """
    a = np.asarray(outcomes_a, dtype=float)
    b = np.asarray(outcomes_b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("no episodes to replay")
    a, b = a[:n], b[:n]

    rng = np.random.default_rng(seed)
    stops: list[int] = []
    for _ in range(trials):
        order = rng.permutation(n)
        state = run(a[order], b[order], alpha=alpha)
        stops.append(state.stopped_at if state.stopped_at is not None else n)

    stops_arr = np.asarray(stops)
    resolved = stops_arr[stops_arr < n]
    return {
        "n_available": int(n),
        "trials": int(trials),
        "stop_rate": float(len(resolved) / trials),
        "median_stop": int(np.median(stops_arr)),
        "p90_stop": int(np.percentile(stops_arr, 90)),
        "mean_saving": float(1 - stops_arr.mean() / n),
        "median_saving": float(1 - np.median(stops_arr) / n),
    }


def false_positive_rate(p: float, n: int, alpha: float = 0.05, trials: int = 20_000,
                        seed: int = 0) -> float:
    """Realised false-positive rate under the null, by simulation.

    This is the release gate, not a diagnostic. A sequential test whose empirical FPR exceeds
    alpha is strictly worse than shipping no sequential test at all, because it manufactures
    confident wrong answers rather than declining to answer.
    """
    rng = np.random.default_rng(seed)
    a = rng.random((trials, n)) < p
    b = rng.random((trials, n)) < p
    lam = np.asarray(DEFAULT_LAMBDAS)

    log_capital = np.zeros((trials, len(lam)))
    fired = np.zeros(trials, dtype=bool)
    threshold = np.log(1.0 / alpha)
    for t in range(n):
        d = (b[:, t].astype(float) - a[:, t].astype(float))[:, None]
        log_capital += np.log1p(lam[None, :] * d)
        m = log_capital.max(axis=1, keepdims=True)
        mixture = m[:, 0] + np.log(np.exp(log_capital - m).mean(axis=1))
        fired |= mixture >= threshold  # "ever crossed", which is what Ville bounds
    return float(fired.mean())
