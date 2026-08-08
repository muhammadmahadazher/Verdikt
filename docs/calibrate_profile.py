"""Experiment D: does the multimodality test hold its false-positive rate on unimodal data?

`verdikt profile` ships only if it does. The data here is unimodal BY CONSTRUCTION - actions
are a smooth function of state plus symmetric noise - so every neighbourhood flagged as
multimodal is a false positive. The nominal rate is alpha = 0.05; the ship condition is <= 0.07
in every cell, including the heavy-tailed ones where a Gaussian null is known to collapse.

    python docs/calibrate_profile.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verdikt.profile import multimodal_fraction, neighbourhoods  # noqa: E402

SHIP_THRESHOLD = 0.07
ALPHA = 0.05


def unimodal_dataset(n_episodes=40, ep_len=60, obs_dim=2, act_dim=2, noise="gaussian",
                     scale=0.25, seed=0):
    """Episodic data where the action is a smooth function of state plus symmetric noise.

    There is exactly one mode per state by construction, so any detection is a false alarm.
    """
    rng = np.random.default_rng(seed)
    states, actions, episodes = [], [], []
    weights = rng.normal(size=(obs_dim, act_dim))
    for ep in range(n_episodes):
        pos = rng.uniform(-1, 1, size=obs_dim)
        traj = []
        for _ in range(ep_len):
            pos = pos + rng.normal(0, 0.12, size=obs_dim)   # smooth trajectory
            traj.append(pos.copy())
        traj = np.asarray(traj)
        mean_action = np.tanh(traj @ weights)
        if noise == "gaussian":
            eps = rng.normal(0, scale, size=mean_action.shape)
        else:
            df = int(noise[2:-1])                            # "t(3)" -> 3
            eps = rng.standard_t(df, size=mean_action.shape) * scale / np.sqrt(df / (df - 2))
        states.append(traj)
        actions.append(mean_action + eps)
        episodes.append(np.full(ep_len, ep))
    return (np.concatenate(states), np.concatenate(actions), np.concatenate(episodes))


def measure(noise, blocked, k, seed, permutations=99, sample=220):
    states, actions, episodes = unimodal_dataset(noise=noise, seed=seed)
    if not blocked:
        # the failure mode we are guarding against: neighbours drawn from the same trajectory
        episodes = np.zeros_like(episodes)
        block_radius = 0
    else:
        block_radius = 15
    rng = np.random.default_rng(seed + 1000)
    _anchors, nbrs = neighbourhoods(states, episodes, k=k, block_radius=block_radius,
                                    sample=sample, rng=rng)
    if len(nbrs) < 30:
        return float("nan")
    return multimodal_fraction(actions, nbrs, permutations=permutations, alpha=ALPHA, rng=rng)


def main() -> None:
    print("EXPERIMENT D - false-positive rate on unimodal-by-construction data")
    print(f"nominal alpha = {ALPHA};  ship condition = every cell <= {SHIP_THRESHOLD}\n")
    print(f"{'noise':>10s} {'blocking':>10s} {'k':>4s} {'FPR':>8s}   verdict")

    rows, failures = [], 0
    for noise in ("gaussian", "t(8)", "t(5)", "t(3)"):
        for blocked in (True, False):
            for k in (16, 24):
                vals = [measure(noise, blocked, k, seed) for seed in (0, 1, 2)]
                fpr = float(np.nanmean(vals))
                ok = fpr <= SHIP_THRESHOLD
                failures += not ok
                rows.append((noise, blocked, k, fpr, ok))
                label = "episode" if blocked else "NONE"
                print(f"{noise:>10s} {label:>10s} {k:>4d} {fpr:>8.4f}   "
                      f"{'pass' if ok else 'FAIL'}")

    blocked_rows = [r for r in rows if r[1]]
    print(f"\nworst cell with episode blocking : {max(r[3] for r in blocked_rows):.4f}")
    print(f"worst cell without blocking      : {max(r[3] for r in rows if not r[1]):.4f}")
    print(f"\n{failures} of {len(rows)} cells exceed the ship threshold")
    print("SHIP" if not any(not r[4] for r in blocked_rows) else "DO NOT SHIP")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[{time.time() - t0:.0f}s]")
