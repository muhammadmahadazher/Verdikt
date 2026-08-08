"""Build tiny LeRobot-shaped datasets, healthy and deliberately corrupted - one per rule.

The generator is the fixture. It is committed, deterministic and readable, so a sceptical
reader can verify what "broken" means for each rule instead of trusting a binary blob. Every
lint rule must fire on its own corruption and stay silent on the healthy baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_EPISODES = 6
FRAMES_PER_EPISODE = 40
FPS = 10


def _trajectories(action_lag: int = 0, seed: int = 0):
    """A position-controlled agent chasing a target: dstate ~ k * (action - state).

    With action_lag != 0 the action stream is shifted, which is exactly the teleop-latency
    fault DS008 exists to find.
    """
    rng = np.random.default_rng(seed)
    states, actions, episodes, frames = [], [], [], []
    for ep in range(N_EPISODES):
        pos = rng.uniform(100, 400, size=2)
        targets = rng.uniform(50, 450, size=(FRAMES_PER_EPISODE, 2))
        ep_state, ep_action = [], []
        for t in range(FRAMES_PER_EPISODE):
            ep_state.append(pos.copy())
            ep_action.append(targets[t])
            pos = pos + 0.45 * (targets[t] - pos) + rng.normal(0, 0.5, size=2)
        ep_state = np.asarray(ep_state)
        ep_action = np.asarray(ep_action)
        if action_lag:
            ep_action = np.roll(ep_action, action_lag, axis=0)
        states.append(ep_state)
        actions.append(ep_action)
        episodes.append(np.full(FRAMES_PER_EPISODE, ep))
        frames.append(np.arange(FRAMES_PER_EPISODE))
    return (np.concatenate(states), np.concatenate(actions),
            np.concatenate(episodes), np.concatenate(frames))


def _stats(state: np.ndarray, action: np.ndarray, mean_shift_sigma: float = 0.0,
           with_quantiles: bool = False) -> dict:
    out = {}
    for name, arr in (("observation.state", state), ("action", action)):
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)
        if name == "observation.state" and mean_shift_sigma:
            mean = mean + mean_shift_sigma * std
        entry = {"mean": mean.tolist(), "std": std.tolist(),
                 "min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
                 "count": [int(arr.shape[0])]}
        if with_quantiles:
            entry["q01"] = np.quantile(arr, 0.01, axis=0).tolist()
            entry["q99"] = np.quantile(arr, 0.99, axis=0).tolist()
        out[name] = entry
    return out


def build(root: Path, *, break_rule: str | None = None) -> Path:
    """Write one dataset. `break_rule` selects which corruption to introduce."""
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    lag = 2 if break_rule == "DS008" else 0
    state, action, episode_index, frame_index = _trajectories(action_lag=lag)
    total_frames = int(state.shape[0])

    pq.write_table(
        pa.table({
            "observation.state": [row.tolist() for row in state],
            "action": [row.tolist() for row in action],
            "episode_index": episode_index.astype("int64"),
            "frame_index": frame_index.astype("int64"),
            "timestamp": (frame_index / FPS).astype("float64"),
            "index": np.arange(total_frames, dtype="int64"),
        }),
        root / "data" / "chunk-000" / "file-000.parquet",
    )

    starts = np.arange(N_EPISODES) * FRAMES_PER_EPISODE
    ends = starts + FRAMES_PER_EPISODE
    if break_rule == "DS003":
        ends[2] -= 7  # leave a hole: ranges no longer tile the frame axis
    pq.write_table(
        pa.table({
            "episode_index": np.arange(N_EPISODES, dtype="int64"),
            "dataset_from_index": starts.astype("int64"),
            "dataset_to_index": ends.astype("int64"),
            "length": (ends - starts).astype("int64"),
        }),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    info = {
        "codebase_version": "v9.9" if break_rule == "DS002" else "v3.0",
        "robot_type": "synthetic",
        "total_episodes": N_EPISODES,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": float(FPS) if break_rule == "DS001" else FPS,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    stats = _stats(state, action,
                   mean_shift_sigma=2.0 if break_rule == "DS005" else 0.0,
                   with_quantiles=(break_rule != "DS006"))
    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return root


QUANTILE_CONFIG = {"policy": {"normalization_mapping": {"STATE": "QUANTILES",
                                                        "ACTION": "QUANTILES"}}}
MEANSTD_CONFIG = {"policy": {"normalization_mapping": {"STATE": "MEAN_STD",
                                                       "ACTION": "MEAN_STD"}}}
