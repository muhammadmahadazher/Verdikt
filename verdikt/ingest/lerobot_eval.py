"""Adapter for `lerobot-eval` output (`eval_info.json`), and for vla-on-a-budget results.

Both are the same on-disk shape - vla-on-a-budget's `results/*.json` files ARE lerobot eval
output - so one parser serves both, and the second registration exists only to name the
provenance explicitly in `--adapter` and in the golden fixtures.

Per-episode fields available in this format:
  successes[i]    bool     - the graded outcome
  max_rewards[i]  float    - best coverage reached; usable as partial credit when in [0,1]
  sum_rewards[i]  float    - integrated reward, not normalised, NOT used as progress
Notably absent: any per-episode seed. See `verdikt doctor --check-seeds`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import LabelSource, Rollout
from .base import AdapterError, register


class _LeRobotEvalAdapter:
    name = "lerobot"
    _description = "lerobot-eval eval_info.json (also vla-on-a-budget results/*.json)"

    def sniff(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return isinstance(data, dict) and "per_task" in data and "overall" in data

    def parse(self, path: Path, policy_id: str | None = None) -> list[Rollout]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AdapterError(f"{path}: not readable as JSON ({exc})") from exc

        if "per_task" not in data:
            raise AdapterError(
                f"{path}: no 'per_task' key. this does not look like lerobot-eval output; "
                "if the format changed upstream, pin the version or use --adapter csv."
            )

        pid = policy_id or _infer_policy_id(path)
        run_id = f"{pid}:{path.stem}"
        rollouts: list[Rollout] = []

        for task_block in data["per_task"]:
            task = str(task_block.get("task_group", "unknown"))
            metrics = task_block.get("metrics", {})
            successes = metrics.get("successes")
            max_rewards = metrics.get("max_rewards") or []
            if successes is None:
                raise AdapterError(f"{path}: task {task} has no 'successes' list")

            for i, ok in enumerate(successes):
                progress = None
                if i < len(max_rewards):
                    mr = float(max_rewards[i])
                    # only claim partial credit when the signal is genuinely normalised;
                    # an un-normalised reward masquerading as progress would corrupt every
                    # downstream statistic that uses it.
                    if 0.0 <= mr <= 1.0:
                        progress = mr
                rollouts.append(
                    Rollout(
                        run_id=run_id,
                        policy_id=pid,
                        task=task,
                        suite=str(data.get("suite", task)),
                        episode_idx=i,
                        seed=None,
                        success=bool(ok),
                        progress=progress,
                        label_source=LabelSource.SIMULATOR,
                    )
                )

        if not rollouts:
            raise AdapterError(f"{path}: parsed zero rollouts")
        return rollouts


def _infer_policy_id(path: Path) -> str:
    """Best-effort name: the run directory usually carries the policy, the filename rarely does."""
    stem = path.stem
    if stem in ("eval_info", "results", "eval"):
        parent = path.parent.name
        for prefix in ("eval_", "train_"):
            parent = parent.removeprefix(prefix)
        return parent or stem
    for prefix in ("eval_", "results_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


_adapter = _LeRobotEvalAdapter()
register(_adapter)


class _VlaOnABudgetAdapter(_LeRobotEvalAdapter):
    """Same shape, named separately so provenance is explicit in fixtures and CLI output."""

    name = "vla-on-a-budget"
    _description = "vla-on-a-budget results/*.json (identical shape to lerobot-eval)"


register(_VlaOnABudgetAdapter())
