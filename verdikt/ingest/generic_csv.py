"""The permanent escape hatch.

No adapter will ever exist for every harness, and an adapter written against output its
author has never generated is a liability. So: describe your columns once with --map and
every downstream command works. This is deliberately the most boring code in the project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import LabelSource, Rollout
from .base import AdapterError, register

CANONICAL = {"policy_id", "episode_idx", "success", "progress", "seed", "task", "suite",
             "steps", "wall_clock_s", "label_source", "run_id"}


class _GenericCsvAdapter:
    name = "csv"
    _description = "any CSV/TSV/JSON-lines table, mapped with --map dest=source"

    def sniff(self, path: Path) -> bool:
        return path.suffix.lower() in (".csv", ".tsv", ".jsonl")

    def parse(self, path: Path, policy_id: str | None = None,
              mapping: dict[str, str] | None = None) -> list[Rollout]:
        mapping = mapping or {}
        unknown = set(mapping) - CANONICAL
        if unknown:
            raise AdapterError(
                f"--map targets {sorted(unknown)} are not canonical fields. "
                f"valid targets: {sorted(CANONICAL)}"
            )
        try:
            if path.suffix.lower() == ".jsonl":
                df = pd.read_json(path, lines=True)
            else:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
        except Exception as exc:
            raise AdapterError(f"{path}: unreadable table ({exc})") from exc

        col = {dest: src for dest, src in mapping.items()}
        for field in CANONICAL:
            col.setdefault(field, field)

        missing = [c for f, c in col.items() if f in ("success",) and c not in df.columns]
        if missing and col.get("progress") not in df.columns:
            raise AdapterError(
                f"{path}: no success column (looked for {col['success']!r}) and no progress "
                "column. map one with --map success=<your column>."
            )

        pid_default = policy_id or path.stem
        rollouts: list[Rollout] = []
        def make_getter(row):
            """Bind the row explicitly; a closure over the loop variable is a real footgun."""

            def val(field, default=None):
                c = col[field]
                return row[c] if c in df.columns and not pd.isna(row[c]) else default

            return val

        for i, row in df.iterrows():
            val = make_getter(row)
            success = val("success")
            progress = val("progress")
            rollouts.append(
                Rollout(
                    run_id=str(val("run_id", f"{pid_default}:csv")),
                    policy_id=str(val("policy_id", pid_default)),
                    task=str(val("task", "unknown")),
                    suite=str(val("suite", "unknown")),
                    episode_idx=int(val("episode_idx", i)),
                    seed=None if val("seed") is None else int(val("seed")),
                    success=None if success is None else bool(success),
                    progress=None if progress is None else float(progress),
                    steps=None if val("steps") is None else int(val("steps")),
                    wall_clock_s=None if val("wall_clock_s") is None else float(val("wall_clock_s")),
                    label_source=str(val("label_source", LabelSource.UNKNOWN)),
                )
            )
        if not rollouts:
            raise AdapterError(f"{path}: parsed zero rows")
        return rollouts


register(_GenericCsvAdapter())
