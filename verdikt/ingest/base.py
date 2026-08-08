"""Adapter contract.

Adapters are the only place that knows about someone else's file format, and they are the
first thing to rot: LeRobot has changed its dataset format three times and renamed CLI flags
across two releases. So an adapter declares the schema shape it understands and *fails loudly*
on anything it does not recognise. Silently mis-mapping a field is worse than refusing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schema import Rollout


class AdapterError(RuntimeError):
    """Raised when input does not match what this adapter promises to parse."""


class Adapter(Protocol):
    name: str

    def sniff(self, path: Path) -> bool:
        """Cheap check: does this file look like mine?"""

    def parse(self, path: Path, policy_id: str | None = None) -> list[Rollout]:
        """Convert one file into rollouts, or raise AdapterError."""


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    _REGISTRY[adapter.name] = adapter
    return adapter


def get(name: str) -> Adapter:
    if name not in _REGISTRY:
        raise AdapterError(f"unknown adapter {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def autodetect(path: Path) -> Adapter:
    """Pick an adapter by content, never by file extension."""
    for adapter in _REGISTRY.values():
        try:
            if adapter.sniff(path):
                return adapter
        except Exception:  # a sniff must never crash the run
            continue
    raise AdapterError(
        f"no adapter recognises {path.name}. use --adapter to force one, or --adapter csv "
        "with --map to describe your columns explicitly."
    )
