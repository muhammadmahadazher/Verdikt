"""Ingest layer: any harness's output -> one canonical rollout table."""

from . import generic_csv, lerobot_eval  # noqa: F401  (import registers the adapters)
from .base import Adapter, AdapterError, autodetect, available, get, register

__all__ = ["Adapter", "AdapterError", "autodetect", "available", "get", "register"]
