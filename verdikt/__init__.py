"""Verdikt - a decision layer for robot-policy evaluation.

Reads the eval JSON, dataset files and run configs you already have, and refuses to let you
draw a conclusion the data does not support.
"""

__version__ = "0.3.1"

from .schema import ArmSummary, ComparisonResult, Plan, Rollout, RunManifest, Verdict

__all__ = [
           "ArmSummary",
           "ComparisonResult",
           "Plan",
           "Rollout",
           "RunManifest",
           "Verdict",
           "__version__",
]
