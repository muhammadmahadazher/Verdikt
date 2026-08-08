"""Canonical data model. Every command speaks these types, so adapters cannot drift.

The one non-negotiable invariant: a success rate never travels without its `n`. Anything
that carries a rate carries the numerator and denominator that produced it.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"


class Verdict(IntEnum):
    """Verdict states, valued so they can be returned directly as process exit codes."""

    BETTER = 0
    REGRESSION = 1
    UNDERPOWERED = 2
    NOT_COMPARABLE = 3


class LabelSource(str):
    """Where a success label came from. Printed on every line that consumes it."""

    HUMAN = "human"
    SIMULATOR = "simulator"
    SCRIPTED = "scripted"
    UNKNOWN = "unknown"


class Rollout(BaseModel):
    """One episode of one policy. The atom of every statistic in this tool."""

    run_id: str
    policy_id: str
    task: str = "unknown"
    suite: str = "unknown"
    episode_idx: int
    seed: int | None = None
    success: bool | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    steps: int | None = None
    wall_clock_s: float | None = None
    label_source: str = LabelSource.UNKNOWN
    manifest_id: str | None = None

    @model_validator(mode="after")
    def _need_an_outcome(self):
        if self.success is None and self.progress is None:
            raise ValueError(
                f"rollout {self.policy_id}/{self.episode_idx} has neither success nor progress; "
                "there is nothing to measure"
            )
        return self


class ArmSummary(BaseModel):
    """A policy arm, always carrying the evidence behind its rate."""

    policy_id: str
    n: int
    successes: int
    n_ungraded: int = 0  # rollouts present but carrying no success label - never hidden
    rate: float
    ci_low: float
    ci_high: float
    ci_method: str
    one_sided_bound: float | None = None  # populated for 0/n and n/n
    mean_progress: float | None = None
    samples_seen: float | None = None
    letter: str = ""


class PairTest(BaseModel):
    """One pairwise comparison, with the test that produced it named in the record."""

    a: str
    b: str
    test: str
    p_value: float
    alpha_adjusted: float
    significant: bool
    alt_test: str | None = None
    alt_p_value: float | None = None
    suppressed_reason: str | None = None


class Confound(BaseModel):
    """A reason two arms must not be ranked against each other."""

    field: str
    a_value: str
    b_value: str
    ratio: float | None = None
    kind: Literal["EXPECTED", "CAUSE", "COMPUTE_CONFOUND", "DATA_CONFOUND"]
    message: str


class Plan(BaseModel):
    """A pre-registered analysis commitment, hashed before the data is looked at."""

    schema_version: str = SCHEMA_VERSION
    test: str
    alpha: float
    alternative: str
    baseline_rate: float | None = None
    mde: float | None = None
    power: float | None = None
    planned_n: int | None = None
    hypothesis: str = ""
    commitment_hash: str = ""


class RunManifest(BaseModel):
    """Everything needed to decide whether two runs are comparable at all."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    policy_id: str
    policy_type: str | None = None
    dataset_repo_id: str | None = None
    dataset_revision: str | None = None
    dataset_content_hash: str | None = None
    normalization_mode: str | None = None
    batch_size: int | None = None
    grad_accum: int = 1
    steps: int | None = None
    seed: int | None = None
    lerobot_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    peak_vram_gib: float | None = None
    wall_clock_h: float | None = None
    plan_hash: str | None = None

    @property
    def samples_seen(self) -> float | None:
        """batch x grad_accum x steps. The single most useful comparability number.

        Note gradient accumulation is included because it changes the optimisation, but it
        does NOT change samples_seen relative to a larger batch - so it is never a remedy
        for a sample deficit.
        """
        if self.batch_size is None or self.steps is None:
            return None
        return float(self.batch_size) * float(self.grad_accum) * float(self.steps)


class Finding(BaseModel):
    """A lint or doctor result. Carries its own citation so a threshold can be audited."""

    rule_id: str
    severity: Literal["error", "warning", "info"]
    message: str
    detail: str = ""
    citation: str = ""
    fix: str = ""
    location: str = ""


class ComparisonResult(BaseModel):
    """The full output of `verdikt compare` - serialisable, so CI can consume it."""

    schema_version: str = SCHEMA_VERSION
    arms: list[ArmSummary]
    pairs: list[PairTest]
    confounds: list[Confound] = []
    verdict: Verdict
    reason: str
    required_n: int | None = None
    plan: Plan | None = None
    label_sources: list[str] = []
    notes: list[str] = []  # advisory remarks, e.g. pairing that did not pay off
