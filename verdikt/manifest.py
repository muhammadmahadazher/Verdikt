"""Run provenance, and the comparability check that arithmetic alone can settle.

`samples_seen = batch_size x grad_accum x steps` is the most useful number nobody records.
Two policies whose sample budgets differ by 10x cannot be ranked as an architecture result:
the budget is a sufficient alternative explanation for any gap you observe. This module
captures that number from a training run and refuses the comparison when it is violated.

Note gradient accumulation is included for completeness but is never a remedy for a sample
deficit: accumulating 4x8 instead of stepping at 32 leaves samples_seen unchanged.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from .schema import RunManifest


def capture(run_dir: str | Path, policy_id: str | None = None) -> RunManifest:
    """Build a manifest from a LeRobot training output directory.

    Reads `checkpoints/last/pretrained_model/train_config.json`, which LeRobot writes for
    every run. Nothing here imports lerobot, so a broken install stays diagnosable.
    """
    run = Path(run_dir)
    cfg_path = _find_config(run)
    if cfg_path is None:
        raise FileNotFoundError(
            f"no train_config.json under {run}. looked in checkpoints/last/pretrained_model/ "
            "and the directory root."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    policy = cfg.get("policy") or {}
    dataset = cfg.get("dataset") or {}
    norm = policy.get("normalization_mapping") or {}

    return RunManifest(
        run_id=str(cfg.get("job_name") or run.name),
        policy_id=policy_id or str(policy.get("type") or run.name),
        policy_type=policy.get("type"),
        dataset_repo_id=dataset.get("repo_id"),
        dataset_revision=dataset.get("revision"),
        normalization_mode=_norm_signature(norm),
        batch_size=cfg.get("batch_size"),
        grad_accum=int(cfg.get("gradient_accumulation_steps") or 1),
        steps=cfg.get("steps"),
        seed=cfg.get("seed"),
        torch_version=_safe_torch_version(),
        gpu_name=_safe_gpu_name(),
        lerobot_version=_safe_lerobot_version(),
        cuda_version=platform.platform(),
    )


def _find_config(run: Path) -> Path | None:
    for candidate in (
        run / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
        run / "train_config.json",
        run / "pretrained_model" / "train_config.json",
    ):
        if candidate.exists():
            return candidate
    hits = sorted(run.glob("**/train_config.json"))
    return hits[0] if hits else None


def _norm_signature(mapping: dict) -> str | None:
    """A stable one-line signature, so a normalisation change is visible in a diff."""
    if not mapping:
        return None
    return ",".join(f"{k}={v}" for k, v in sorted(mapping.items()))


def _safe_torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def _safe_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _safe_lerobot_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("lerobot")
    except Exception:
        return None


# ------------------------------------------------------------------- diff
FIELDS = [
    ("policy_type", "EXPECTED"),
    ("dataset_repo_id", "DATA_CONFOUND"),
    ("dataset_revision", "DATA_CONFOUND"),
    ("normalization_mode", "DATA_CONFOUND"),
    ("batch_size", "CAUSE"),
    ("steps", "CAUSE"),
    ("seed", "EXPECTED"),
    ("lerobot_version", "CAUSE"),
]


def diff(a: RunManifest, b: RunManifest, ratio_limit: float = 2.0) -> list[dict]:
    """Field-by-field comparison, classified by what a difference would mean."""
    rows: list[dict] = []
    for field, kind in FIELDS:
        va, vb = getattr(a, field), getattr(b, field)
        rows.append({
            "field": field,
            "a": "-" if va is None else str(va),
            "b": "-" if vb is None else str(vb),
            "class": "ok" if va == vb else kind,
        })

    sa, sb = a.samples_seen, b.samples_seen
    if sa and sb:
        r = max(sa, sb) / min(sa, sb)
        rows.append({
            "field": "samples_seen",
            "a": f"{sa:.3g}", "b": f"{sb:.3g}",
            "class": "COMPUTE_CONFOUND" if r >= ratio_limit else "ok",
            "note": f"{r:.1f}x",
        })
    else:
        rows.append({"field": "samples_seen", "a": f"{sa or '-'}", "b": f"{sb or '-'}",
                     "class": "unknown",
                     "note": "batch_size or steps missing; comparability cannot be checked"})
    return rows


def comparable(rows: list[dict]) -> bool:
    return not any(r["class"] in ("COMPUTE_CONFOUND", "DATA_CONFOUND") for r in rows)
