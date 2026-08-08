"""Preflight for silent failures - the class of bug that trains, evaluates, and lies.

Every check here corresponds to something that costs days and raises no exception. None of
them is clever; all of them are expensive to learn the hard way. A check that cannot run
returns an `info` finding rather than crashing: doctor's whole job is to work on a machine
where something is already broken.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from .schema import Finding


def run_all(train_config: str | Path | None = None,
            dataset_meta: str | Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    findings.append(check_cuda_wheel())
    findings.extend(check_battery())
    findings.extend(check_video_backend())
    if train_config:
        cfg = _load_json(Path(train_config))
        if cfg is None:
            findings.append(Finding(
                rule_id="DR000", severity="warning",
                message=f"could not read train config at {train_config}",
            ))
        else:
            findings.extend(check_rename_map(cfg, dataset_meta))
            findings.extend(check_normalization(cfg, dataset_meta))
    return findings


# ------------------------------------------------------------------ DR001
def check_cuda_wheel() -> Finding:
    """The CPU wheel that silently replaces your CUDA build.

    `pip install lerobot` (and many robotics packages) can resolve to a CPU-only torch.
    Training then runs at roughly 1/50th speed with no error - the single most expensive
    silent failure in this stack.
    """
    try:
        import torch
    except Exception as exc:
        return Finding(rule_id="DR001", severity="info",
                       message="torch is not importable; cannot check the CUDA build",
                       detail=str(exc)[:200])

    cuda_build = getattr(torch.version, "cuda", None)
    has_device = False
    try:
        has_device = torch.cuda.is_available() or _nvidia_present()
    except Exception:
        has_device = _nvidia_present()

    if cuda_build is None and has_device:
        return Finding(
            rule_id="DR001", severity="error",
            message="torch is a CPU-only build but this machine has an NVIDIA GPU",
            detail=f"torch {torch.__version__} was compiled without CUDA. training will run "
                   "on the CPU at a fraction of the speed, and nothing will raise.",
            fix="pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu126 --force-reinstall",
            citation="observed when installing lerobot over an existing CUDA torch",
        )
    if cuda_build is None:
        return Finding(rule_id="DR001", severity="info",
                       message="torch is a CPU build and no NVIDIA GPU was detected",
                       detail="consistent; nothing to fix unless you expected a GPU")
    return Finding(rule_id="DR001", severity="info",
                   message=f"torch {torch.__version__} has a CUDA {cuda_build} build",
                   detail=f"cuda available: {__import__('torch').cuda.is_available()}")


def _nvidia_present() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return n > 0
    except Exception:
        pass
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


# ------------------------------------------------------------------ DR002
def check_battery() -> list[Finding]:
    """Laptop discrete GPUs get powered off on battery, mid-run, with no useful error.

    The failure presents as `No CUDA GPUs are available` and the device vanishing from the
    OS device list - easily mistaken for a driver crash.
    """
    if platform.system() != "Windows":
        return []
    try:
        import ctypes

        class _S(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        s = _S()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return []
        if s.ACLineStatus == 0:
            return [Finding(
                rule_id="DR002", severity="warning",
                message="running on battery: a laptop dGPU may be powered down mid-run",
                detail="OEM power management can de-enumerate the discrete GPU while training. "
                       "CUDA then reports 'No CUDA GPUs are available' and the device leaves "
                       "Device Manager - which looks like a driver crash but is not.",
                fix="plug in the charger before starting a long run",
            )]
    except Exception:
        return []
    return []


# ------------------------------------------------------------------ DR003
def check_video_backend() -> list[Finding]:
    """`torchcodec` missing means a silent fallback to a slower decoder."""
    try:
        import torchcodec  # noqa: F401

        return []
    except Exception:
        return [Finding(
            rule_id="DR003", severity="info",
            message="torchcodec is unavailable; video decoding will fall back to pyav",
            detail="expected on Windows, macOS-Intel and Linux-ARM. correctness is unaffected; "
                   "dataloading is slower.",
        )]


# ------------------------------------------------------------------ DR004
def check_rename_map(cfg: dict, dataset_meta: str | Path | None) -> list[Finding]:
    """Keys in `rename_map` that pass config validation and are never applied to a batch.

    This is what blocks cross-embodiment fine-tuning from a multi-camera base checkpoint: the
    config validates, training starts, and the policy then reports that every image feature is
    missing - or worse, trains on a silently wrong mapping.
    """
    rmap = cfg.get("rename_map") or {}
    if not rmap:
        return []
    findings = [Finding(
        rule_id="DR004", severity="warning",
        message=f"rename_map is set ({len(rmap)} entries) - verify it reaches the batch",
        detail="in some LeRobot releases rename_map is honoured during config validation but "
               "not applied to training batches, so the policy sees the original keys.",
        fix="after one training step, print the batch keys and confirm the renamed keys are "
            "present; if not, adapt the dataset or the policy config instead.",
        citation="observed on lerobot 0.5.1 fine-tuning smolvla_base onto a 1-camera dataset",
    )]
    features = _dataset_features(dataset_meta)
    if features is not None:
        unknown = [src for src in rmap if src not in features]
        if unknown:
            findings.append(Finding(
                rule_id="DR004", severity="error",
                message=f"rename_map sources not present in the dataset: {unknown}",
                detail=f"dataset features are {sorted(features)[:8]}...",
                fix="map from keys the dataset actually has",
            ))
    return findings


# ------------------------------------------------------------------ DR005
def check_normalization(cfg: dict, dataset_meta: str | Path | None) -> list[Finding]:
    """A declared normalisation mode whose statistics do not exist falls back to identity."""
    policy = cfg.get("policy") or {}
    mapping = policy.get("normalization_mapping") or {}
    wants_quantiles = any(str(v).upper().startswith("Q") for v in mapping.values())
    if not wants_quantiles:
        return []

    stats = _dataset_stats(dataset_meta)
    if stats is None:
        return [Finding(
            rule_id="DR005", severity="warning",
            message="policy requests quantile normalisation; dataset stats were not provided",
            detail="pass --dataset-meta <dataset>/meta to verify q01/q99 exist",
            fix="verdikt doctor --train-config <cfg> --dataset-meta <dataset>/meta",
        )]
    missing = [k for k, v in stats.items()
               if isinstance(v, dict) and not ({"q01", "q99"} <= set(v))]
    if missing:
        return [Finding(
            rule_id="DR005", severity="error",
            message="quantile normalisation requested but q01/q99 are missing from stats",
            detail=f"features without quantiles: {missing[:6]}",
            fix="recompute dataset statistics with quantiles, or switch to MEAN_STD",
            citation="silent identity fallback; documented as a common reproducibility trap",
        )]
    return []


# ---------------------------------------------------------------- helpers
def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dataset_features(meta: str | Path | None) -> set[str] | None:
    if not meta:
        return None
    info = _load_json(Path(meta) / "info.json") if Path(meta).is_dir() else _load_json(Path(meta))
    if not info:
        return None
    feats = info.get("features")
    return set(feats) if isinstance(feats, dict) else None


def _dataset_stats(meta: str | Path | None) -> dict | None:
    if not meta:
        return None
    p = Path(meta)
    return _load_json(p / "stats.json") if p.is_dir() else None
