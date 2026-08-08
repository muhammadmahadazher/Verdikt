"""Dataset integrity rules - the failures that train, evaluate, and never raise.

Deliberate constraint: this module **never imports lerobot**. It reads `meta/*.json` and the
parquet files with pyarrow directly, so it still works on a machine where the training stack
is broken - which is exactly when you need it.

Every rule states its threshold and its provenance. A threshold a user cannot audit is a
threshold they have to take on faith, and this tool does not ask for faith.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .schema import Finding

# ------------------------------------------------------------------ thresholds
SIGMA_TOLERANCE = 0.10       # DS005: |recomputed - stored| mean, in units of stored sigma
STD_LOG_TOLERANCE = 0.10     # DS005: |log(recomputed std / stored std)|, ~10% scale error
ALIGNMENT_LAGS = 5           # DS008: search window for state/action lag, in frames
SUPPORTED_CODEBASE = ("v2.0", "v2.1", "v3.0")


@dataclass
class DatasetView:
    """Everything the rules need, read once. Parsing is defensive by design."""

    root: Path
    info: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    episodes: object | None = None
    episode_files: int = 0
    data_files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def codebase_version(self) -> str:
        return str(self.info.get("codebase_version", "unknown"))


def load(root: str | Path) -> DatasetView:
    root = Path(root)
    view = DatasetView(root=root)

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        view.errors.append(f"no meta/info.json under {root}")
        return view
    try:
        view.info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        view.errors.append(f"meta/info.json is not valid JSON: {exc}")
        return view

    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        try:
            view.stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception as exc:
            view.errors.append(f"meta/stats.json is not valid JSON: {exc}")

    # Episode metadata rolls over into multiple chunk files on larger datasets. Reading only
    # the first one makes every later episode look missing, which fires DS003 on a perfectly
    # healthy dataset - the worst possible failure for a linter.
    ep_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if ep_files:
        try:
            import pyarrow as pa

            tables = [pq.read_table(p) for p in ep_files]
            view.episodes = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
            view.episode_files = len(ep_files)
        except Exception as exc:
            view.errors.append(f"episodes parquet unreadable: {exc}")

    view.data_files = sorted((root / "data").glob("**/*.parquet"))
    return view


# ------------------------------------------------------------------ DS001
def ds001_fps(view: DatasetView) -> list[Finding]:
    """`fps` declared as a float where an int is expected (30.0 vs 30).

    Downstream code compares fps by identity in places; a float that prints identically
    fails the comparison and produces confusing timestamp-tolerance errors.
    """
    fps = view.info.get("fps")
    if fps is None:
        return [Finding(rule_id="DS001", severity="error", message="meta/info.json has no fps",
                        location="meta/info.json")]
    if isinstance(fps, float) and not fps.is_integer():
        return [Finding(rule_id="DS001", severity="warning",
                        message=f"fps is fractional ({fps})",
                        detail="fractional frame rates are legal but frequently indicate a "
                               "conversion error; timestamp tolerance checks may misfire.",
                        location="meta/info.json")]
    if isinstance(fps, float):
        return [Finding(
            rule_id="DS001", severity="warning",
            message=f"fps is stored as a float ({fps}) where an int is conventional",
            detail="LeRobot compares fps values in several places; a float that renders the "
                   "same as an int can fail an equality check.",
            fix=f'set "fps": {int(fps)} in meta/info.json',
            citation="huggingface/lerobot fps type mismatch reports",
            location="meta/info.json")]
    if fps <= 0:
        return [Finding(rule_id="DS001", severity="error", message=f"fps is {fps}",
                        location="meta/info.json")]
    return [Finding(rule_id="DS001", severity="info", message=f"fps {fps} ({type(fps).__name__})")]


# ------------------------------------------------------------------ DS002
def ds002_version(view: DatasetView) -> list[Finding]:
    """An unrecognised codebase_version means every other rule is guessing."""
    v = view.codebase_version
    if v not in SUPPORTED_CODEBASE:
        return [Finding(
            rule_id="DS002", severity="warning",
            message=f"codebase_version {v!r} is not one this build was tested against",
            detail=f"tested: {', '.join(SUPPORTED_CODEBASE)}. rules that depend on layout may "
                   "not apply. verdikt refuses to guess rather than mis-report.",
            location="meta/info.json")]
    return [Finding(rule_id="DS002", severity="info", message=f"codebase_version {v}")]


# ------------------------------------------------------------------ DS003
def ds003_episode_index(view: DatasetView) -> list[Finding]:
    """Episode index ranges must be monotone, contiguous, and match the actual row count.

    When `dataset_from_index` / `dataset_to_index` disagree with the parquet, the dataloader
    silently serves padded or scrambled frames - training proceeds and the policy learns from
    garbage past the first few episodes.
    """
    if view.episodes is None:
        return [Finding(rule_id="DS003", severity="info",
                        message="no episodes parquet (pre-v3 layout); skipped")]
    cols = view.episodes.column_names
    if "dataset_from_index" not in cols or "dataset_to_index" not in cols:
        return [Finding(rule_id="DS003", severity="info",
                        message="episodes parquet has no dataset_from/to_index; skipped")]

    lo = np.asarray(view.episodes["dataset_from_index"])
    hi = np.asarray(view.episodes["dataset_to_index"])
    out: list[Finding] = []

    if np.any(hi <= lo):
        bad = int(np.argmax(hi <= lo))
        out.append(Finding(
            rule_id="DS003", severity="error",
            message=f"episode {bad} has an empty or inverted frame range "
                    f"[{lo[bad]}, {hi[bad]})",
            location="meta/episodes"))

    order = np.argsort(lo)
    gaps = lo[order][1:] - hi[order][:-1]
    if np.any(gaps != 0):
        n_gaps = int(np.count_nonzero(gaps))
        first = int(np.argmax(gaps != 0))
        out.append(Finding(
            rule_id="DS003", severity="error",
            message=f"episode frame ranges are not contiguous ({n_gaps} discontinuities)",
            detail=f"first gap after episode index {first}: ends at {hi[order][first]}, "
                   f"next starts at {lo[order][first + 1]}",
            fix="re-run the dataset conversion; do not train on this until it is contiguous",
            citation="v2.1 -> v3.0 conversion index scrambling reports",
            location="meta/episodes"))

    declared = view.info.get("total_frames")
    covered = int(hi.max()) if len(hi) else 0
    if declared is not None and covered != declared:
        out.append(Finding(
            rule_id="DS003", severity="error",
            message=f"episode ranges cover {covered} frames but info.json declares {declared}",
            location="meta/info.json"))

    actual = _parquet_rows(view)
    if actual is not None and declared is not None and actual != declared:
        out.append(Finding(
            rule_id="DS003", severity="error",
            message=f"data parquet holds {actual} rows but info.json declares {declared}",
            fix="the metadata and the data disagree; regenerate the metadata",
            location="data/"))

    n_declared = view.info.get("total_episodes")
    if n_declared is not None and view.episodes.num_rows != n_declared:
        out.append(Finding(
            rule_id="DS003", severity="error",
            message=f"episodes parquet has {view.episodes.num_rows} rows but info.json "
                    f"declares {n_declared} episodes",
            location="meta/"))

    return out or [Finding(rule_id="DS003", severity="info",
                           message=f"episode indexing consistent across "
                                   f"{view.episodes.num_rows} episodes")]


# ------------------------------------------------------------------ DS005
def ds005_stats(view: DatasetView, features: list[str] | None = None) -> list[Finding]:
    """Recompute normalisation statistics from the data and diff against what is stored.

    Stored statistics that do not match the data are the quietest catastrophic bug in this
    stack: every prediction is shrunk or shifted, the loss still falls, and nothing errors.
    The classic cause is computing stats over the whole dataset and then training on a subset.
    """
    if not view.stats or not view.data_files:
        return [Finding(rule_id="DS005", severity="info",
                        message="no stats.json or no data parquet; skipped")]

    targets = features or [k for k in ("observation.state", "action") if k in view.stats]
    if not targets:
        return [Finding(rule_id="DS005", severity="warning",
                        message="stats.json contains no observation.state or action entry",
                        detail=f"present keys: {sorted(view.stats)[:8]}")]

    out: list[Finding] = []
    for feat in targets:
        stored = view.stats.get(feat) or {}
        if "mean" not in stored or "std" not in stored:
            out.append(Finding(rule_id="DS005", severity="warning",
                               message=f"{feat}: stats.json has no mean/std"))
            continue
        recomputed = _streaming_moments(view, feat)
        if recomputed is None:
            out.append(Finding(rule_id="DS005", severity="info",
                               message=f"{feat}: not present in the data parquet; skipped"))
            continue
        mean_r, std_r = recomputed
        mean_s = np.asarray(stored["mean"], dtype=float).ravel()
        std_s = np.asarray(stored["std"], dtype=float).ravel()
        if mean_s.shape != mean_r.shape or std_s.shape != mean_r.shape:
            out.append(Finding(
                rule_id="DS005", severity="error",
                message=f"{feat}: stats.json shapes do not match the data",
                detail=f"stored mean {mean_s.shape}, stored std {std_s.shape}, data "
                       f"{mean_r.shape}. verdikt will not broadcast these together - a "
                       "silently broadcast comparison would report a meaningless sigma.",
                location="meta/stats.json"))
            continue

        # Dimensions with (near-)zero stored sigma cannot be expressed in sigma units. They
        # are counted and reported rather than quietly dropped from the "matches" claim.
        checkable = std_s > 1e-12
        n_skipped = int((~checkable).sum())
        mean_drift = np.full(mean_r.shape, np.nan)
        mean_drift[checkable] = np.abs(mean_r - mean_s)[checkable] / std_s[checkable]
        # A stored std that is wrong is just as damaging as a wrong mean: it rescales every
        # prediction. Compare it as a ratio, since it has no natural sigma unit.
        std_ratio = np.full(std_r.shape, np.nan)
        std_ratio[checkable] = std_r[checkable] / std_s[checkable]

        worst_mean = float(np.nanmax(mean_drift)) if checkable.any() else 0.0
        worst_std = float(np.nanmax(np.abs(np.log(std_ratio[checkable])))) if checkable.any() \
            else 0.0
        skipped_note = (f" ({n_skipped} of {mean_r.size} dimensions have zero stored sigma and "
                        "could not be checked)" if n_skipped else "")

        if checkable.any() and worst_mean > SIGMA_TOLERANCE:
            dim = int(np.nanargmax(mean_drift))
            out.append(Finding(
                rule_id="DS005", severity="error",
                message=f"{feat}: stored mean disagrees with the data by {worst_mean:.2f} sigma",
                detail=f"worst dimension {dim}: stored {mean_s[dim]:.4g}, recomputed "
                       f"{mean_r[dim]:.4g} (sigma {std_s[dim]:.4g}). every prediction will be "
                       f"shifted by this amount and nothing will raise.{skipped_note}",
                fix="recompute meta/stats.json over exactly the episodes you train on",
                citation=f"threshold: {SIGMA_TOLERANCE} sigma",
                location="meta/stats.json"))
        elif checkable.any() and worst_std > STD_LOG_TOLERANCE:
            dim = int(np.nanargmax(np.abs(np.log(std_ratio))))
            out.append(Finding(
                rule_id="DS005", severity="error",
                message=f"{feat}: stored std disagrees with the data by a factor of "
                        f"{std_ratio[dim]:.2f}",
                detail=f"worst dimension {dim}: stored {std_s[dim]:.4g}, recomputed "
                       f"{std_r[dim]:.4g}. normalised inputs will be scaled wrongly, which "
                       f"shrinks or amplifies every gradient.{skipped_note}",
                fix="recompute meta/stats.json over exactly the episodes you train on",
                citation=f"threshold: |log ratio| > {STD_LOG_TOLERANCE}",
                location="meta/stats.json"))
        elif not checkable.any():
            out.append(Finding(
                rule_id="DS005", severity="warning",
                message=f"{feat}: every stored sigma is zero, so nothing could be verified",
                detail="a constant feature is legal but unusual; confirm it is intended."))
        else:
            out.append(Finding(
                rule_id="DS005", severity="info",
                message=f"{feat}: stored stats match the data (mean drift {worst_mean:.3f} "
                        f"sigma, std ratio within {np.exp(worst_std):.3f}x)"
                        f"{skipped_note}"))
    return out


# ------------------------------------------------------------------ DS006
def ds006_normalization(view: DatasetView, train_config: dict | None = None) -> list[Finding]:
    """A normalisation mode whose statistics are absent falls back to identity, silently."""
    modes = {}
    if train_config:
        modes = (train_config.get("policy") or {}).get("normalization_mapping") or {}
    if not modes:
        available = _stat_keys(view)
        has_q = "q01" in available and "q99" in available
        return [Finding(
            rule_id="DS006", severity="info",
            message=("stats.json provides quantiles (q01/q99)" if has_q
                     else "stats.json provides mean/std only - QUANTILES normalisation would "
                          "fall back to identity"),
            detail=f"available statistics: {sorted(available)}",
            fix=None if has_q else "pass --train-config to check against what your policy asks for")]

    out: list[Finding] = []
    for group, mode in modes.items():
        need = {"MEAN_STD": {"mean", "std"}, "MIN_MAX": {"min", "max"},
                "QUANTILES": {"q01", "q99"}}.get(str(mode).upper())
        if not need:
            continue
        # Check each feature separately. Pooling the keys across all features hides the case
        # that actually happens: most features carry quantiles and one does not, so the mode
        # silently falls back to identity for exactly that feature.
        for feature, entry in sorted(view.stats.items()):
            if not isinstance(entry, dict):
                continue
            if not _feature_in_group(feature, group):
                continue
            missing = need - set(entry)
            if missing:
                out.append(Finding(
                    rule_id="DS006", severity="error",
                    message=f"{group} requests {mode} but {feature} lacks {sorted(missing)}",
                    detail="LeRobot falls back to identity normalisation when the required "
                           "statistics are absent for a feature; training proceeds, that "
                           "feature is left unnormalised, and nothing raises.",
                    fix=f"recompute statistics for {feature} including {sorted(missing)}, or "
                        f"switch {group} to a mode your statistics support",
                    location="meta/stats.json"))
    return out or [Finding(rule_id="DS006", severity="info",
                           message="every declared normalisation mode has the statistics it "
                                   "needs, for every feature it covers")]


def _feature_in_group(feature: str, group: str) -> bool:
    """Map a LeRobot feature key onto the normalisation group that governs it."""
    g = group.upper()
    if g == "VISUAL":
        return "image" in feature or "pixels" in feature
    if g == "STATE":
        return feature.startswith("observation.state") or feature.endswith(".state")
    if g == "ACTION":
        return feature == "action" or feature.startswith("action")
    return False


# ------------------------------------------------------------------ DS008
def ds008_alignment(view: DatasetView) -> list[Finding]:
    """State/action temporal alignment: is the action at frame t the one that produced t+1?

    Uncompensated teleoperation latency, or an off-by-one in a converter, shifts the action
    stream relative to the state stream. The policy then learns to predict the action that
    was already executed - a fault that looks like "the policy is sluggish".
    """
    if not view.data_files:
        return [Finding(rule_id="DS008", severity="info", message="no data parquet; skipped")]
    try:
        import pyarrow as pa

        cols = ["observation.state", "action", "episode_index"]
        # Read every data file: a verdict drawn from the first shard alone would be reported
        # as a dataset-wide claim while covering only part of the data.
        parts = [pq.read_table(p, columns=cols) for p in view.data_files]
        table = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
    except Exception as exc:
        return [Finding(rule_id="DS008", severity="info",
                        message=f"state/action columns unavailable; skipped ({exc})")]

    state = _to_matrix(table["observation.state"])
    action = _to_matrix(table["action"])
    ep = np.asarray(table["episode_index"])
    if state is None or action is None or state.shape[0] < 4 * ALIGNMENT_LAGS:
        return [Finding(rule_id="DS008", severity="info", message="insufficient data; skipped")]

    dim = min(state.shape[1], action.shape[1])
    # dstate[i] = state[i+1] - state[i], paired with action[i] - the action TAKEN AT state i.
    # Pairing with action[i+1] instead produces a spurious lag on every healthy dataset.
    same_ep = ep[1:] == ep[:-1]
    dstate = np.diff(state[:, :dim], axis=0)[same_ep]
    act = action[:-1, :dim][same_ep]
    st = state[:-1, :dim][same_ep]

    # Two candidate drive signals. Position-control action spaces (action is a target pose)
    # drive velocity through the error term, not through the raw action; velocity-control
    # spaces drive it directly. Try both and report which one explains the data.
    candidates = {"action": act}
    if action.shape[1] == state.shape[1]:
        candidates["action - state (position control)"] = act - st

    best = {"lag": 0, "corr": -2.0, "signal": "action"}
    curves: dict[str, dict[int, float]] = {}
    for name, signal in candidates.items():
        curve = {}
        for lag in range(-ALIGNMENT_LAGS, ALIGNMENT_LAGS + 1):
            rolled = np.roll(signal, lag, axis=0)
            trim = slice(ALIGNMENT_LAGS, -ALIGNMENT_LAGS)
            curve[lag] = _mean_corr(dstate[trim], rolled[trim])
        curves[name] = curve
        lag_best = max(curve, key=lambda k: curve[k])
        if curve[lag_best] > best["corr"]:
            best = {"lag": lag_best, "corr": curve[lag_best], "signal": name}

    if best["corr"] < 0.10:
        return [Finding(
            rule_id="DS008", severity="info",
            message=f"state/action correlation is weak at every lag (max {best['corr']:.3f})",
            detail="alignment cannot be assessed for this action space. this is not evidence "
                   "of a fault - some action spaces simply do not drive state linearly.")]

    if abs(best["lag"]) == ALIGNMENT_LAGS:
        return [Finding(
            rule_id="DS008", severity="warning",
            message=f"alignment peaks at the edge of the search window (lag {best['lag']:+d})",
            detail=f"the true lag may be beyond +/-{ALIGNMENT_LAGS} frames, so verdikt will "
                   "not name one. correlation "
                   f"{best['corr']:.3f} using '{best['signal']}'.",
            fix="inspect the alignment manually over a wider window before training",
            citation="a boundary peak is inconclusive by construction")]

    if best["lag"] != 0:
        return [Finding(
            rule_id="DS008", severity="warning",
            message=f"state/action alignment peaks at lag {best['lag']:+d}, expected 0",
            detail=f"correlation {best['corr']:.3f} using '{best['signal']}' "
                   f"(lag 0 gives {curves[best['signal']][0]:.3f}). probable uncompensated "
                   "teleop latency or an off-by-one in conversion: the policy would be trained "
                   "to predict an action that has already been executed.",
            fix=f"shift the action stream by {-best['lag']:+d} frames, or fix the converter",
            citation=f"searched lags {-ALIGNMENT_LAGS}..{ALIGNMENT_LAGS}")]

    return [Finding(
        rule_id="DS008", severity="info",
        message=f"state/action aligned at lag 0 (correlation {best['corr']:.3f})",
        detail=f"drive signal: {best['signal']}")]


# ---------------------------------------------------------------- engine
RULES = {
    "DS001": ds001_fps,
    "DS002": ds002_version,
    "DS003": ds003_episode_index,
    "DS005": ds005_stats,
    "DS006": ds006_normalization,
    "DS008": ds008_alignment,
}


def run_all(root: str | Path, train_config: dict | None = None) -> list[Finding]:
    view = load(root)
    if view.errors:
        return [Finding(rule_id="DS000", severity="error", message=e) for e in view.errors]

    findings: list[Finding] = []
    for rule_id, fn in RULES.items():
        try:
            if rule_id == "DS006":
                findings.extend(fn(view, train_config))
            else:
                findings.extend(fn(view))
        except Exception as exc:  # a broken rule must not hide the others
            findings.append(Finding(
                rule_id=rule_id, severity="warning",
                message=f"rule crashed and was skipped: {type(exc).__name__}: {exc}"))
    return findings


def to_sarif(findings: list[Finding], tool_version: str = "0.1.0") -> dict:
    """SARIF 2.1.0, so GitHub code scanning can display dataset findings like code findings."""
    level = {"error": "error", "warning": "warning", "info": "note"}
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "verdikt", "version": tool_version,
                                "informationUri": "https://github.com/muhammadmahadazher/Verdikt"}},
            "results": [{
                "ruleId": f.rule_id,
                "level": level.get(f.severity, "note"),
                "message": {"text": f.message + (f"\n{f.detail}" if f.detail else "")},
                "locations": [{"physicalLocation": {
                    "artifactLocation": {"uri": f.location or "meta/info.json"}}}],
            } for f in findings if f.severity != "info"],
        }],
    }


# ---------------------------------------------------------------- helpers
def _parquet_rows(view: DatasetView) -> int | None:
    """Row count across every data shard. Handles are closed explicitly: on Windows an open
    parquet handle blocks deleting or moving the dataset directory afterwards."""
    total = 0
    try:
        for p in view.data_files:
            with pq.ParquetFile(p) as pf:
                total += pf.metadata.num_rows
        return total
    except Exception:
        return None


def _stat_keys(view: DatasetView) -> set[str]:
    keys: set[str] = set()
    for v in view.stats.values():
        if isinstance(v, dict):
            keys |= set(v)
    return keys


def _to_matrix(column) -> np.ndarray | None:
    """Parquet list-column -> 2-D float array, tolerant of scalar columns."""
    try:
        arr = column.to_pylist()
    except Exception:
        return None
    if not arr:
        return None
    if isinstance(arr[0], (list, tuple)):
        return np.asarray(arr, dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1, 1)


def _streaming_moments(view: DatasetView, feature: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-dimension mean and std over all row groups, without loading the dataset at once."""
    count = 0
    mean = None
    m2 = None
    for path in view.data_files:
        try:
            handle = pq.ParquetFile(path)
        except Exception:
            continue
        with handle as pf:  # closed explicitly; an open handle locks the directory on Windows
            if feature not in pf.schema_arrow.names:
                return None
            for batch in pf.iter_batches(columns=[feature], batch_size=8192):
                block = _to_matrix(batch[feature])
                if block is None:
                    continue
                if mean is None:
                    mean = np.zeros(block.shape[1])
                    m2 = np.zeros(block.shape[1])
                for row in block:  # Welford, numerically stable for long streams
                    count += 1
                    delta = row - mean
                    mean += delta / count
                    m2 += delta * (row - mean)
    if mean is None or count < 2:
        return None
    return mean, np.sqrt(m2 / (count - 1))


def _mean_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-dimension Pearson correlation, ignoring constant dimensions."""
    corrs = []
    for j in range(a.shape[1]):
        x, y = a[:, j], b[:, j]
        sx, sy = x.std(), y.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        c = float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
        if math.isfinite(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else -2.0
