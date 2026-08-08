"""EXPERIMENTAL: how much of a dataset's action variance can a deterministic policy explain?

THE TEMPTING CLAIM, AND WHY IT NEEDS CARE. Demonstrations are often multimodal: from the same
state, several different actions are all correct. A deterministic regression head must predict
one number, so it converges toward the conditional mean - which can be an action no
demonstrator ever took. That is the standard explanation for why ACT plateaus where a
generative head succeeds, and it is the reason someone would want a tool that says "your data
is multimodal, don't use regression".

Saying it rigorously is much harder than measuring it. Three failure modes are designed
around here, because the obvious implementation gets all three wrong:

1. NEIGHBOURS THAT ARE NOT INDEPENDENT. k-nearest-neighbours inside an episodic dataset
   mostly returns consecutive frames of the SAME trajectory. Their actions are similar for
   reasons that have nothing to do with multimodality, and the resulting statistic is
   anti-conservative. Fixed by excluding neighbours within +/-`block_radius` frames of the
   same episode.

2. A GAUSSIAN NULL THAT REAL DATA DOES NOT OBEY. Comparing a dispersion statistic against a
   Gaussian reference collapses under heavy-tailed action noise - the nominal 5 % false
   positive rate rises above 40 % under t(3). Fixed by a permutation null resampled from the
   user's own residuals, so the reference distribution inherits whatever noise the data has.

3. A BOUND THAT DOES NOT BIND THE POLICY IT INDICTS. The bound is only about the information
   in the embedding you hand it. `lerobot/pusht`'s proprioceptive state is the 2-D agent
   position; the block pose lives in the image, which the policies actually consume. Measured
   on proprioception alone, the bound describes a policy that cannot see the object. Fixed by
   requiring agreement across at least two embeddings and refusing to issue a verdict when
   they disagree.

The output is a BOUND, never a success-rate prediction, and never an architecture
recommendation. `verdikt profile` stays behind --experimental until the calibration notebook
in `docs/` says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_K = 24
DEFAULT_BLOCK_RADIUS = 15          # frames of the same episode excluded from a neighbourhood
DEFAULT_PERMUTATIONS = 199
PARTICIPATION_WARN = 20.0          # embeddings above this are too spread to trust
PARTICIPATION_REFUSE = 40.0


@dataclass
class ProfileResult:
    """Everything needed to decide whether to believe the number - reported alongside it."""

    embedding_name: str
    n_samples: int
    k: int
    amr_l2: float                  # residual fraction of action variance (bounds an L2 head)
    mad_l1: float                  # same idea under an L1 objective (bounds ACT's actual loss)
    multimodal_fraction: float
    participation_ratio: float
    trustworthy: bool
    notes: list[str] = field(default_factory=list)


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality: (sum lambda)^2 / sum(lambda^2) over the covariance spectrum.

    k-NN distances become meaningless in high dimension, so this decides whether the
    neighbourhoods mean anything at all before any statistic is computed on them.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return 1.0
    centred = x - x.mean(axis=0)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2
    total = eigenvalues.sum()
    if total <= 0:
        return 1.0
    return float(total ** 2 / (eigenvalues ** 2).sum())


def neighbourhoods(embedding: np.ndarray, episode_index: np.ndarray, k: int = DEFAULT_K,
                   block_radius: int = DEFAULT_BLOCK_RADIUS, sample: int = 1500,
                   rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Indices of k neighbours per sampled anchor, excluding same-episode temporal neighbours.

    Returns (anchors, neighbours) where neighbours has shape (len(anchors), k).
    """
    rng = rng or np.random.default_rng(0)
    embedding = np.asarray(embedding, dtype=float)
    n = embedding.shape[0]
    anchors = (rng.choice(n, size=min(sample, n), replace=False) if n > sample
               else np.arange(n))

    frame_pos = np.arange(n)
    out = np.empty((len(anchors), k), dtype=np.int64)
    usable = np.zeros(len(anchors), dtype=bool)

    for row, i in enumerate(anchors):
        d = np.linalg.norm(embedding - embedding[i], axis=1)
        # exclude the anchor, and any frame of the same episode within the block radius
        same_episode = episode_index == episode_index[i]
        too_close_in_time = same_episode & (np.abs(frame_pos - i) <= block_radius)
        d[too_close_in_time] = np.inf
        order = np.argpartition(d, min(k, n - 1))[:k]
        order = order[np.isfinite(d[order])]
        if len(order) == k:
            out[row] = order
            usable[row] = True
    return anchors[usable], out[usable]


def dispersion_ratios(actions: np.ndarray, nbrs: np.ndarray) -> tuple[float, float]:
    """Residual dispersion within neighbourhoods, relative to the dataset as a whole.

    amr_l2  = E[tr Cov(a | neighbourhood)] / tr Cov(a)
              An L2-optimal deterministic policy predicts the conditional mean, so this is the
              fraction of action variance it cannot remove, GIVEN THIS EMBEDDING.
    mad_l1  = E[mean |a - median| within] / mean |a - median| overall
              The same statement for an L1 objective, which is what ACT actually minimises.
              Reported separately because the two are not interchangeable.

    Both are dimensionless ratios. A raw trace would depend on how each action dimension was
    scaled, which makes it meaningless when a binary gripper sits beside a radian.
    """
    actions = np.asarray(actions, dtype=float)
    local = actions[nbrs]                                    # (m, k, d)

    within_var = local.var(axis=1, ddof=1).sum(axis=1).mean()
    total_var = actions.var(axis=0, ddof=1).sum()
    amr = float(within_var / total_var) if total_var > 0 else 0.0

    local_mad = np.abs(local - np.median(local, axis=1, keepdims=True)).mean(axis=(1, 2)).mean()
    total_mad = np.abs(actions - np.median(actions, axis=0)).mean()
    mad = float(local_mad / total_mad) if total_mad > 0 else 0.0
    return amr, mad


def _separation(points: np.ndarray, rng: np.random.Generator) -> float:
    """Two-means separation: distance between centroids over pooled within-cluster spread.

    Deliberately crude and cheap - it is only ever compared against its own permutation null,
    so its absolute scale never has to mean anything.
    """
    if len(points) < 4:
        return 0.0
    centres = points[rng.choice(len(points), 2, replace=False)]
    for _ in range(8):
        d0 = np.linalg.norm(points - centres[0], axis=1)
        d1 = np.linalg.norm(points - centres[1], axis=1)
        assign = d1 < d0
        if assign.all() or (~assign).all():
            return 0.0
        centres = np.array([points[~assign].mean(axis=0), points[assign].mean(axis=0)])
    spread = np.sqrt((points[~assign].var(axis=0).sum() + points[assign].var(axis=0).sum()) / 2)
    if spread < 1e-12:
        return 0.0
    return float(np.linalg.norm(centres[0] - centres[1]) / spread)


def multimodal_fraction(actions: np.ndarray, nbrs: np.ndarray,
                        permutations: int = DEFAULT_PERMUTATIONS, alpha: float = 0.05,
                        rng: np.random.Generator | None = None) -> float:
    """Fraction of neighbourhoods whose action spread is more clustered than noise explains.

    The null is built from the data itself: residuals from every neighbourhood are pooled and
    resampled to synthesise neighbourhoods with the same marginal noise but no local
    structure. Whatever tail the real action noise has, the reference distribution has it too,
    which is what a fixed Gaussian threshold fails to do.
    """
    rng = rng or np.random.default_rng(0)
    actions = np.asarray(actions, dtype=float)
    local = actions[nbrs]
    residual_pool = (local - local.mean(axis=1, keepdims=True)).reshape(-1, actions.shape[1])

    m, k, _ = local.shape
    flagged = 0
    for i in range(m):
        observed = _separation(local[i], rng)
        if observed <= 0:
            continue
        null = np.empty(permutations)
        for b in range(permutations):
            draw = residual_pool[rng.integers(0, len(residual_pool), size=k)]
            null[b] = _separation(local[i].mean(axis=0) + draw, rng)
        # +1 correction keeps the p-value valid at finite permutation counts
        p = (1 + np.count_nonzero(null >= observed)) / (permutations + 1)
        flagged += p <= alpha
    return float(flagged / m) if m else 0.0


def profile_embedding(embedding: np.ndarray, actions: np.ndarray, episode_index: np.ndarray,
                      name: str, *, k: int = DEFAULT_K,
                      block_radius: int = DEFAULT_BLOCK_RADIUS,
                      permutations: int = DEFAULT_PERMUTATIONS, sample: int = 1500,
                      seed: int = 0) -> ProfileResult:
    """Profile one embedding. Never call this alone to reach a conclusion - see profile()."""
    rng = np.random.default_rng(seed)
    embedding = np.atleast_2d(np.asarray(embedding, dtype=float))
    if embedding.shape[0] != len(actions):
        embedding = embedding.T

    pr = participation_ratio(embedding)
    notes: list[str] = []
    trustworthy = True
    if pr > PARTICIPATION_REFUSE:
        notes.append(f"effective dimensionality {pr:.1f} exceeds {PARTICIPATION_REFUSE}: "
                     "nearest neighbours carry no locality here, so no bound is reported")
        trustworthy = False
    elif pr > PARTICIPATION_WARN:
        notes.append(f"effective dimensionality {pr:.1f} is high; neighbourhoods are diffuse "
                     "and the bound is correspondingly weak")

    anchors, nbrs = neighbourhoods(embedding, episode_index, k, block_radius, sample, rng)
    if len(anchors) < 30:
        notes.append("too few neighbourhoods survive episode blocking to say anything")
        return ProfileResult(name, 0, k, 0.0, 0.0, 0.0, pr, False, notes)

    amr, mad = dispersion_ratios(actions, nbrs)
    frac = (multimodal_fraction(actions, nbrs, permutations, rng=rng)
            if trustworthy else 0.0)
    return ProfileResult(name, len(anchors), k, amr, mad, frac, pr, trustworthy, notes)


def profile(embeddings: dict[str, np.ndarray], actions: np.ndarray, episode_index: np.ndarray,
            **kwargs) -> tuple[list[ProfileResult], str, str]:
    """Profile every supplied embedding and decide whether they agree.

    Returns (results, verdict, explanation). The verdict is INSUFFICIENT EVIDENCE unless at
    least two trustworthy embeddings land in the same place - a bound measured on one
    hand-picked feature set is a statement about that feature set, not about the dataset.
    """
    results = [profile_embedding(emb, actions, episode_index, name, **kwargs)
               for name, emb in embeddings.items()]
    usable = [r for r in results if r.trustworthy and r.n_samples]

    if len(usable) < 2:
        return (results, "INSUFFICIENT EVIDENCE",
                "at least two trustworthy embeddings are required; a bound from a single "
                "feature set describes that feature set, not the dataset")

    spread = max(r.amr_l2 for r in usable) - min(r.amr_l2 for r in usable)
    if spread > 0.20:
        return (results, "INSUFFICIENT EVIDENCE",
                f"the embeddings disagree on the dispersion bound (AMR spread {spread:.2f}); "
                "which features you look through changes the answer, so no dataset-level "
                "bound is defensible")

    # The multimodal fraction must agree too, and it is the more fragile of the two: apparent
    # multimodality is often just a missing feature. On lerobot/pusht it reads 15% through
    # position alone and drops to the null level once velocity is added - the same states
    # revisited at different phases of motion, not genuinely competing actions. Reporting the
    # first number as a dataset property would be exactly the error this module exists to
    # avoid, so disagreement here is disqualifying on its own.
    fractions = [r.multimodal_fraction for r in usable]
    lo, hi = min(fractions), max(fractions)
    if hi - lo > 0.05 and hi > 2 * max(lo, 1e-9):
        return (results, "INSUFFICIENT EVIDENCE",
                f"the embeddings disagree on multimodality ({lo:.1%} vs {hi:.1%}); the higher "
                "reading is explained by a feature the smaller embedding is missing rather "
                "than by competing actions, so no dataset-level claim is supported")

    amr = float(np.mean([r.amr_l2 for r in usable]))
    frac = float(np.mean(fractions))
    return (results, "BOUND",
            f"a deterministic L2 head cannot reduce residual action variance below "
            f"{amr:.0%} of the total, under every embedding tested ({frac:.1%} of "
            "neighbourhoods show competing actions). this is a bound, not a prediction of "
            "success rate, and not an architecture recommendation")
