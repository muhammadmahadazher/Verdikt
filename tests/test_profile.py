"""Calibration and refusal behaviour for the experimental multimodality profile.

The dangerous failure here is not a crash - it is a confident number that a reader repeats in
a design review. So the tests are mostly about the tool declining to speak: on unimodal data
it must not cry multimodality, and when two views of the same frames disagree it must refuse
to issue a dataset-level claim rather than average them into a plausible-looking figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from verdikt.profile import (
    multimodal_fraction,
    neighbourhoods,
    participation_ratio,
    profile,
    profile_embedding,
)

ALPHA = 0.05
SHIP_THRESHOLD = 0.07


def unimodal(n_episodes=25, ep_len=50, noise="gaussian", scale=0.25, seed=0):
    """Actions are a smooth function of state plus symmetric noise: one mode per state."""
    rng = np.random.default_rng(seed)
    states, actions, episodes = [], [], []
    w = rng.normal(size=(2, 2))
    for ep in range(n_episodes):
        pos = rng.uniform(-1, 1, size=2)
        traj = []
        for _ in range(ep_len):
            pos = pos + rng.normal(0, 0.12, size=2)
            traj.append(pos.copy())
        traj = np.asarray(traj)
        mean = np.tanh(traj @ w)
        if noise == "gaussian":
            eps = rng.normal(0, scale, size=mean.shape)
        else:
            df = int(noise[2:-1])
            eps = rng.standard_t(df, size=mean.shape) * scale / np.sqrt(df / (df - 2))
        states.append(traj)
        actions.append(mean + eps)
        episodes.append(np.full(ep_len, ep))
    return np.concatenate(states), np.concatenate(actions), np.concatenate(episodes)


def bimodal(n_episodes=25, ep_len=50, seed=0):
    """From each state, two genuinely competing actions - go left or go right."""
    rng = np.random.default_rng(seed)
    states, actions, episodes = [], [], []
    for ep in range(n_episodes):
        pos = rng.uniform(-1, 1, size=2)
        traj, act = [], []
        for _ in range(ep_len):
            pos = pos + rng.normal(0, 0.12, size=2)
            traj.append(pos.copy())
            branch = 1.0 if rng.random() < 0.5 else -1.0
            act.append(np.array([branch * 1.5, 0.0]) + rng.normal(0, 0.15, size=2))
        states.append(np.asarray(traj))
        actions.append(np.asarray(act))
        episodes.append(np.full(ep_len, ep))
    return np.concatenate(states), np.concatenate(actions), np.concatenate(episodes)


def fraction_for(states, actions, episodes, seed=0, k=16, permutations=99, sample=150):
    rng = np.random.default_rng(seed)
    _anchors, nbrs = neighbourhoods(states, episodes, k=k, block_radius=15, sample=sample,
                                    rng=rng)
    if len(nbrs) < 30:
        pytest.skip("not enough neighbourhoods survive blocking")
    return multimodal_fraction(actions, nbrs, permutations=permutations, alpha=ALPHA, rng=rng)


class TestCalibration:
    """On data that is unimodal by construction, every detection is a false positive.

    This is a fast REGRESSION GUARD, not the ship gate. A single small sample estimates the
    rate with a standard error near 0.02, so asserting the 0.07 ship threshold here would
    fail on noise alone and teach everyone to ignore it. The real gate is
    `docs/calibrate_profile.py`: 16 cells, three seeds each, which measured 0.038 - 0.067 with
    episode blocking. What this test catches is catastrophic miscalibration - the Gaussian-null
    version of this statistic reaches 0.47 under t(3), and would fail here by a mile.
    """

    GUARD = 0.15  # ~5 standard errors above the measured rate

    @pytest.mark.parametrize("noise", ["gaussian", "t(5)", "t(3)"])
    def test_false_positive_rate_does_not_collapse_under_heavy_tails(self, noise):
        rates = [fraction_for(*unimodal(noise=noise, seed=s), seed=s, permutations=49,
                              sample=180) for s in (1, 2)]
        mean_rate = float(np.mean(rates))
        assert mean_rate <= self.GUARD, (
            f"{noise}: false-positive rate {mean_rate:.3f} indicates the null is broken; "
            "run docs/calibrate_profile.py for the full picture"
        )


class TestPower:
    def test_genuine_bimodality_is_detected(self):
        states, actions, episodes = bimodal(seed=2)
        assert fraction_for(states, actions, episodes, seed=2) > 0.30

    def test_bimodal_data_has_high_dispersion(self):
        states, actions, episodes = bimodal(seed=3)
        res = profile_embedding(states, actions, episodes, "state", sample=150,
                                permutations=49, seed=3)
        uni_states, uni_actions, uni_eps = unimodal(seed=3)
        uni = profile_embedding(uni_states, uni_actions, uni_eps, "state", sample=150,
                                permutations=49, seed=3)
        assert res.amr_l2 > uni.amr_l2


class TestEmbeddingGuards:
    def test_participation_ratio_tracks_effective_dimension(self):
        rng = np.random.default_rng(0)
        flat = rng.normal(size=(500, 8)) @ np.diag([1, 0.01, 0.01, 0.01, 0, 0, 0, 0])
        assert participation_ratio(flat) < 2.0
        assert participation_ratio(rng.normal(size=(500, 8))) > 6.0

    def test_high_dimensional_embedding_is_refused(self):
        rng = np.random.default_rng(0)
        states, actions, episodes = unimodal(seed=0)
        wide = rng.normal(size=(len(states), 80))  # neighbours are meaningless here
        res = profile_embedding(wide, actions, episodes, "noise", sample=120,
                                permutations=19, seed=0)
        assert not res.trustworthy
        assert any("dimensionality" in n for n in res.notes)


class TestRefusal:
    def test_single_embedding_is_never_enough(self):
        states, actions, episodes = unimodal(seed=4)
        _res, verdict, why = profile({"only": states}, actions, episodes, sample=120,
                                     permutations=19, seed=4)
        assert verdict == "INSUFFICIENT EVIDENCE"
        assert "two trustworthy embeddings" in why

    def test_disagreeing_embeddings_refuse_a_dataset_claim(self):
        """The real case: lerobot/pusht reads 15% multimodal through position alone and 4.5%
        once velocity is added. Averaging those would invent a fact."""
        states, actions, episodes = unimodal(seed=5)
        rng = np.random.default_rng(5)
        embeddings = {
            "informative": states,
            "uninformative": rng.normal(size=states.shape) * 0.01,  # carries no locality
        }
        _res, verdict, _why = profile(embeddings, actions, episodes, sample=120,
                                      permutations=19, seed=5)
        assert verdict == "INSUFFICIENT EVIDENCE"

    def test_agreeing_embeddings_produce_a_bound_labelled_as_such(self):
        states, actions, episodes = unimodal(seed=6)
        embeddings = {"a": states, "b": states + 1e-6}  # same information, different values
        _res, verdict, why = profile(embeddings, actions, episodes, sample=120,
                                     permutations=19, seed=6)
        assert verdict == "BOUND"
        assert "not a prediction of success rate" in why
