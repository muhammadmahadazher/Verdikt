"""Every lint rule must fire on its own corruption and stay silent on a healthy dataset.

The second half of that sentence is the one that matters. A rule that flags a correct dataset
destroys trust permanently, and DS008's first implementation did exactly that on the canonical
`lerobot/pusht` dataset before the pairing was corrected.
"""

from __future__ import annotations

import pytest

from verdikt.lint import run_all, to_sarif

from .dataset_fixtures import MEANSTD_CONFIG, QUANTILE_CONFIG, build


def severities(findings, rule_id):
    return {f.severity for f in findings if f.rule_id == rule_id}


@pytest.fixture(scope="module")
def healthy(tmp_path_factory):
    return build(tmp_path_factory.mktemp("healthy"))


class TestHealthyDataset:
    """No false positives. This is the expensive half of the contract."""

    def test_no_errors_at_all(self, healthy):
        findings = run_all(healthy, MEANSTD_CONFIG)
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], f"false positives on a healthy dataset: {errors}"

    def test_no_warnings_at_all(self, healthy):
        findings = run_all(healthy, MEANSTD_CONFIG)
        warnings = [f for f in findings if f.severity == "warning"]
        assert warnings == [], f"false warnings on a healthy dataset: {warnings}"

    def test_alignment_is_clean_at_lag_zero(self, healthy):
        findings = run_all(healthy, MEANSTD_CONFIG)
        ds008 = [f for f in findings if f.rule_id == "DS008"]
        assert len(ds008) == 1
        assert ds008[0].severity == "info"
        assert "lag 0" in ds008[0].message


class TestCorruptions:
    def test_ds001_float_fps(self, tmp_path):
        findings = run_all(build(tmp_path / "d", break_rule="DS001"), MEANSTD_CONFIG)
        assert "warning" in severities(findings, "DS001")

    def test_ds002_unknown_version(self, tmp_path):
        findings = run_all(build(tmp_path / "d", break_rule="DS002"), MEANSTD_CONFIG)
        assert "warning" in severities(findings, "DS002")

    def test_ds003_non_contiguous_episodes(self, tmp_path):
        findings = run_all(build(tmp_path / "d", break_rule="DS003"), MEANSTD_CONFIG)
        assert "error" in severities(findings, "DS003")

    def test_ds005_shifted_stats(self, tmp_path):
        """A 2-sigma mean shift silently biases every prediction; nothing else would catch it."""
        findings = run_all(build(tmp_path / "d", break_rule="DS005"), MEANSTD_CONFIG)
        assert "error" in severities(findings, "DS005")
        msg = next(f for f in findings if f.rule_id == "DS005" and f.severity == "error").message
        assert "sigma" in msg

    def test_ds006_quantiles_without_quantile_stats(self, tmp_path):
        findings = run_all(build(tmp_path / "d", break_rule="DS006"), QUANTILE_CONFIG)
        assert "error" in severities(findings, "DS006")

    def test_ds006_passes_when_stats_support_the_mode(self, healthy):
        findings = run_all(healthy, QUANTILE_CONFIG)
        assert "error" not in severities(findings, "DS006")

    def test_ds008_detects_a_shifted_action_stream(self, tmp_path):
        findings = run_all(build(tmp_path / "d", break_rule="DS008"), MEANSTD_CONFIG)
        assert "warning" in severities(findings, "DS008")
        msg = next(f for f in findings if f.rule_id == "DS008").message
        assert "lag" in msg


class TestEngine:
    def test_missing_dataset_reports_cleanly(self, tmp_path):
        findings = run_all(tmp_path / "nothing-here")
        assert len(findings) == 1 and findings[0].rule_id == "DS000"

    def test_sarif_is_wellformed_and_omits_info(self, tmp_path):
        findings = run_all(build(tmp_path / "d", break_rule="DS003"), MEANSTD_CONFIG)
        sarif = to_sarif(findings)
        assert sarif["version"] == "2.1.0"
        results = sarif["runs"][0]["results"]
        assert results and all(r["level"] in ("error", "warning") for r in results)

    def test_a_crashing_rule_does_not_hide_the_others(self, healthy, monkeypatch):
        import verdikt.lint as lint_mod

        def boom(_view):
            raise RuntimeError("synthetic failure")

        monkeypatch.setitem(lint_mod.RULES, "DS001", boom)
        findings = run_all(healthy, MEANSTD_CONFIG)
        assert any(f.rule_id == "DS001" and "crashed" in f.message for f in findings)
        assert any(f.rule_id == "DS003" for f in findings)
