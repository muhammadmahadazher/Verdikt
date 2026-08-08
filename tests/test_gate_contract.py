"""The JSON contract the GitHub Action depends on.

`action.yml` parses `verdikt compare --format json` in a shell step, where a renamed field
fails at the worst possible moment - inside somebody else's CI, on a pull request. These tests
pin the field names and the verdict each committed fixture must produce, so the contract
cannot drift silently.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REQUIRED_TOP = {"arms", "pairs", "confounds", "verdict", "reason", "required_n",
                "schema_version", "label_sources"}
REQUIRED_ARM = {"policy_id", "n", "successes", "n_ungraded", "rate", "ci_low", "ci_high",
                "ci_method", "letter"}


def run_compare(pattern: str, baseline: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "verdikt.cli", "compare", str(FIXTURES / pattern),
         "--baseline", baseline, "--format", "json"],
        capture_output=True, text=True,
    )
    if not proc.stdout.strip():
        pytest.fail(f"no JSON on stdout (exit {proc.returncode}): {proc.stderr[:400]}")
    return proc.returncode, json.loads(proc.stdout)


class TestJsonContract:
    def test_top_level_fields_exist(self):
        _code, data = run_compare("pusht_n200/*.json", "diffusion")
        assert REQUIRED_TOP <= set(data), f"missing: {REQUIRED_TOP - set(data)}"

    def test_arm_fields_exist(self):
        _code, data = run_compare("pusht_n200/*.json", "diffusion")
        for arm in data["arms"]:
            assert REQUIRED_ARM <= set(arm), f"missing: {REQUIRED_ARM - set(arm)}"

    def test_exit_code_equals_verdict_in_json_mode(self):
        code, data = run_compare("pusht_n200/*.json", "diffusion")
        assert code == data["verdict"], "the action reads the exit code AND the json; they " \
                                        "must agree"


class TestBraceExpansion:
    """A quoted {a,b} pattern never reaches the shell, and Python's glob has no brace syntax.

    The gate self-test found this the hard way: the action passed
    `pusht_n200/{diffusion,upstream}.json`, glob matched nothing, and the job failed with a
    confusing "no files matched". Users will write this pattern because it works unquoted.
    """

    @pytest.mark.parametrize("pattern,expected", [
        ("a/{x,y}.json", ["a/x.json", "a/y.json"]),
        ("{a,b}/{x,y}.json", ["a/x.json", "a/y.json", "b/x.json", "b/y.json"]),
        ("a/*.json", ["a/*.json"]),
        ("plain.json", ["plain.json"]),
        ("{one}.json", ["one.json"]),
        ("un{balanced.json", ["un{balanced.json"]),
    ])
    def test_expansion(self, pattern, expected):
        from verdikt.cli import _expand_braces

        assert _expand_braces(pattern) == expected

    def test_brace_pattern_works_end_to_end(self):
        code, data = run_compare("pusht_n200/{diffusion,upstream}.json", "diffusion")
        assert code == 0 and data["verdict"] == 0
        assert {a["policy_id"] for a in data["arms"]} == {"diffusion", "upstream"}


class TestGateScenarios:
    """Each scenario in .github/workflows/gate-selftest.yml, asserted here too, so a break
    is caught by pytest rather than only by a red workflow."""

    def test_genuine_improvement_passes(self):
        code, data = run_compare("pusht_n200/[du]*.json", "diffusion")
        assert data["verdict"] == 0 and code == 0

    def test_real_regression_is_caught(self):
        code, data = run_compare("pusht_n200/[ad]*.json", "diffusion")
        assert data["verdict"] == 1 and code == 1

    def test_canonical_underpowered_case(self):
        """35% vs 70% at n=20 - looks decisive, is p=0.056."""
        code, data = run_compare("underpowered_n20/*.json", "candidate")
        assert data["verdict"] == 2 and code == 2
        assert data["required_n"] and data["required_n"] > 20
        assert "not evidence that they are equivalent" in data["reason"]
