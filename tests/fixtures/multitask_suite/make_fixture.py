"""Generate the multi-task fixture: a pooled win that no task supports.

The scenario is not contrived, it is the ordinary shape of an evaluation that drifted. Two
checkpoints are run over a two-task suite. Partway through, someone decides the interesting
failures are on `pick_bowl` and gives the new checkpoint more episodes there - and gives the
old one more episodes on `stack_blocks`, because that is where it was being debugged.

Nobody cheated and no number is wrong. But the arms now have different task mixes, and the
pooled success rate says the new checkpoint gained 29 points when it did not gain anywhere:

    task            act_v1            act_v2
    pick_bowl       14/20  (70%)      55/80  (69%)     v2 slightly worse
    stack_blocks    16/80  (20%)       4/20  (20%)     identical
    ----------------------------------------------------------------
    pooled          30/100 (30%)      59/100 (59%)     "v2 wins by 29 points"

Deterministic: the successes are placed by position, not sampled, so the fixture is stable
across platforms and Python versions.
"""

from __future__ import annotations

import csv
from pathlib import Path

# task, policy, successes, n
CELLS = [
    ("pick_bowl", "act_v1", 14, 20),
    ("pick_bowl", "act_v2", 55, 80),
    ("stack_blocks", "act_v1", 16, 80),
    ("stack_blocks", "act_v2", 4, 20),
]


def main() -> None:
    out = Path(__file__).parent / "suite.csv"
    rows = []
    for task, policy, successes, n in CELLS:
        for i in range(n):
            rows.append({
                "run_id": f"{policy}-{task}",
                "policy_id": policy,
                "task": task,
                "suite": "bench",
                "episode_idx": i,
                "success": int(i < successes),
            })
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rollouts)")


if __name__ == "__main__":
    main()
