# Is it safe to pair episodes by index?

`verdikt compare --paired` treats episode *i* of one run as the same scene as episode *i* of
another. That assumption is worth a lot of statistical power and is silently wrong if the two
evaluations drew different scenes — so this is the measurement behind it.

## What was run

Two evaluations of the **same policy** (ACT on PushT) with identical settings — `--seed 1000`,
`--eval.batch_size 5`, 50 episodes, same checkpoint, same machine — using `lerobot-eval`
0.5.1. If the harness re-draws scenes per run, the two should look unrelated. If it seeds the
environment deterministically, they should look nearly identical.

## Result

| | |
|---|---|
| identical success vector | yes (0/50 in both — see caveat) |
| median &#124;Δ max_reward&#124; | **0.000536** |
| within 0.01 | 31 / 50 episodes |
| within 0.05 | 41 / 50 |
| within 0.20 | 49 / 50 |
| worst episode | 0.688 |

**The median is the informative number.** Had the two runs drawn different initial states, the
per-episode reward would differ by O(0.1–0.5) almost everywhere. Instead half the episodes
agree to about 5e-4. That is the signature of the *same* initial scene, replayed.

The scatter is real too: a handful of episodes diverge substantially, one by 0.69. PushT is
contact-rich, so a float-level difference in an early action gets amplified into a different
outcome. GPU kernel non-determinism is enough to seed that.

## What this justifies, and what it does not

**Justified:** with a fixed `--seed` and an identical `--eval.batch_size`, episode *i* is the
same scene across runs, so pairing by index is meaningful and McNemar is applicable. Batch
size matters because vectorised environments derive per-episode seeds from their slot.

**Not justified:** treating rollouts as reproducible. They are not, on GPU. Two runs of the
same policy on the same scene can reach different outcomes, and that within-pair noise is
genuine — it does not bias McNemar, but it is not zero.

**Caveat on the success vector.** ACT scored 0/50 in both runs, so "identical success vectors"
is two vectors of zeros and proves very little on its own. The reward distribution is what
carries the evidence here. A repeat with a policy that succeeds often would strengthen it.

## Why the flag still exists

None of the above is visible in `eval_info.json` — it records no seed, no batch size, and no
initial state. Verdikt cannot verify from the file that you ran both arms this way, so it
refuses to assume it:

```
paired comparison needs episodes that are known to be the same scene, and this source
records no per-episode seed. re-run the evaluation with a fixed --seed and identical
batch size for both policies, then pass --assume-aligned to confirm you did — verdikt
will not assume it for you.
```

When the harness *does* record a per-episode seed, Verdikt pairs on the seed and no assertion
is needed — including when the arms logged the same scenes in a different order.

## Reproducing

```bash
lerobot-eval --policy.path=<ckpt> --env.type=pusht --eval.n_episodes=50 \
  --eval.batch_size=5 --seed=1000 --output_dir=runs/determinism/run1
# ...identical command, different output dir...
python - <<'PY'
import json, numpy as np, pathlib
m = lambda p: json.loads(pathlib.Path(p).read_text())["per_task"][0]["metrics"]
a, b = m("runs/determinism/run1/eval_info.json"), m("runs/determinism/run2/eval_info.json")
d = np.abs(np.array(a["max_rewards"]) - np.array(b["max_rewards"]))
print("median", np.median(d), "| within 0.01:", int((d < 0.01).sum()), "/", len(d))
PY
```
