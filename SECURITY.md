# Security Policy

## Scope

Verdikt reads local files (JSON, CSV, Parquet, YAML) and writes local files. It makes no
network requests in the core package, executes no user-supplied code, and requires no
credentials. Optional extras (`hub`, `wandb`) talk to their respective services using
credentials you have already configured for those tools.

## Reporting a vulnerability

Please open a [GitHub Security Advisory](https://github.com/muhammadmahadazher/Verdikt/security/advisories/new)
rather than a public issue. Expect an initial response within seven days.

Relevant classes of issue:

- crafted input files causing arbitrary code execution or path traversal
- unsafe deserialisation in an adapter
- dependency vulnerabilities reachable from Verdikt's code paths

## A note on statistical correctness

A wrong number in this tool is a bug with real consequences — someone may ship a policy on the
strength of it. Statistical errors are treated with the same priority as security issues.
Report them as normal issues with the expected value and its source.
