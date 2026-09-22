# Open-Source Release Checklist

## Repository surface

- [ ] README 60-second experience is current
- [ ] DESIGN.md is current
- [ ] architecture doc is current
- [ ] benchmark methodology is current
- [ ] reproduction guide is current
- [ ] CONTRIBUTING.md exists
- [ ] LICENSE exists

## Reports

- [ ] GPU-kernel report is backed by raw data
- [ ] vLLM serving report is backed by raw data
- [ ] M4 runtime report is backed by trace artifacts
- [ ] multi-GPU report is backed by topology/NCCL/serving data
- [ ] production-readiness report is updated
- [ ] final M7 optimization report has no placeholder claim
- [ ] M8 cross-engine report is backed by repeated results

## Tests and CI

- [ ] `ruff format --check`
- [ ] `ruff check`
- [ ] CPU CI passes
- [ ] relevant local GPU correctness tests pass
- [ ] selected benchmark reproduces from a clean checkout

## Git hygiene

- [ ] no large raw artifacts accidentally committed
- [ ] no credentials/tokens in repository history
- [ ] no generated caches/venvs
- [ ] commit history is reviewable
- [ ] release tag points at the exact documented commit

## Upstream attempt

- [ ] at least one real upstream issue, patch, or PR attempt exists
- [ ] `docs/upstream_contribution.json` records the URL and summary

Do not fabricate an upstream attempt. If no useful upstream contribution exists
yet, the public release gate remains open.
