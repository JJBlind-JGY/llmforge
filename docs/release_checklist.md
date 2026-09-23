# Open-Source Release Checklist

LLMForge uses two different completion gates:

```text
release_ready
portfolio_complete
```

They intentionally answer different questions.

## Gate 1 — `release_ready`

Question:

> Can a new user clone, understand, test, build, and legally use the public
> repository without depending on unfinished private experiments?

This gate does **not** require every M3–M8 experiment to be complete.

### Public repository surface

- [ ] README public entry is current
- [ ] DESIGN.md is current
- [ ] docs/index.md exists
- [ ] getting-started guide works from a clean clone
- [ ] learning path exists
- [ ] hardware tiers are documented
- [ ] compatibility/version policy is documented
- [ ] architecture and benchmark methodology are current
- [ ] reproduction guide is current
- [ ] M1/M2 validated reports are present
- [ ] synthetic public examples are present
- [ ] CONTRIBUTING.md exists
- [ ] SECURITY.md exists
- [ ] CODE_OF_CONDUCT.md exists
- [ ] LICENSE exists
- [ ] CHANGELOG.md exists
- [ ] CITATION.cff exists

### Tests and packaging

- [ ] `ruff format --check .`
- [ ] `ruff check .`
- [ ] `pytest -q`
- [ ] `uv build`
- [ ] built wheel installs in a clean smoke environment
- [ ] CPU GitHub Actions pass
- [ ] no CPU CI step silently requires a CUDA runtime

### Git / data hygiene

- [ ] no credentials or tokens are committed
- [ ] no private hostnames, usernames, or private IPs are exposed unnecessarily
- [ ] raw local artifacts remain gitignored
- [ ] committed examples are explicitly synthetic or sanitized
- [ ] no generated caches/venvs/build outputs are committed
- [ ] commit history is reviewable

A repository can pass this gate while later GPU-validation campaigns are still
pending.

## Gate 2 — `portfolio_complete`

Question:

> Is the complete portfolio story, including the evidence-driven optimization
> and cross-engine validation, finished and defensible?

This gate is stricter and requires `release_ready` first.

### Measured experiment reports

- [ ] vLLM serving baseline report is complete
- [ ] M4 runtime trace report is complete
- [ ] M5 multi-GPU report is complete
- [ ] M6 production-readiness report is complete
- [ ] M7 final optimization report is complete
- [ ] M8 cross-engine validation report is complete
- [ ] none of the final reports contain `PENDING` / `TODO` placeholders

### M7

- [ ] final candidate passed the evidence-selection gate
- [ ] correctness passed
- [ ] baseline/candidate repetitions are preserved
- [ ] profiler/runtime evidence supports the mechanism
- [ ] trade-offs and regressions are reported
- [ ] README includes only the validated final result

### M8

- [ ] exact SGLang version is pinned in compatibility docs
- [ ] vLLM/SGLang correctness validation is complete
- [ ] repeated cross-engine runs exist
- [ ] the mechanism-generality conclusion is appropriately limited

### Upstream contribution

- [ ] at least one real issue, patch, or PR attempt exists
- [ ] `docs/upstream_contribution.json` records the URL and summary

Do not fabricate an upstream contribution merely to make this gate pass.

## Run the Audit

Human-readable/non-strict construction audit:

```bash
uv run python scripts/release/audit_release.py --repo-root .
```

Require the public OSS release surface:

```bash
uv run python scripts/release/audit_release.py \
  --repo-root . \
  --strict
```

Require the complete portfolio:

```bash
uv run python scripts/release/audit_release.py \
  --repo-root . \
  --require-portfolio-complete
```

The alpha release is allowed to be:

```text
release_ready = true
portfolio_complete = false
```

That state is expected until the remaining GPU-validation work is finished.
