# Contributing to LLMForge

## Principles

A contribution should improve one of:

```text
correctness
measurement quality
runtime understanding
performance evidence
reproducibility
documentation
```

Avoid adding empty abstractions or benchmark claims without raw evidence.

## Development workflow

Use the repository's `uv` environment and run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

before opening a pull request.

## Performance changes

A performance-oriented contribution should include:

```text
problem statement
baseline
workload
environment
raw data
correctness test
before/after metrics
trade-offs
```

Do not submit only a chart or a single best number.

## Framework-version changes

vLLM and SGLang change quickly.

Any runtime-version upgrade should record the new version and re-run relevant
regression benchmarks.

## GPU-specific code

CPU CI is not a substitute for real GPU validation.

State clearly when a change is CPU-tested but GPU-validation-pending.

## Issues and upstream work

When a bug belongs primarily to vLLM, SGLang, PyTorch, Triton, or another
upstream project, prefer a minimal upstream reproduction/issue when appropriate
instead of permanently hiding the issue in LLMForge.
