## Summary

What does this pull request change?

## Why

What problem does it solve, and why does it belong in LLMForge?

## Validation

List the commands/tests you ran.

```text
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Add GPU/runtime commands when applicable.

## Performance Changes

If this PR changes performance-sensitive code, include:

```text
baseline
workload
environment
raw artifact
before/after metrics
correctness check
trade-offs / regressions
```

If performance is not affected, write:

```text
Not performance-sensitive.
```

## Checklist

- [ ] The change is scoped to one coherent capability.
- [ ] Tests cover the changed behavior.
- [ ] Documentation/configuration is updated when necessary.
- [ ] No secrets or private machine identifiers are included.
- [ ] Performance claims, if any, are backed by reproducible evidence.
- [ ] I did not present an upstream configuration knob as an original optimization.
