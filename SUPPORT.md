# Support

LLMForge Infra is an open-source research and engineering project rather than a
hosted support service.

The best support channel depends on the type of question.

## Bug in LLMForge

Open a bug report and include:

```text
Git commit
operating system
Python version
relevant optional/runtime dependencies
command
expected behavior
actual behavior
minimal reproduction
```

Use:

```text
.github/ISSUE_TEMPLATE/bug_report.yml
```

## Benchmark or Performance Question

Performance issues require more context than ordinary bugs.

Please include:

```text
GPU model and count
driver
CUDA/toolkit/runtime version
PyTorch / vLLM / SGLang version when relevant
model and revision
Git commit
workload
concurrency / arrival process
command
raw artifact or profiler evidence
```

Use:

```text
.github/ISSUE_TEMPLATE/benchmark_issue.yml
```

A single screenshot or one benchmark number is usually not enough to diagnose a
performance issue.

## Feature Request

Use:

```text
.github/ISSUE_TEMPLATE/feature_request.yml
```

Before proposing a large feature, review the project's scope and
[DESIGN.md](DESIGN.md).

## Environment / Version Problems

Read:

- [Getting Started](docs/getting_started.md)
- [Hardware Tiers](docs/hardware_tiers.md)
- [Compatibility](docs/compatibility.md)

vLLM, SGLang, PyTorch, CUDA, and Triton change quickly. Include exact versions
when reporting runtime problems.

## Security

Do not disclose sensitive vulnerabilities in a public issue.

See:

[SECURITY.md](SECURITY.md)

## General Learning Questions

The public learning sequence is:

[Learning Path](docs/learning_path.md)

The repository aims to make the main concepts understandable from the
documentation and executable examples, but it does not provide guaranteed
one-on-one support for every hardware/runtime combination.
