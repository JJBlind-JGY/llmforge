# Compatibility

LLM infrastructure dependencies change quickly.

This document separates:

```text
project requirement
validated environment
experimental integration
```

An adapter existing in the source tree does not imply that every version of the
corresponding runtime has been validated.

## Repository Development Requirements

| Component | Requirement |
| --- | --- |
| Python | `>=3.12,<3.13` |
| Package/dependency tool | `uv` |
| Version control | Git |
| CPU-side tests | no GPU required |

The Python distribution is `llmforge-infra`; the import package remains:

```python
import llmforge
```

## Primary Validated Development / GPU Line

The following environment reflects the project's primary validated development
and GPU execution line.

| Component | Validated value | Scope |
| --- | --- | --- |
| Local development OS | Windows 11 | source editing, tests, Git |
| GPU execution OS | Ubuntu 20.04.6 / Linux kernel 5.15 | CUDA, serving, profiling |
| Python | 3.12.14 | project/runtime environments |
| GPU | NVIDIA GeForce RTX 4090 24 GB | primary GPU experiments |
| GPU count | 4 | distributed test machine |
| Compute capability | 8.9 / SM89 | native CUDA build target |
| NVIDIA driver | 580.126.09 | primary GPU machine |
| Native CUDA Toolkit | 12.1 | M1 native CUDA build |
| Native host compiler | g++-11 | CUDA 12.1 build |
| PyTorch | 2.13.0+cu130 | project GPU/runtime line |
| vLLM | 0.29.0 | primary serving line |
| Model | `Qwen/Qwen3-8B` | serving/runtime validation |
| Model revision | `47719a242beab8f9aecc40ce3928b034dd5dd559` | pinned serving line |
| Model dtype | BF16 | serving/runtime line |

These values describe a validated environment. They are **not** all mandatory
for reading or using the repository.

## CUDA Version Semantics

Do not confuse:

```text
nvidia-smi "CUDA Version"
```

with:

```text
nvcc toolkit version
```

On the primary machine, the NVIDIA driver advertised support for a newer CUDA
runtime level while the native CUDA kernels were intentionally built with the
CUDA 12.1 toolkit and g++-11.

A benchmark report should state the version that actually matters for that
execution path.

## vLLM

Primary pinned serving line:

```text
vLLM 0.29.0
Qwen/Qwen3-8B
BF16
single-GPU baseline
```

M4 runtime integration is version-sensitive because scheduler/KV/source
internals can change between vLLM releases.

When changing vLLM:

1. re-run runtime source discovery;
2. re-run correctness tests;
3. re-run serving regressions;
4. do not reuse old performance numbers as though the runtime were unchanged.

## SGLang

The repository contains the M8 SGLang adapter and cross-engine analysis path.

Status:

```text
adapter / config path: CODE READY
live cross-engine validation: pending
version pin for final M8 experiment: pending
```

The final public compatibility matrix should pin the exact SGLang version used
for the cross-engine experiment after that environment is validated.

Do not present an unvalidated SGLang version as supported merely because the CLI
arguments look compatible.

## PyTorch and GPU Extras

The repository's optional GPU dependency reflects the validated project line.

Users should not assume that arbitrary combinations of:

```text
PyTorch
CUDA
vLLM
SGLang
Triton
```

are interchangeable.

vLLM and SGLang should remain in dedicated runtime environments unless a shared
environment has been explicitly validated.

## Model Compatibility

The serving harness is conceptually model-agnostic, but published numbers are
not.

If you change the model, record at minimum:

```text
model repository
revision
dtype
context length
attention/KV configuration
serving runtime
```

Results from another model are a new experiment.

## Status Vocabulary

Documentation uses:

| Status | Meaning |
| --- | --- |
| `VALIDATED` | exercised in the documented environment |
| `CODE READY` | implementation/tests/configuration exist |
| `EXPERIMENTAL` | interface exists but the intended live experiment is incomplete |
| `PENDING` | validation or data collection has not yet been completed |

## Upgrade Policy

When upgrading a performance-critical dependency:

```text
pin version
  ↓
run correctness
  ↓
run targeted regression
  ↓
inspect profiler/runtime behavior when relevant
  ↓
update compatibility record
```

Framework-version changes are part of the experiment environment, not an
invisible maintenance detail.
