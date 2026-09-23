# Hardware Tiers

You do **not** need four GPUs to use LLMForge.

Different parts of the repository have different hardware requirements. The
project separates learning, analysis, single-GPU runtime work, and multi-GPU
experiments so users can choose a path that matches their machine.

## Overview

| Tier | Example hardware | Recommended scope |
| --- | --- | --- |
| CPU | laptop / CI runner | tests, configs, artifact analysis, metrics, release tooling |
| GPU-S | 1 GPU with modest VRAM | CUDA/Triton kernels, smaller-model runtime experiments |
| GPU-M | 1 × ~24 GB GPU | validated Qwen3-8B-class serving/runtime path |
| GPU-D | 2 GPUs | NCCL, TP=2, topology-sensitive experiments |
| GPU-X | 4 GPUs | complete distributed benchmark matrix and placement studies |

These tiers describe practical project paths, not universal model-memory
guarantees.

## Tier 0 — CPU-Only

A GPU is not required for:

```text
repository tests
configuration validation
runtime/event schema tests
serving metric analysis
distributed-analysis logic
observability primitives
optimization selection/correctness logic
cross-engine result normalization
release audit
```

Start with:

[Getting Started](getting_started.md)

Then use:

[Example Artifacts](../examples/README.md)

This path is especially useful for understanding the project's evidence flow:

```text
artifact
  ↓
analysis
  ↓
report
```

without reproducing the GPU workload itself.

## Tier 1 — Single GPU, Smaller VRAM

A smaller GPU can still be useful for:

```text
VectorAdd
Reduction
small GEMM
Triton exercises
small Transformer runtime experiments
smaller-model serving
```

If you substitute a smaller model, treat the result as a different experiment.

Do not compare its numerical results directly with the repository's Qwen3-8B
results unless the workload, model, dtype, and environment are intentionally
matched.

## Tier 2 — One ~24 GB GPU

The project's primary large-model single-GPU line was validated on an
RTX 4090-class 24 GB GPU.

This tier is suitable for the project's Qwen3-8B BF16 serving/runtime path when
sufficient VRAM is actually free.

Before starting a server, check:

```text
other GPU processes
free VRAM
model weight footprint
KV-cache allocation
activation/headroom requirements
```

A nominal “24 GB GPU” is not enough if several GiB are already occupied by
another process.

## Tier 3 — Two GPUs

Two GPUs enable:

```text
NCCL collective experiments
TP=2
replica comparison
same-domain vs cross-domain placement studies
```

Before choosing GPU IDs, inspect topology:

```bash
python scripts/distributed/capture_topology.py
```

GPU numbering alone does not tell you whether the devices share a NUMA domain,
PCIe hierarchy, or high-speed interconnect.

## Tier 4 — Four GPUs

Four GPUs enable the full M5-style experiment matrix:

```text
single GPU baseline
TP=2
larger TP configuration where viable
multiple replicas
same-NUMA placement
cross-NUMA placement
collective scaling
```

This is useful for studying distributed behavior, but it is **not** a
requirement for using the repository.

## Model Substitution

A user may substitute another model to fit available hardware.

When doing so, record:

```text
model name
revision
parameter count
dtype
attention/KV configuration
context length
```

The purpose is to preserve the methodology, not to pretend results from
different models are directly comparable.

## Topology Matters

For multi-GPU experiments, save the topology with the experiment:

```bash
python scripts/distributed/capture_topology.py
```

Distinguish:

```text
GPU count
from
GPU connectivity
```

Two experiments with the same number of GPUs can have different communication
behavior because the physical topology differs.

## Local Machine Data

Public configs use portable defaults where possible.

Machine-specific details such as:

```text
hostname
username
absolute home path
GPU UUID
private IP address
```

should remain in local artifacts and should not be committed.

Hardware properties necessary to interpret a benchmark should remain public:

```text
GPU model
GPU count
driver
CUDA/runtime version
CPU/NUMA topology when relevant
```

## Recommended Path

If you are unsure where to start:

```text
CPU-only validation
    ↓
Stage 1 GPU kernels
    ↓
Stage 2 Transformer runtime
    ↓
single-GPU serving
    ↓
multi-GPU only when needed
```

See the [Learning Path](learning_path.md).
