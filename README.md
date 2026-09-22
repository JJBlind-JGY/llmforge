# LLMForge

**A Production-Oriented LLM Inference Infrastructure & Performance Engineering Stack**

## What is LLMForge?

LLMForge is an educational and engineering project for studying modern LLM
inference from GPU kernels to serving runtimes, distributed execution, runtime
instrumentation, production observability, and evidence-driven optimization.

The project follows one rule:

```text
Build
  ↓
Measure
  ↓
Understand
  ↓
Optimize
  ↓
Reproduce
```

It is not a collection of disconnected CUDA demos and it does not publish
performance claims without a defined workload, environment, raw artifacts, and
correctness/regression checks.

## What problem does it help study?

LLM inference performance is a cross-layer problem:

```text
Request
  ↓
Queue
  ↓
Scheduler
  ↓
Prefill / Decode
  ↓
KV Cache
  ↓
Model Runtime
  ↓
CUDA / Triton Kernels
  ↓
GPU
  ↓
NCCL / Multi-GPU
```

A kernel can become faster without improving end-to-end latency. Higher
throughput can worsen tail latency. More GPUs can introduce communication
overhead. Prefix reuse can change a benchmark without changing model compute.

LLMForge makes those boundaries explicit and provides reproducible artifacts for
reasoning about them.

## Architecture

```text
                         ┌─────────────────────────┐
                         │      Workload Layer     │
                         │ fixed / mixed / prefix  │
                         │ burst / Poisson         │
                         └────────────┬────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Serving Measurement                            │
│ TTFT / TPOT / ITL / E2E / throughput / goodput                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Runtime / Scheduler                            │
│ queue / token budget / KV / preemption / scheduling decisions         │
└─────────────────────┬──────────────────────────────────┬────────────────┘
                      │                                  │
                      ▼                                  ▼
          ┌─────────────────────┐             ┌──────────────────────┐
          │ Model / Attention   │             │ Distributed Runtime  │
          │ Prefill / Decode    │             │ NCCL / TP / replica │
          │ KV / SDPA / Flash   │             │ topology            │
          └──────────┬──────────┘             └──────────┬───────────┘
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       ▼
                           ┌─────────────────────┐
                           │ CUDA / Triton / GPU │
                           └─────────────────────┘

M6 overlays production observability.
M7 consumes evidence from all earlier layers.
M8 validates selected behavior across vLLM and SGLang and prepares OSS release.
```

See [DESIGN.md](DESIGN.md) and
[docs/architecture.md](docs/architecture.md).

## Key Results

Only measured results belong here.

Current validated project evidence includes GPU-kernel and Transformer-runtime
experiments from M1/M2, while later serving/runtime/distributed/optimization
claims are published only after their corresponding report is backed by real
server artifacts.

Detailed reports:

```text
reports/gpu_kernel_performance.md
reports/vllm_baseline.md
reports/runtime_trace.md
reports/multi_gpu_inference.md
reports/production_readiness.md
reports/final_optimization.md
```

The final M7 optimization number must not be added to this README until its
selection gate, correctness gate, repeated benchmark, and regression guardrails
all pass.

## Milestone Map

```text
M0  Reproducible Infra Foundation
M1  CUDA / Triton / GPU Performance
M2  Transformer Runtime
M3  High-Performance Serving
M4  vLLM Runtime Internals
M5  Distributed Inference
M6  Production Observability
M7  Evidence-Driven Original Optimization
M8  Cross-Engine Validation & Open-Source Release
```

## Supported Environment

Primary development workflow:

```text
Windows 11
  └── source editing / tests / Git

Linux GPU server
  └── CUDA / vLLM / profiling / NCCL / benchmarks
```

Primary GPU line used during project development:

```text
4 × NVIDIA GeForce RTX 4090 24 GB
CUDA-capable Linux server
vLLM pinned per experiment
Qwen/Qwen3-8B serving line
```

Exact driver/CUDA/framework/model revisions are recorded per experiment rather
than assumed from this README.

## Quick Start

Clone and install the development environment according to the repository's
existing `uv` workflow.

Run CPU-side correctness tests:

```bash
uv run pytest
```

Inspect serving/runtime/production configs before reserving a GPU.

Examples:

```bash
python scripts/production/print_production_config.py

python scripts/distributed/validate_inference_profiles.py

python scripts/optimization/audit_candidate_catalog.py
```

## Reproduce

Start with:

```text
docs/reproduction.md
docs/benchmark_methodology.md
```

The expected data flow is:

```text
raw JSON / CSV / profiler trace
        ↓
analysis script
        ↓
table / figure / report
        ↓
README claim
```

Never reproduce a performance number from a screenshot alone.

## Cross-Engine Validation

vLLM remains the primary backend.

M8 adds SGLang to ask a different question:

> Is the observed serving behavior implementation-specific, or does the same
> qualitative behavior appear in another modern inference engine?

See:

```text
docs/cross_engine_validation.md
docs/system_radar.md
```

## Testing and CI

The CPU CI validates pure infrastructure logic:

```text
runtime schemas
distributed analysis
observability
optimization gates
engine adapters
release audit
```

GPU integration tests and performance benchmarks run on the real GPU server and
are not replaced by fake CI GPU measurements.

## Open Source

See:

```text
CONTRIBUTING.md
LICENSE
docs/release_checklist.md
```

A public release is not considered complete until a real upstream issue, patch,
or pull-request attempt is recorded.
