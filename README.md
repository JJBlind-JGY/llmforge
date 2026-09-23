# LLMForge Infra

**Evidence-Driven LLM Inference Infrastructure & Performance Engineering**

LLMForge Infra is a hands-on systems project for studying modern LLM inference
from GPU kernels to Transformer runtime, serving, scheduling, KV cache,
distributed execution, observability, optimization, and cross-engine
validation.

It is designed around one engineering loop:

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

LLMForge is not intended to replace vLLM or SGLang. Its purpose is to provide a
measurement- and evidence-oriented environment for understanding how LLM
inference systems behave and how performance changes should be validated.

## Why LLMForge?

LLM inference performance is a cross-layer systems problem.

```text
Workload / Client
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
Attention / GEMM
       ↓
CUDA / Triton
       ↓
GPU
       ↓
NCCL / Multi-GPU
```

A faster kernel does not automatically produce a faster serving system.
Higher throughput can worsen tail latency. More GPUs can introduce enough
communication overhead to reduce scaling efficiency. Prefix reuse can change a
benchmark without changing model arithmetic.

LLMForge makes these boundaries explicit and connects performance claims to
code, workload definitions, environment records, raw artifacts, analysis, and
limitations.

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
│ TTFT / ITL / TPOT / E2E / throughput / percentiles                   │
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
```

Production observability spans the serving/runtime layers. The optimization
framework consumes evidence from earlier layers, and cross-engine validation
tests whether selected behavior is specific to one runtime implementation.

For the full design, see:

- [DESIGN.md](DESIGN.md)
- [Architecture](docs/architecture.md)

## Validated Results

Only measured results belong in this section.

The following results come from completed M1/M2 experiments on the documented
RTX 4090-class environment. They are hardware-, shape-, implementation-, and
workload-specific.

| Experiment | Measured result | Interpretation |
| --- | ---: | --- |
| CUDA Reduction, N=16,777,216 | `23.1096 ms → 0.0864 ms` | atomic baseline → warp-shuffle reduction, `267.47×` relative speedup |
| Native CUDA GEMM, N=4096 | `5.62 → 7.31 TFLOPS` | naive → shared-memory tiled16; vendor-library headroom remains large |
| Triton BF16 GEMM, N=4096 | `174.54 effective TFLOPS` | shape-specific Tensor-Core-oriented result; not a universal library comparison |
| Long-context Prefill, S=4096 | `13.4318 ms → 2.5595 ms` | naive attention → SDPA, `5.25×` wall-time reduction in the measured setup |
| Toy autoregressive generation, P=128/T=32 | `4592 → 159` token evaluations | illustrates the computation avoided by KV caching |

Detailed evidence and limitations:

- [GPU Kernel Performance Report](reports/gpu_kernel_performance.md)
- [Transformer Runtime Report](reports/transformer_runtime.md)

A result is not promoted to this table until the corresponding experiment has
real measurements and a written methodology/limitation record.

## Project Status

LLMForge separates **implementation status** from **experimental validation**.

| Area | Engineering status | Experimental status |
| --- | --- | --- |
| Reproducible benchmark foundation | ✅ ready | ✅ validated |
| CUDA / Triton GPU performance | ✅ ready | ✅ validated |
| Transformer runtime | ✅ ready | ✅ validated |
| Serving workload/metric harness | ✅ ready | 🧪 baseline/bring-up validated; full matrix pending |
| vLLM runtime tracing | ✅ ready | ⏳ live trace campaign pending |
| Distributed inference | ✅ ready | ⏳ full M5 campaign pending |
| Production observability | ✅ ready | ⏳ live integration validation pending |
| Optimization framework | ✅ ready | ⏳ final evidence-driven candidate selection pending |
| SGLang cross-engine path | ✅ ready | ⏳ live M8 validation pending |

Legend:

```text
✅ validated / ready
🧪 partially validated
⏳ live experiment pending
```

This status table is intentionally conservative. Code availability is not
treated as proof of a performance claim.

## Quick Start

### CPU-side repository development

Requirements:

```text
Python 3.12
uv
Git
```

Clone and install:

```bash
git clone https://github.com/JJBlind-JGY/llmforge.git
cd llmforge

uv sync --group dev
```

Validate:

```bash
uv run ruff check .
uv run pytest -q
```

Inspect your environment:

```bash
uv run llmforge-env --role local-dev
```

A GPU is **not** required for the first project experience.

### Explore the analysis pipeline without a GPU

Use the synthetic public examples:

[Example Artifacts](examples/README.md)

Build an M7 observation sheet:

```bash
uv run python scripts/optimization/build_observation_sheet.py \
  --manifest examples/manifests/m7_observation_manifest.json \
  --output artifacts/examples/m7_observation.json
```

Run the M8 cross-engine normalization example:

```bash
uv run python scripts/engines/analyze_cross_engine.py \
  --manifest examples/manifests/m8_cross_engine_result_manifest.json \
  --output artifacts/examples/m8_cross_engine.json
```

The example values are synthetic and are **not** performance claims.

## Explore LLMForge

### I want to learn AI infrastructure

Follow the structured:

[Learning Path](docs/learning_path.md)

You do not need four GPUs:

[Hardware Tiers](docs/hardware_tiers.md)

### I want to run experiments

Start with:

[Getting Started](docs/getting_started.md)

Then check the validated/runtime-specific versions:

[Compatibility](docs/compatibility.md)

### I want to reproduce results

Read:

- [Benchmark Methodology](docs/benchmark_methodology.md)
- [Reproduction Guide](docs/reproduction.md)

### I want to understand the runtime

Read:

- [vLLM Request Lifecycle](docs/vllm_request_lifecycle.md)
- [Multi-GPU Inference](docs/multi_gpu_inference.md)
- [Production Observability](docs/production_observability.md)

### I want to understand the optimization workflow

Read:

- [Optimization Methodology](docs/optimization_methodology.md)
- [M7 Upstream Gap Audit](docs/m7_upstream_gap_audit.md)

### I want to compare serving engines

Read:

- [Cross-Engine Validation](docs/cross_engine_validation.md)
- [System Radar](docs/system_radar.md)

The full documentation entry point is:

[LLMForge Documentation](docs/index.md)

## Hardware Model

The project was developed using:

```text
Windows 11
    └── source editing / tests / Git

Linux GPU server
    └── CUDA / profiling / vLLM / NCCL / performance experiments
```

The primary validated GPU line used RTX 4090-class 24 GB devices.

This is **not** a requirement for reading or using the repository.

Different paths are documented in:

[Hardware Tiers](docs/hardware_tiers.md)

Exact runtime/model/toolchain versions are tracked in:

[Compatibility](docs/compatibility.md)

## Repository Layout

```text
llmforge/
├── src/llmforge/        # Python infrastructure and analysis code
├── kernels/             # native CUDA / GPU kernels
├── scripts/             # benchmark, runtime, distributed, production tools
├── configs/             # reproducible public configs and examples
├── tests/               # CPU-capable and optional-runtime tests
├── docs/                # architecture, methodology, learning, how-to
├── reports/             # completed measured technical reports
├── templates/           # report templates for incomplete experiments
├── examples/            # public synthetic/sanitized example artifacts
├── artifacts/           # local raw experiment data; gitignored
└── docker/              # production-oriented tooling container
```

The repository intentionally separates:

```text
Docs      = knowledge
Reports   = measured evidence
Templates = future experiment output
Artifacts = raw local data
Examples  = safe public teaching data
```

## Benchmark Philosophy

Every performance result should answer:

```text
What workload?
What environment?
What baseline?
What timing boundary?
How many repetitions?
What correctness check?
What profiler/runtime evidence?
What trade-off?
```

Core rules:

- baseline first;
- raw data before derived claims;
- distinguish cold start from steady state;
- use correct CUDA synchronization/timing;
- keep repeated runs;
- report percentiles for serving latency;
- do not hide regressions;
- preserve useful negative results;
- do not infer causal mechanisms from a single correlated metric.

See:

[Benchmark Methodology](docs/benchmark_methodology.md)

## Non-Goals

LLMForge Infra is not:

- a replacement for vLLM or SGLang;
- a production-proven cloud serving platform;
- a Kubernetes distribution;
- a universal inference-engine leaderboard;
- a model-training framework;
- a repository that treats every implemented idea as a validated optimization.

Scope control is part of the project design.

## Development Milestones

The repository was built through the following internal milestones:

| Milestone | Area |
| --- | --- |
| M0 | Reproducible Infrastructure Foundation |
| M1 | CUDA / Triton / GPU Performance |
| M2 | Transformer Runtime |
| M3 | High-Performance Serving |
| M4 | vLLM Runtime Internals |
| M5 | Distributed Inference |
| M6 | Production Observability |
| M7 | Evidence-Driven Optimization |
| M8 | Cross-Engine Validation & Open-Source Release |

The milestone sequence records development history. New users should normally
start from the [Learning Path](docs/learning_path.md), which is organized by
systems concepts.

## Contributing

Contributions are welcome when they improve:

```text
correctness
measurement quality
runtime understanding
performance evidence
reproducibility
documentation
```

See:

[CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Project Principle

The project does not optimize for the largest possible feature list.

It optimizes for a coherent chain:

```text
Code
  +
Data
  +
Documents
  +
Understanding
```

and for performance claims that can be traced back to evidence.
