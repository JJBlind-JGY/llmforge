# LLMForge Infra Documentation

LLMForge Infra is an evidence-driven LLM inference infrastructure and performance
engineering project.

It connects GPU execution, Transformer runtime, serving, scheduling, KV cache,
distributed inference, observability, optimization, and cross-engine validation
through one reproducible engineering workflow:

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

This documentation is organized by **what a visitor wants to do**, rather than
by the order in which the repository was originally developed.

## Start Here

### I want to learn AI infrastructure

Follow the structured [Learning Path](learning_path.md).

It walks through:

```text
GPU Execution
  ↓
Transformer Runtime
  ↓
LLM Serving
  ↓
Scheduler & KV Runtime
  ↓
Distributed Inference
  ↓
Observability
  ↓
Optimization Engineering
  ↓
Cross-Engine Validation
```

Each stage defines a systems question, what to read, what to run, what to
observe, and an exit criterion.

### I want to run LLMForge

Start with [Getting Started](getting_started.md).

A four-GPU server is **not** required to explore the repository. See
[Hardware Tiers](hardware_tiers.md) for CPU-only, single-GPU, and multi-GPU
paths.

### I want to reproduce benchmark results

Read these documents first:

1. [Benchmark Methodology](benchmark_methodology.md)
2. [Reproduction Guide](reproduction.md)
3. [Compatibility](compatibility.md)

The project follows a raw-data-first rule:

```text
raw artifact
   ↓
analysis script
   ↓
derived result
   ↓
technical report
   ↓
README claim
```

A screenshot or an isolated benchmark number is not treated as sufficient
performance evidence.

### I want to understand the system design

Start with:

- [Architecture](architecture.md)
- [DESIGN.md](../DESIGN.md)

Then continue into the subsystem documents below.

## Documentation Map

| Area | Primary document | What it explains |
| --- | --- | --- |
| Benchmark discipline | [Benchmark Methodology](benchmark_methodology.md) | timing, warmup, repeats, percentiles, evidence boundaries |
| GPU execution | [GPU Kernel Report](../reports/gpu_kernel_performance.md) | CUDA/Triton optimization and profiler evidence |
| Transformer runtime | [Transformer Runtime Report](../reports/transformer_runtime.md) | autoregressive generation, KV cache, Prefill/Decode, attention runtime |
| Serving | [Serving Benchmark Contract](serving_benchmark_contract.md) | workloads, TTFT/TPOT/ITL/E2E, serving experiment semantics |
| Serving execution | [Serving Execution](serving_execution.md) | how to execute the M3 workload suite |
| vLLM runtime | [vLLM Request Lifecycle](vllm_request_lifecycle.md) | request → scheduler → KV → model execution |
| Distributed inference | [Multi-GPU Inference](multi_gpu_inference.md) | topology, NCCL, TP, replica serving |
| Observability | [Production Observability](production_observability.md) | metrics, logs, request IDs, tracing, health/readiness |
| Optimization | [Optimization Methodology](optimization_methodology.md) | evidence gate, correctness, before/after, regression guardrails |
| Upstream-gap audit | [M7 Upstream Gap Audit](m7_upstream_gap_audit.md) | avoiding “parameter tuning as contribution” |
| Cross-engine | [Cross-Engine Validation](cross_engine_validation.md) | vLLM/SGLang comparison methodology |
| System landscape | [System Radar](system_radar.md) | vLLM, SGLang, Mooncake, Dynamo, llm-d |
| Reproduction | [Reproduction Guide](reproduction.md) | reproducing project experiments |
| Hardware | [Hardware Tiers](hardware_tiers.md) | what can be run on different machines |
| Versions | [Compatibility](compatibility.md) | validated versus experimental environments |

## Evidence Status

LLMForge distinguishes three states:

```text
CODE READY
GPU VALIDATED
EXPERIMENT COMPLETE
```

Their meanings are intentionally different.

**CODE READY** means an implementation, configuration, test, or analysis path
exists and passes the applicable CPU-side checks.

**GPU VALIDATED** means the path has been exercised in the documented GPU/runtime
environment.

**EXPERIMENT COMPLETE** means the experiment has a frozen workload and
environment, correctness checks where applicable, repeated measurements, raw
artifacts, analysis, and a written interpretation.

Code existing in the repository does **not** automatically imply that a
performance claim has been validated.

## Reports, Templates, and Artifacts

The repository uses four different concepts:

```text
docs/
    knowledge, architecture, methodology, how-to

reports/
    measured conclusions backed by completed experiments

templates/
    report structures for experiments that have not yet been completed

artifacts/
    local machine-generated raw data; not committed by default
```

Small synthetic examples for CPU-only exploration live under
[`examples/`](../examples/).

## Development History

LLMForge was developed through milestones M0–M8. Those milestone labels are
useful for project history, but the public documentation is intentionally
organized by systems concepts.

Historical engineering notes, when retained, live under:

```text
docs/development/
```

They are not the primary starting point for new users.
