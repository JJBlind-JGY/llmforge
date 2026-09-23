# LLMForge Learning Path

This learning path is organized by **systems concepts**, not by the historical
order in which repository files were created.

The goal is not simply to execute scripts. At each stage, the learner should be
able to answer one concrete systems question using code and evidence.

## How to Use This Path

Each stage contains:

```text
Question
Prerequisites
Read
Run
Observe
Artifact
Exit Criteria
```

You do not need to complete every stage on the same machine.

See [Hardware Tiers](hardware_tiers.md) before starting GPU or multi-GPU work.

---

# Stage 0 — Systems Orientation and Benchmark Discipline

## Question

How do we know whether a performance conclusion is trustworthy?

## Prerequisites

Basic Python, Git, and command-line usage.

## Read

- [Architecture](architecture.md)
- [Benchmark Methodology](benchmark_methodology.md)
- [Reproduction Guide](reproduction.md)

## Run

Validate the repository:

```bash
uv run ruff check .
uv run pytest -q
```

Inspect your machine:

```bash
uv run llmforge-env --role local-dev
```

## Observe

Identify the difference between:

```text
source code
configuration
raw artifact
derived metric
technical report
performance claim
```

## Artifact

For GPU experiments later, save an environment fingerprint under the local,
gitignored `artifacts/` tree.

## Exit Criteria

You can explain why the following are not interchangeable:

```text
one fast run
vs repeated benchmark

wall-clock time
vs asynchronous CUDA launch time

profiler allocation
vs physical DRAM traffic

configuration choice
vs profiler evidence

code ready
vs experiment complete
```

---

# Stage 1 — GPU Execution

## Question

Why can two mathematically equivalent GPU kernels have very different runtime?

## Prerequisites

Basic CUDA execution concepts:

```text
thread
block
grid
warp
SM
global memory
shared memory
```

## Read

- [GPU Kernel Performance Report](../reports/gpu_kernel_performance.md)
- [Benchmark Methodology](benchmark_methodology.md)

## Run

Representative scripts:

```bash
python scripts/benchmark_native_vector_add.py
python scripts/benchmark_native_reduction.py
python scripts/benchmark_native_matmul.py
python scripts/benchmark_triton_vector_add.py
python scripts/benchmark_triton_matmul.py
```

For profiling, use the repository's NCU scripts on the Linux GPU server.

## Observe

Focus on:

```text
latency
useful bandwidth
effective TFLOPS
atomic contention
shared-memory conflicts
occupancy
cache behavior
DRAM throughput
scheduler stalls
```

Do not reduce the analysis to “higher occupancy is always faster” or “more
shared memory is always better.”

## Artifact

Keep benchmark CSV/JSON and profiler output locally.

The public measured summary belongs in:

[GPU Kernel Performance Report](../reports/gpu_kernel_performance.md)

## Exit Criteria

Given a kernel and profiler output, you can argue whether it is primarily
limited by:

```text
memory traffic
compute throughput
atomic contention
synchronization
on-chip memory behavior
instruction/dependency latency
parallelism/occupancy
```

You can also explain why a hand-written tiled GEMM may still remain far behind a
vendor library.

---

# Stage 2 — Transformer Runtime

## Question

Why is autoregressive LLM inference not one ordinary forward pass?

## Prerequisites

Basic Transformer attention and matrix multiplication.

## Read

- [Transformer Runtime Report](../reports/transformer_runtime.md)
- [Architecture](architecture.md)

## Run

Representative scripts:

```bash
python scripts/demo_naive_generation.py
python scripts/demo_kv_cache_generation.py
python scripts/benchmark_prefill_decode.py
python scripts/profile_prefill_decode.py
```

## Observe

Track:

```text
model token evaluations
KV-cache growth
Prefill latency
Decode latency
attention intermediates
GPU self time
end-to-end wall time
```

Pay special attention to measurement boundaries. A faster attention kernel does
not automatically imply a proportional end-to-end Decode improvement.

## Artifact

A completed runtime experiment should preserve:

```text
model configuration
sequence lengths
dtype
attention implementation
timing method
profiler evidence
limitations
```

## Exit Criteria

You can explain:

```text
Prefill
Decode
KV Cache
GQA / KV heads
why caching changes repeated computation
why long-context Prefill differs from one-token Decode
why SDPA/Flash avoids materializing large score intermediates
```

You can distinguish reduced materialization/memory traffic from the arithmetic
complexity of exact attention.

---

# Stage 3 — High-Performance LLM Serving

## Question

Why is `model.generate()` not a serving system?

## Prerequisites

Stage 2 concepts and basic HTTP/client-server knowledge.

## Read

- [Serving Benchmark Contract](serving_benchmark_contract.md)
- [Serving Execution](serving_execution.md)
- [Benchmark Methodology](benchmark_methodology.md)

## Run

Representative workload paths:

```text
fixed synthetic
mixed prompt/output lengths
concurrency sweep
shared-prefix workload
burst arrivals
Poisson arrivals
```

Representative scripts:

```bash
python scripts/run_serving_benchmark.py --help
python scripts/run_m3_serving_suite.py --help
```

Do not reserve a GPU until you understand the workload and output schema.

## Observe

Primary serving metrics:

```text
TTFT
ITL
TPOT
E2E
request throughput
output-token throughput
P50 / P95 / P99
```

Also observe how variable request lengths create scheduling and batching
inefficiency.

## Artifact

Each serving run should preserve:

```text
workload
model/revision
engine version
Git commit
server configuration
raw per-request/result data
summary metrics
```

## Exit Criteria

You can explain:

- why higher concurrency can increase throughput;
- why TTFT may worsen as concurrency increases;
- why throughput and latency are not the same objective;
- why static batching wastes work on uneven requests;
- why continuous batching and Paged KV matter.

---

# Stage 4 — Scheduler and KV Runtime

## Question

What actually happens between an HTTP request and a GPU model-execution step?

## Prerequisites

Stage 3 serving concepts.

## Read

- [vLLM Request Lifecycle](vllm_request_lifecycle.md)
- [M7 Upstream Gap Audit](m7_upstream_gap_audit.md)

## Run

Inspect the installed/pinned runtime:

```bash
python scripts/runtime/inspect_vllm_runtime.py
```

Use the M4 runtime trace path for actual experiments:

```bash
python scripts/runtime/analyze_runtime_trace.py --help
```

## Observe

Important runtime evidence:

```text
waiting requests
running requests
scheduler step duration
scheduled prompt tokens
scheduled output tokens
KV usage
preemption
queue wait
```

## Artifact

A runtime investigation should preserve the trace and a machine-readable
summary before rendering a report.

## Exit Criteria

You can trace:

```text
API request
  ↓
engine request
  ↓
scheduler
  ↓
KV allocation
  ↓
model runner
  ↓
attention/backend
  ↓
sampling/output
```

When TTFT or TPOT degrades, you know which runtime state to inspect instead of
guessing from the client metric alone.

---

# Stage 5 — Distributed Inference

## Question

When does adding GPUs improve inference, and when does communication dominate?

## Prerequisites

GPU basics plus serving metrics.

## Read

- [Multi-GPU Inference](multi_gpu_inference.md)
- [Hardware Tiers](hardware_tiers.md)

## Run

First inspect topology:

```bash
python scripts/distributed/capture_topology.py
```

Then inspect/validate profiles:

```bash
python scripts/distributed/validate_inference_profiles.py
```

Representative experiments:

```text
NCCL collective sweep
single GPU
TP=2
multi-replica serving
same-NUMA GPU placement
cross-NUMA placement
```

## Observe

Separate three layers:

```text
collective microbenchmark
model-level communication
end-to-end serving behavior
```

Useful metrics include:

```text
collective latency
algorithm bandwidth
bus bandwidth
output throughput
TTFT
TPOT
scaling efficiency
GPU utilization
```

## Artifact

Keep topology, NCCL results, serving outputs, and profiler evidence tied to the
same environment.

## Exit Criteria

You can explain why:

```text
2 GPUs != automatically 2× faster
```

and why topology can change communication behavior.

---

# Stage 6 — Production Observability

## Question

How do we explain a serving regression after it happens?

## Prerequisites

Serving and runtime concepts.

## Read

- [Production Observability](production_observability.md)

## Run

Inspect the effective config:

```bash
python scripts/production/print_production_config.py
```

The M6 control plane exposes concepts such as:

```text
liveness
readiness
normalized metrics
structured logs
request correlation
tracing
GPU telemetry
reproducible run manifests
```

## Observe

Connect:

```text
client symptom
   ↓
request ID / trace ID
   ↓
server metrics
   ↓
runtime trace
   ↓
GPU telemetry
```

## Artifact

A production-style run should preserve the effective config, process log,
manifest, and completion status without leaking credentials.

## Exit Criteria

You can explain the difference between:

```text
liveness
readiness

client TTFT
runtime queue wait

high-cardinality request identity
low-cardinality metrics
```

and can locate the evidence needed to investigate a latency regression.

---

# Stage 7 — Evidence-Driven Optimization

## Question

When is a performance change a real systems optimization rather than parameter
tuning?

## Prerequisites

At least one reproducible serving/runtime bottleneck from earlier stages.

## Read

- [Optimization Methodology](optimization_methodology.md)
- [M7 Upstream Gap Audit](m7_upstream_gap_audit.md)

## Workflow

```text
Observation
  ↓
Hypothesis
  ↓
Profiler / Runtime Evidence
  ↓
Candidate
  ↓
Correctness Gate
  ↓
Repeated Benchmark
  ↓
Regression Guardrails
  ↓
Trade-off Report
```

## Run

Representative tools:

```bash
python scripts/optimization/audit_candidate_catalog.py
python scripts/optimization/evaluate_selection_gate.py --help
python scripts/optimization/freeze_experiment.py --help
python scripts/optimization/analyze_experiment.py --help
```

CPU-only users can exercise part of this pipeline with
[Example Artifacts](../examples/README.md).

## Observe

Ask:

```text
Did the candidate actually activate?
Did the runtime mechanism change?
Did the primary objective improve?
Did throughput, TTFT, correctness, or error rate regress?
Can a static upstream knob achieve the same effect?
```

## Artifact

A final optimization claim should link:

```text
frozen experiment
selection gate
correctness result
raw baseline runs
raw candidate runs
runtime/profiler evidence
analysis
trade-off report
```

## Exit Criteria

You are willing to reject your own candidate when the evidence does not support
the proposed mechanism.

---

# Stage 8 — Cross-Engine Validation

## Question

Is an observed serving/runtime behavior specific to one implementation, or does
the same qualitative behavior appear in another modern engine?

## Prerequisites

A well-defined M3/M7 phenomenon and a stable client measurement boundary.

## Read

- [Cross-Engine Validation](cross_engine_validation.md)
- [System Radar](system_radar.md)

## Engines

```text
Primary backend: vLLM
Second backend:  SGLang
```

## Run

Inspect engine launch profiles:

```bash
python scripts/engines/print_engine_launch.py --help
```

Probe readiness:

```bash
python scripts/engines/probe_engine.py --help
```

Analyze repeated results:

```bash
python scripts/engines/analyze_cross_engine.py --help
```

A CPU-only synthetic example is available under:

[Example Artifacts](../examples/README.md)

## Observe

Do not reduce the experiment to:

```text
Which engine is faster?
```

Instead ask whether a stressed workload changes the same metric in the same
qualitative direction in both engines.

## Artifact

Keep correctness evidence and at least three repeated runs per engine for a
cross-engine comparison.

## Exit Criteria

You can distinguish:

```text
engine ranking
from
mechanism generality
```

and you do not interpret same-direction behavior as causal proof.

---

# Final Learning Outcome

A learner finishing this path should be able to reason across the stack:

```text
workload
  ↓
queue / scheduler
  ↓
Prefill / Decode
  ↓
KV state
  ↓
attention / GEMM
  ↓
CUDA / Triton
  ↓
GPU
  ↓
NCCL / topology
```

More importantly, the learner should be able to connect a performance claim to
evidence and state its limitations.
