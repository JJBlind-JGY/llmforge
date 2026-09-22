# LLMForge Architecture

## End-to-end request path

```text
Client / Workload Generator
        ↓
OpenAI-Compatible API
        ↓
Request Queue
        ↓
Scheduler
        ↓
Prefill / Decode
        ↓
KV Cache Manager
        ↓
Model Executor
        ↓
GEMM / Attention / Norm / Sampling
        ↓
CUDA / Triton Kernel
        ↓
GPU
        ↓
NCCL when multi-GPU
```

## Evidence path

```text
M3 client result
        │
        ├── TTFT
        ├── TPOT
        ├── ITL
        └── E2E
        ↓
M4 runtime trace
        │
        ├── queue wait
        ├── scheduled prompt/output tokens
        ├── KV pressure
        └── preemption
        ↓
M5 distributed evidence
        │
        ├── topology
        ├── collective latency
        └── TP/replica scaling
        ↓
M6 production observability
        │
        ├── logs
        ├── metrics
        └── spans
        ↓
M7 optimization evidence gate
        ↓
M8 cross-engine validation
```

## Repository philosophy

Every important mechanism should eventually have:

```text
Code
Data
Document
Knowledge
```

Code without real data is `CODE READY`, not `EXPERIMENT COMPLETE`.
