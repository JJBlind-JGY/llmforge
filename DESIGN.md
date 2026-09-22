# LLMForge Design

## Design goal

LLMForge is designed to provide one coherent AI-infrastructure engineering path
from GPU execution to serving-system behavior.

The repository prioritizes:

```text
mechanism
measurement
reproducibility
correctness
trade-off visibility
```

over feature count.

## Layer model

```text
Layer 7  Cross-engine / OSS validation          M8
Layer 6  Evidence-driven optimization           M7
Layer 5  Production observability               M6
Layer 4  Distributed inference                  M5
Layer 3  Runtime internals / scheduler           M4
Layer 2  Serving / workload / metrics            M3
Layer 1  Transformer execution                   M2
Layer 0  CUDA / Triton / GPU                     M1
Foundation  Reproducible experiment harness      M0
```

## Primary backend policy

vLLM is the primary serving/runtime backend.

SGLang is introduced only in M8 as an independent engine for cross-engine
validation and architecture study.

This avoids splitting the project into two incomplete serving stacks.

## Measurement boundaries

LLMForge keeps measurement sources explicit.

```text
client-observed
engine/runtime-observed
GPU/profiler-observed
distributed-communication-observed
```

The project does not silently equate them.

Examples:

```text
client TTFT != scheduler queue wait
profiler allocation != physical DRAM traffic
NCCL microbenchmark != model communication time
config says SDPA != profiler proof of a Flash kernel
```

## Optimization policy

M7 cannot select a headline optimization from intuition alone.

A candidate must pass an eight-condition evidence gate and must preserve
correctness.

The final report exposes regressions and negative results.

## Cross-engine policy

M8 uses the same LLMForge client against OpenAI-compatible endpoints when
comparing vLLM and SGLang.

Engine-native benchmarks remain useful secondary evidence, but they do not
replace the primary measurement boundary.

## Scope control

LLMForge studies Mooncake, NVIDIA Dynamo, and llm-d as architecture radar items.

They are not all required local deployments.

A fake Kubernetes deployment is less useful for this project than a real,
well-profiled single-node inference experiment.
