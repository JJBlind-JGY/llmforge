# Getting Started

This guide provides the shortest path from a clean clone to a working LLMForge
development environment.

The first steps are intentionally **CPU-capable**. A GPU is not required to
inspect the project, run most infrastructure tests, analyze example artifacts,
or read the learning path.

## 1. Requirements

Minimum repository-development requirements:

```text
Git
Python 3.12
uv
```

GPU experiments have additional requirements described in
[Hardware Tiers](hardware_tiers.md) and [Compatibility](compatibility.md).

## 2. Clone the Repository

```bash
git clone https://github.com/JJBlind-JGY/llmforge.git
cd llmforge
```

## 3. Install the Development Environment

Install the project and development tools:

```bash
uv sync --group dev
```

This path is intended for repository development and CPU-side infrastructure.

It does **not** attempt to install vLLM and SGLang into the same environment.
Those runtimes are intentionally kept in dedicated environments because their
PyTorch/CUDA/kernel dependencies evolve quickly.

## 4. Validate the Repository

Run linting:

```bash
uv run ruff check .
```

Run the test suite:

```bash
uv run pytest -q
```

Tests requiring an optional dependency should skip cleanly when that dependency
is not installed.

## 5. Inspect Your Environment

LLMForge includes an environment fingerprint command:

```bash
uv run llmforge-env --role local-dev
```

On a GPU server:

```bash
uv run llmforge-env \
  --role gpu-server \
  --output artifacts/environment.json
```

The generated artifact may contain machine-specific information and is intended
for local experiment records. Do not commit private hostnames, usernames,
absolute home paths, credentials, or GPU UUIDs to the public repository.

## 6. CPU-Only Exploration

Users without a GPU can still explore:

```text
configuration validation
runtime schemas
serving metric analysis
distributed-analysis logic
production observability primitives
optimization gates
cross-engine result normalization
release tooling
```

Start with the synthetic examples:

[Example Artifacts](../examples/README.md)

For example, build an M7 observation sheet:

```bash
uv run python scripts/optimization/build_observation_sheet.py \
  --manifest examples/manifests/m7_observation_manifest.json \
  --output artifacts/examples/m7_observation.json
```

Then inspect:

```text
artifacts/examples/m7_observation.json
```

You can also run the cross-engine analysis pipeline entirely from synthetic
example data:

```bash
uv run python scripts/engines/analyze_cross_engine.py \
  --manifest examples/manifests/m8_cross_engine_result_manifest.json \
  --output artifacts/examples/m8_cross_engine.json
```

These examples demonstrate the **artifact → analysis** workflow. Their numbers
are synthetic and are not project performance claims.

## 7. Single-GPU Exploration

Before running GPU work, read:

- [Hardware Tiers](hardware_tiers.md)
- [Compatibility](compatibility.md)
- [Benchmark Methodology](benchmark_methodology.md)

Typical progression:

```text
CUDA/Triton kernels
    ↓
Transformer runtime
    ↓
single-GPU serving
    ↓
runtime tracing
```

Native CUDA work is Linux-oriented and should use the compiler/toolchain
documented in the relevant report.

The serving/runtime path should use the pinned runtime environment associated
with the target experiment rather than assuming the CPU development
environment is sufficient.

## 8. Multi-GPU Exploration

Never infer topology from GPU numbering.

Capture the target machine first:

```bash
uv run python scripts/distributed/capture_topology.py
```

Then inspect/validate distributed profiles:

```bash
uv run python scripts/distributed/validate_inference_profiles.py
```

See:

[Multi-GPU Inference](multi_gpu_inference.md)

## 9. Serving Experiments

Serving experiments use a separate runtime environment.

Before running them:

1. read [Serving Benchmark Contract](serving_benchmark_contract.md);
2. read [Serving Execution](serving_execution.md);
3. verify the model/runtime versions against [Compatibility](compatibility.md);
4. record the Git commit and environment;
5. keep raw result JSON files locally.

The primary measurement boundary is the LLMForge client, not a mixture of
unrelated engine-native benchmark outputs.

## 10. Where to Go Next

If your goal is learning:

[Learning Path](learning_path.md)

If your goal is reproducing results:

[Reproduction Guide](reproduction.md)

If your goal is understanding the design:

[Architecture](architecture.md)

If your goal is contributing:

[CONTRIBUTING.md](../CONTRIBUTING.md)

## 11. Important Project Rule

A successful command is not automatically a successful experiment.

LLMForge treats a result as publishable only after the relevant methodology,
correctness checks, repeated measurements, environment record, and limitations
are documented.
