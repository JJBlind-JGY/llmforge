# LLMForge Example Artifacts

This directory allows users to explore LLMForge's **analysis pipeline without a
GPU**.

All numerical values in this directory are **synthetic teaching examples**.
They are intentionally small, machine-independent, and must not be interpreted
as published LLMForge performance results.

Real measured conclusions belong in:

```text
reports/
```

Local machine-generated data belongs in the gitignored:

```text
artifacts/
```

## Example 1 — Build an M7 Observation Sheet

Inputs:

```text
examples/artifacts/m7/serving_result.json
examples/artifacts/m7/runtime_summary.json
examples/manifests/m7_observation_manifest.json
```

Run:

```bash
uv run python scripts/optimization/build_observation_sheet.py \
  --manifest examples/manifests/m7_observation_manifest.json \
  --output artifacts/examples/m7_observation.json
```

Expected result:

```text
artifacts/examples/m7_observation.json
```

This demonstrates the project flow:

```text
M3-style client metrics
        +
M4-style runtime evidence
        ↓
M7 observation sheet
```

The example does **not** select an optimization candidate or make a performance
claim.

## Example 2 — Cross-Engine Result Normalization

Inputs:

```text
examples/artifacts/m8/vllm/
examples/artifacts/m8/sglang/
examples/manifests/m8_cross_engine_result_manifest.json
```

Each engine has three synthetic serving-result files because the M8 analysis
requires repeated runs.

Run:

```bash
uv run python scripts/engines/analyze_cross_engine.py \
  --manifest examples/manifests/m8_cross_engine_result_manifest.json \
  --output artifacts/examples/m8_cross_engine.json
```

Expected result:

```text
artifacts/examples/m8_cross_engine.json
```

This demonstrates:

```text
repeated raw-style results
       ↓
per-engine median aggregation
       ↓
cross-engine normalized comparison
```

The example is **not** a vLLM-vs-SGLang benchmark.

## Why Synthetic Examples?

The public repository should not require access to the original development
server or expose private machine metadata.

Synthetic examples provide a stable public contract for:

```text
artifact schema
analysis code
report pipeline
```

while real experiments remain tied to explicit environments and reports.

## Public / Private Boundary

Safe public information includes:

```text
GPU model
GPU count
driver/CUDA version
runtime version
model/revision
workload definition
Git commit
sanitized raw samples where intentionally published
```

Do not publish secrets or unnecessary machine identity:

```text
API keys
tokens
private IPs
usernames
hostnames
absolute private home paths
GPU UUIDs
```
