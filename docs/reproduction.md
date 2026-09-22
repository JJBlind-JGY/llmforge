# Reproduction Guide

## 1. Clone and validate CPU-side infrastructure

```bash
uv run pytest
```

## 2. Record the target GPU/server environment

Use the existing environment/topology scripts before running benchmarks.

For multi-GPU work:

```bash
python scripts/distributed/capture_topology.py
```

## 3. Reproduce the primary vLLM line

Use the pinned model/runtime/config referenced by the target report.

Do not assume README defaults are sufficient for historical numbers.

## 4. Reproduce M3 serving workloads

Use the existing LLMForge workload generator/client so TTFT/TPOT semantics stay
consistent.

## 5. Reproduce M4 runtime evidence

Use the M4 tracing scheduler for runtime-level queue/KV/scheduling evidence.

## 6. Reproduce M5 distributed evidence

Run same-NUMA and cross-NUMA communication/serving conditions separately.

## 7. Reproduce M7 optimization

Follow:

```text
docs/optimization_methodology.md
```

The selection gate and correctness gate must pass before a final optimization
claim is accepted.

## 8. Reproduce M8 cross-engine experiment

Print launch commands:

```bash
python scripts/engines/print_engine_launch.py   --config configs/engines/m8_cross_engine.json   --engine vllm   --hf-home "$HOME/workspace/.cache/huggingface"
```

and:

```bash
python scripts/engines/print_engine_launch.py   --config configs/engines/m8_cross_engine.json   --engine sglang   --hf-home "$HOME/workspace/.cache/huggingface"
```

Probe each engine before benchmarking:

```bash
python scripts/engines/probe_engine.py   --engine vllm   --base-url http://127.0.0.1:8000
```

```bash
python scripts/engines/probe_engine.py   --engine sglang   --base-url http://127.0.0.1:30000
```

Then run the same M3 client/workload against each endpoint.

Analyze:

```bash
python scripts/engines/analyze_cross_engine.py   --manifest configs/engines/m8_result_manifest.json   --output artifacts/engines/cross_engine.json
```

## 9. Release audit

Before tagging a public release:

```bash
python scripts/release/audit_release.py   --repo-root .   --strict
```

A strict audit intentionally fails when final optimization evidence or a real
upstream contribution attempt is missing.
