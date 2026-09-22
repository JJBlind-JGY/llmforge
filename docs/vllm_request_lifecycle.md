# M4 — vLLM V1 Request Lifecycle

## Goal
Move from vLLM user to runtime reader/modifier.

```text
HTTP Request
   ↓
Engine Request
   ↓
Scheduler
   ↓
KV Allocation
   ↓
Model Runner / InputBatch
   ↓
Attention Backend
   ↓
GPU Forward
   ↓
Sampling
   ↓
Output
```

## Pinned runtime contract
Before runtime work:

```bash
python scripts/runtime/inspect_vllm_runtime.py \
  --output artifacts/runtime/vllm_source_map.json
```

Logical source targets:

```text
SchedulerConfig
Scheduler
Request
SchedulerOutput
KVCacheManager
GPUModelRunner
InputBatch
attention backend
engine/runtime stats
```

## First instrumentation boundary
`TracingScheduler` is observational. It records request enqueue, waiting/running counts, request-level scheduled tokens, prompt-vs-output token positions, scheduler duration, KV usage ratio, preemption, and request completion/release.

It does not change scheduling policy, token budget, request order, admission, preemption victim selection, KV allocation, or prefix caching.

## Timing
Every event stores `wall_time_ns` and `monotonic_ns`. Wall time aligns independent logs; monotonic time is used for durations.

## Low-overhead trace path

```text
scheduler thread
    ↓ non-blocking enqueue
bounded queue
    ↓
background JSONL writer
```

If the trace queue drops events, the trace is incomplete and should not support fine-grained causal claims.

## M3 ↔ M4 connection
M3 measures client-observed TTFT / TPOT / ITL / E2E. M4 measures engine-observed queue wait, scheduler steps, scheduled prompt/output mix, KV pressure, preemption, and scheduler overhead.

## Status

```text
CODE READY          yes
GPU VALIDATED       pending
EXPERIMENT COMPLETE pending
```
