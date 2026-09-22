# M6 — Production-Oriented Infra

## Why M6 exists

M0-M5 can already benchmark kernels, model execution, serving, runtime behavior,
and multi-GPU inference.

That is still not enough to call the project production-oriented infrastructure.

A serving system must also answer:

```text
Is the service alive?
Is it ready to receive traffic?
Where are requests waiting?
Why did tail latency get worse?
How full is KV cache?
Are GPUs actually busy?
Can failures be correlated across logs and traces?
Can another engineer reproduce this run?
```

This M6 layer implements the original syllabus scope:

```text
standardized config
structured logging
request ID
metrics
trace
Docker
health check
timeout / cancellation
reproducible run script
CI
documentation
```

## Architecture

```text
                  ┌─────────────────────────────┐
                  │        vLLM Server          │
                  │ /health      /metrics       │
                  └──────────────┬──────────────┘
                                 │
                          scrape / probe
                                 │
                                 ▼
┌──────────────┐        ┌───────────────────────┐
│ nvidia-smi   │───────▶│ LLMForge M6 Sidecar  │
│ GPU metrics  │        │                       │
└──────────────┘        │ config                │
                        │ health/readiness       │
M3 RequestResult ──────▶│ normalized metrics    │
M4 Runtime Trace ──────▶│ structured logs       │
M5 GPU samples ────────▶│ spans / trace export  │
                        └───────────┬───────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
              /metrics        /health/ready    JSONL / OTLP
```

The sidecar does not proxy inference requests and therefore does not add itself
to the request critical path.

## 1. Standardized configuration

The effective production config uses:

```text
defaults
   ↓
configs/production/m6_observability.json
   ↓
LLMFORGE_* environment variables
```

The final configuration is validated before the service starts.

Useful overrides:

```text
LLMFORGE_CONTROL_HOST
LLMFORGE_CONTROL_PORT
LLMFORGE_UPSTREAM_BASE_URL
LLMFORGE_REQUEST_TIMEOUT_S
LLMFORGE_GPU_INDICES
LLMFORGE_SCRAPE_VLLM_METRICS
LLMFORGE_SCRAPE_GPU_METRICS
LLMFORGE_LOG_LEVEL
LLMFORGE_TRACING_MODE
LLMFORGE_TRACE_JSONL_PATH
OTEL_EXPORTER_OTLP_ENDPOINT
```

Inspect the effective config:

```bash
python scripts/production/print_production_config.py
```

## 2. Structured logging and request correlation

`contextvars` carries:

```text
request_id
trace_id
parent_span_id
```

JSON logs include the current request/trace identifiers automatically.

This avoids adding `request_id=...` manually to every log call.

Do not use request IDs as Prometheus labels. They are intentionally high
cardinality and belong in logs/traces.

## 3. Normalized metrics

LLMForge exports the syllabus-required signals using explicit units:

```text
llmforge_request_total
llmforge_request_running
llmforge_request_waiting
llmforge_queue_wait_seconds
llmforge_ttft_seconds
llmforge_tpot_seconds
llmforge_e2e_seconds
llmforge_input_tokens
llmforge_output_tokens
llmforge_kv_usage_ratio
llmforge_prefix_cache_hit_total
llmforge_gpu_utilization_ratio
llmforge_gpu_memory_bytes
```

Additional useful signals include:

```text
llmforge_input_tokens_total
llmforge_output_tokens_total
llmforge_prefix_cache_query_total
llmforge_gpu_memory_total_bytes
llmforge_gpu_power_watts
llmforge_observer_error_total
```

### Why histograms are fixed-memory

Latency and token distributions do not retain raw samples in a running service.

A histogram stores only:

```text
bucket counts
sum
count
```

so memory does not grow with service lifetime.

Raw per-request samples still belong in M3 benchmark artifacts when conducting
an experiment.

## 4. Measurement-source labels

M6 deliberately preserves source semantics.

```text
source="client"
```

means M3 client-observed request timing.

```text
source="runtime"
```

means M4 engine/runtime observation.

```text
source="vllm"
```

means values scraped from the vLLM Prometheus endpoint.

These must not be silently treated as interchangeable.

In particular:

```text
client TTFT != automatically the same measurement boundary as server TTFT
client queue wait != runtime request queue wait
```

## 5. vLLM metrics

The sidecar reads only a small normalized subset from the upstream `/metrics`
endpoint:

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:prompt_tokens_total
vllm:generation_tokens_total
vllm:request_success_total
```

The original upstream endpoint remains useful and should not be hidden.

LLMForge's normalized metrics exist to connect M3/M4/M5 evidence under one
project vocabulary, not to replace all vLLM-native observability.

## 6. Health and readiness

The M6 control plane exposes:

```text
GET /health/live
GET /health/ready
GET /metrics
GET /status
```

Liveness means the sidecar process itself is running.

Readiness checks required dependencies such as:

```text
vLLM /health
selected GPUs
```

A required check failing immediately marks `ready=false`.

`failure_threshold` controls the transition from `degraded` to `unhealthy`;
it does not hide a failed dependency from the readiness endpoint.

## 7. Timeout and cancellation

The codebase provides:

```text
CancellationToken
run_with_timeout()
GracefulShutdown
```

The managed command runner also supports:

```text
timeout
SIGINT/SIGTERM
graceful terminate
forced kill after grace period
```

This is the minimum production lifecycle needed before introducing more complex
orchestrators.

## 8. Tracing

Default M6 tracing is bounded JSONL application-span tracing.

Each span includes:

```text
trace_id
span_id
parent_span_id
wall-clock timestamps
monotonic timestamps
status
attributes
```

Nested spans inherit the request trace ID.

OpenTelemetry is optional. The bridge uses an OTLP/HTTP exporter and a batch
span processor when the optional packages are installed.

For a real production backend, send OTLP to an OpenTelemetry Collector rather
than baking a vendor backend directly into LLMForge.

## 9. GPU metrics

M6 reuses the GPU parser introduced by M5 instead of maintaining a second
`nvidia-smi` interpretation.

Host deployment:

```text
scrape_gpu_metrics=true
gpu_required=true
```

Container/CPU-only control-plane deployment can use:

```text
LLMFORGE_SCRAPE_GPU_METRICS=false
LLMFORGE_HEALTH_GPU_REQUIRED=false
```

If an NVIDIA runtime exposes `nvidia-smi` inside the container, GPU collection
may be enabled there as well.

## 10. Running the sidecar

Start vLLM first, then:

```bash
python scripts/production/run_observability_sidecar.py   --config configs/production/m6_observability.json
```

Probe readiness:

```bash
python scripts/production/wait_until_ready.py   --url http://127.0.0.1:9108/health/ready
```

Inspect:

```text
http://127.0.0.1:9108/metrics
http://127.0.0.1:9108/status
```

## 11. Reproducible managed runs

Run a service or benchmark through:

```bash
python scripts/production/run_with_manifest.py   --config configs/production/m6_observability.json   --env CUDA_VISIBLE_DEVICES=3   --   python some_script.py
```

The run directory contains:

```text
manifest.json
process.log
completion.json
```

The manifest records a safe environment whitelist instead of dumping the full
environment, which avoids accidentally storing credentials.

## 12. Docker

Build:

```bash
docker build   -f docker/Dockerfile.observability   -t llmforge-observability:dev .
```

Run on Linux against a host vLLM server:

```bash
docker run --rm   --network host   -e LLMFORGE_SCRAPE_GPU_METRICS=false   -e LLMFORGE_HEALTH_GPU_REQUIRED=false   llmforge-observability:dev
```

For GPU telemetry inside the container, use a correctly installed NVIDIA
Container Toolkit and explicitly validate `nvidia-smi` before enabling the GPU
collector.

## 13. CPU CI versus GPU validation

The GitHub CPU workflow intentionally validates:

```text
config
logging
request context
metric semantics
trace semantics
health/lifecycle
runtime analysis
distributed analysis
```

without downloading a full CUDA/PyTorch stack.

GPU integration remains a separate server validation phase.

This keeps pull requests fast while preserving the project's real GPU benchmark
workflow.

## 14. Why Kubernetes is not in M6 v1

The original syllabus explicitly prioritizes runtime/performance/distributed
inference on the available local multi-GPU server.

Docker is mandatory.

Kubernetes/Slurm concepts are useful, but a fake cluster would add repository
surface area without improving the core performance-engineering evidence.

## M6 exit criteria

Before M6 becomes `EXPERIMENT COMPLETE`, verify:

```text
[ ] sidecar starts next to pinned vLLM
[ ] /health/live works
[ ] /health/ready fails when upstream dies
[ ] /metrics exposes normalized vLLM signals
[ ] GPU telemetry updates
[ ] M3 RequestResult adapter records client latency
[ ] M4 runtime adapter records queue/KV evidence
[ ] structured logs contain request_id and trace_id
[ ] JSONL spans preserve parent/child structure
[ ] optional OTLP export reaches a Collector
[ ] managed run produces manifest/log/completion
[ ] Docker image passes liveness check
[ ] CPU GitHub Actions job passes
```
