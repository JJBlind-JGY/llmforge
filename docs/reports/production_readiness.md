# M6 Production Readiness Report

> Status: CODE READY / LIVE INTEGRATION PENDING.

## Configuration

Pending live validation:

- effective config:
- git commit:
- server command:
- GPU selection:
- vLLM version/model revision:

## Health

Pending:

- `/health/live`:
- `/health/ready` with healthy upstream:
- `/health/ready` after upstream failure:
- GPU-required behavior:

## Metrics

Pending:

- request running/waiting:
- queue wait:
- TTFT/TPOT/E2E:
- input/output tokens:
- KV usage:
- prefix cache hit/query:
- GPU utilization/memory:

## Logging

Pending validation of:

```text
timestamp
level
component
request_id
trace_id
exception
structured fields
```

## Tracing

Pending:

- JSONL span parent/child correctness:
- dropped-span count:
- optional OTLP Collector test:

## Lifecycle

Pending:

- request timeout:
- cancellation:
- SIGINT:
- SIGTERM:
- graceful shutdown timeout:

## Reproducibility

Pending:

- manifest.json:
- process.log:
- completion.json:
- config hash:
- git commit/dirty state:

## Docker

Pending:

- build:
- liveness:
- readiness:
- host networking/upstream reachability:

## CI

Pending repository GitHub Actions run.

## Conclusion

Pending live validation.
