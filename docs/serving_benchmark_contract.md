# LLMForge Serving Benchmark Contract

## Measurement Boundary

Serving latency is measured at the benchmark client.

t_request:
    request transmission begins

t_first_token:
    first streamed output arrives

t_last_token:
    final streamed output arrives

## Latency Metrics

$TTFT =
    t_first_token - t_request$

$ITL_i =
    t_i - t_(i-1)$

$E2E =
    t_last_token - t_request$

$TPOT =
    (E2E - TTFT) /
    (num_output_tokens - 1)$

## Throughput Metrics

$Request Throughput =
    completed_requests / wall_time$

$Output Token Throughput =
    total_output_tokens / wall_time$

$Total Token Throughput =
    (input_tokens + output_tokens) / wall_time$

## Distribution Metrics

Report at least:

- P50
- P95
- P99

for latency-sensitive metrics.

## Reproducibility

Every formal serving experiment records:

model
model revision
engine
engine version
dtype
GPU
GPU count
tensor parallel size
max model length
GPU memory utilization
scheduler parameters
prefix-cache state
prompt-length distribution
output-length distribution
request rate
concurrency
random seed
Git commit
raw per-request samples

## Cache Discipline

Experiments that do not intentionally study prefix-cache
reuse must avoid accidental cache reuse.

## Cold Start

Model loading and engine initialization are excluded from
steady-state serving measurements unless the experiment
explicitly studies cold start.