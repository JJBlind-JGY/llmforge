# Benchmark Methodology

## Environment

Every performance report records:

```text
hardware
driver
CUDA
framework/runtime version
model
model revision
dtype
Git commit
relevant configuration
```

## Warmup

Cold start and warm steady-state are separate experiments.

Model download/load time is not mixed into steady-state serving latency unless
the experiment is explicitly about startup.

## CUDA timing

CUDA work is asynchronous.

Microbenchmarks use correct synchronization or CUDA events rather than raw CPU
wall time around an asynchronous launch.

## Repetition

Keep multiple raw repetitions.

M7/M8 comparative experiments require at least three runs per arm/engine;
five are preferred.

Do not report only the best run.

## Serving metrics

Primary serving metrics:

```text
TTFT
ITL
TPOT
E2E
request throughput
output token throughput
total token throughput
P50 / P95 / P99
```

Client-observed streaming timestamps define the primary M3/M8 latency boundary.

## Workload contract

A performance number is incomplete without:

```text
model
GPU
concurrency
prompt length/distribution
output length/distribution
arrival process/request rate
shared-prefix ratio
cache policy
scheduler/runtime configuration
```

## Raw-data-first rule

```text
raw result
   ↓
analysis script
   ↓
derived table/figure
   ↓
report
```

Do not preserve only screenshots.

## Baseline first

Freeze the baseline before optimizing.

## Trade-offs

Any improvement must show relevant regressions.

Example:

```text
TPOT P99  -18%
throughput -3%
TTFT P99   +4%
```

is more informative than reporting only `TPOT -18%`.

## Cross-engine fairness

M8 compares engines using the same LLMForge client and workload whenever
possible.

The first neutral comparison disables engine-specific prefix reuse.

Unavoidable engine differences are documented rather than hidden.
