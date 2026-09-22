# M8 Cross-Engine Validation Report

> Status: CODE READY / LIVE SGLANG VALIDATION PENDING.

## Environment

Pending real execution:

```text
GPU:
driver:
CUDA:
model/revision:
vLLM version:
SGLang version:
Git commit:
```

## Fairness contract

Pending confirmation:

```text
same model
same GPU
same dtype
same M3 client
same request payloads
same arrival schedule
same concurrency
same warmup/repetitions
cache-neutral baseline
```

## Correctness

Pending:

```text
vLLM artifact:
SGLang artifact:
comparison:
exact-text mode:
```

## Neutral baseline

Pending repeated results:

| Metric | vLLM median | SGLang median | SGLang / vLLM |
| --- | ---: | ---: | ---: |
| Request throughput | - | - | - |
| Output tok/s | - | - | - |
| TTFT P50 | - | - | - |
| TTFT P99 | - | - | - |
| TPOT P50 | - | - | - |
| TPOT P99 | - | - | - |
| E2E P50 | - | - | - |
| E2E P99 | - | - | - |

## Mechanism validation

Question:

> Does the selected M7 symptom/bottleneck show the same qualitative direction
> in SGLang?

Pending:

```text
vLLM baseline:
vLLM stressed:
SGLang baseline:
SGLang stressed:
same_direction:
interpretation:
```

## Prefix-reuse experiment

Optional second experiment after the neutral baseline:

```text
vLLM native prefix caching
vs
SGLang native radix cache
```

This is a mechanism comparison, not the neutral engine baseline.

## Conclusion

No cross-engine generality claim until repeated live results and correctness
evidence exist.
