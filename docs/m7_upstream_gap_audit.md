# M7 Upstream Gap Audit

This document exists to stop LLMForge from presenting an existing vLLM engine
argument as an "original optimization".

Pinned experiment line: **vLLM 0.29.0**.

Before selecting the M7 headline optimization, re-read the pinned official
engine arguments and scheduler source.

Known upstream controls that are **baseline controls**, not contributions:

```text
max_num_batched_tokens
max_num_scheduled_tokens
enable_chunked_prefill
long_prefill_token_threshold
scheduling_policy = fcfs / priority
max_num_queued_reqs
max_num_queued_tokens
scheduler_reserve_full_isl
watermark
```

Therefore the following alone are not acceptable headline claims:

```text
"we changed max_num_batched_tokens"
"we enabled priority scheduling"
"we set a KV watermark"
"we limited queued tokens"
```

The reference implementation included in M7 is different in kind: it changes
the effective scheduling budget **per engine step** based on live mixed
prefill/decode state. Even that implementation remains only a candidate until
the eight-condition selection gate is satisfied.

If the real evidence instead points to fairness, CPU overhead, a GPU kernel, or
multi-GPU communication, the project must select that direction and may leave
the adaptive-budget candidate as a negative/unselected experiment.
