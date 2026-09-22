# M4 vLLM Integration

The first M4 integration does not edit upstream scheduling logic. vLLM exposes a custom scheduler class through `scheduler_cls`; LLMForge uses:

```text
llmforge.runtime.integrations.vllm.tracing_scheduler.TracingScheduler
```

The subclass calls upstream behavior first, then records request counts, scheduled tokens, KV usage, preemptions, and completion events.

Only if a later M4 question requires an internal decision that cannot be observed at this boundary should a small pinned-version source patch be added.

Server validation sequence:
1. install LLMForge editable in the pinned vLLM environment;
2. inspect the installed vLLM source map;
3. run a tiny trace in `sync` mode;
4. validate token/event accounting;
5. switch to `buffered` mode;
6. measure instrumentation overhead against the uninstrumented baseline;
7. run the matching M3 workload;
8. analyze the trace and render `reports/runtime_trace.md`.
