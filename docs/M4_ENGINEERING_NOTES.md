# M4 Engineering Notes

Aligned with the original syllabus: SchedulerConfig, Scheduler, Request state, KVCacheManager, ModelRunner, InputBatch, attention backend, and metrics/engine stats.

Design choices:
1. primitive version-stable trace schema;
2. wall-clock + monotonic timestamps;
3. bounded background trace writer;
4. runtime discovery instead of assuming every source path;
5. vLLM custom `scheduler_cls` for the first non-invasive modification;
6. M3 client metrics remain the TTFT/TPOT source of truth;
7. raw JSONL -> summary JSON -> Markdown report;
8. no scheduler optimization is introduced in M4.
