# M6 Engineering Notes

This final M6 pack is aligned with the original production-oriented syllabus.

Design choices:

1. Configuration has explicit precedence and validation.
2. Request IDs and trace IDs use `contextvars`, not function-argument plumbing.
3. Logs are JSON and automatically correlate request/trace context.
4. Prometheus-style histograms retain only buckets/sum/count, not every sample.
5. Request IDs are never metric labels, avoiding cardinality explosion.
6. M3 client timing, M4 runtime queue/KV evidence, M5 GPU telemetry, and vLLM
   native metrics keep explicit source labels instead of being conflated.
7. vLLM `/health` and `/metrics` remain upstream sources; the LLMForge sidecar
   normalizes selected signals instead of replacing vLLM observability.
8. Liveness and readiness are separate.
9. Required dependency failures immediately set readiness false.
10. JSONL application spans are bounded and request-correlated.
11. OpenTelemetry is optional and lazily imported.
12. Reproducibility manifests whitelist environment variables instead of
    serializing credentials.
13. Docker is implemented, while fake K8s/Slurm infrastructure is intentionally
    excluded per the original syllabus.
14. CPU CI avoids installing the CUDA/PyTorch stack and leaves GPU integration
    to the real server.
15. M6 adds observability and maintainability only; it does not introduce the M7
    optimization before evidence exists.
