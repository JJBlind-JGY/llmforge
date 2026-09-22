# M5 Engineering Notes

This final M5 pack is aligned to the original distributed-inference syllabus.

Design choices:

1. Topology is captured before benchmarking.
2. Same-NUMA and cross-NUMA TP2 are separate experiment conditions.
3. Collective timing uses CUDA events and retains every rank's raw samples.
4. Per-iteration collective latency uses the slowest rank.
5. algBW/busBW semantics match NVIDIA nccl-tests definitions.
6. Serving comparisons reuse the M3 result schema instead of inventing a new
   latency definition.
7. TP=3 is preflight-validated against Q/KV head sharding rather than blindly
   launched.
8. Two-replica serving is represented as vLLM DP=2 for the dense Qwen3-8B
   experiment line.
9. GPU utilization is recorded separately from client latency.
10. NCCL microbenchmarks are never substituted for actual model communication
    profiling.
11. Short profiler runs are separated from performance benchmark runs because
    profiling adds overhead.
12. M5 does not yet introduce an optimization; it builds evidence for M7.
