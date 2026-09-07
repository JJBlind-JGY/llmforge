# LLMForge Benchmark Methodology

## 1. Purpose

Performance results in LLMForge must be reproducible, traceable, and tied to a specific source revision, hardware environment, software stack, and workload.

A performance number without its experimental context is not considered valid evidence.

---

## 2. Experiment Provenance

Every formal benchmark should record at least:

- Git commit and dirty state
- GPU model and count
- NVIDIA driver
- CUDA driver-supported version
- CUDA toolkit / nvcc
- Python version
- PyTorch version and CUDA runtime
- OS / kernel
- CPU / system memory
- GPU topology when relevant
- workload configuration
- dtype / precision mode
- warmup count
- measurement iteration count

---

## 3. Clean Benchmark Environment

Formal benchmark runs must use:

- a clean Git working tree;
- an explicitly selected GPU via `CUDA_VISIBLE_DEVICES`;
- no active Conda environment;
- no unrelated process occupying the benchmark GPU.

Learning/debug runs may relax these conditions, but their results must not be used as canonical performance baselines.

---

## 4. CUDA Asynchronous Execution

CUDA operations are normally asynchronous with respect to the host CPU.

Therefore:

```python
start = time.perf_counter()
operation()
end = time.perf_counter()
```

**does not measure GPU kernel execution latency reliably.**

LLMForge distinguishes:

- **CPU enqueue latency** – CPU-side time required to submit work.
- **Synchronized wall latency** – CPU-observed latency after explicitly waiting for GPU completion.
- **CUDA event latency** – GPU-stream elapsed time measured using CUDA events.

CUDA event timing is the default method for GPU microbenchmarks.

---

## 5. Warmup

Formal steady-state benchmarks must warm up the workload before measurement.

Warmup reduces contamination from effects such as:

- CUDA context initialization
- library initialization
- kernel selection
- caching
- allocator initialization

Cold-start latency must be measured separately when it is the target metric.

---

## 6. Repetition and Statistics

LLMForge never reports only the fastest run.

Raw samples should be retained whenever practical.

Reported statistics may include:

- mean
- median / P50
- P95
- P99
- min
- max

Median is generally used as the main microbenchmark latency statistic.

Serving benchmarks additionally emphasize tail latency.

---

## 7. Workload Definition

Performance numbers are only meaningful together with workload parameters.

GPU kernel benchmarks should record shape, dtype and operation semantics.

Serving benchmarks must later record at least:

- model
- concurrency
- prompt length
- output length
- arrival pattern
- shared-prefix behavior

---

## 8. Correctness Before Performance

Optimized implementations must be compared against a trusted reference before their performance results are accepted.

Tolerance must be appropriate for the tested numerical precision.

---

## 9. Performance Models

When relevant, LLMForge records:

- FLOPs
- transferred bytes
- effective bandwidth
- effective GFLOPS / TFLOPS
- arithmetic intensity

These values are used together with profiler evidence to classify workloads as compute-bound, memory-bound, communication-bound, or under-utilized.

---

## 10. Raw Data First

Preferred experiment flow:

```
raw artifact
    ↓
analysis
    ↓
table / figure
    ↓
report
```

Screenshots are never the sole experimental record.

---

## 11. Baseline Discipline

Before optimization:

- freeze the baseline;
- record the commit;
- save raw benchmark data;
- verify correctness;
- define the workload.

After optimization, the same methodology and workload must be reused.

---

## 12. Current Canonical GPU Node

The primary GPU benchmark node is recorded automatically through `llmforge-env`.

Machine-specific values must not be manually assumed in benchmark scripts.