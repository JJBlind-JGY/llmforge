# LLMForge GPU Kernel Performance Report

## 1. CUDA Reduction

### 1.1 Environment

- GPU: NVIDIA GeForce RTX 4090
- Compute Capability: 8.9
- CUDA Toolkit: 12.1
- Host Compiler: GCC/G++ 11.4
- CUDA Architecture: sm_89
- Nsight Compute: 2023.1
- Block Size: 256
- Input Elements: 16,777,216 FP32

### 1.2 Optimization Evolution

| Variant | Latency (ms) | Useful Input GB/s | Global Atomic Ops | Speedup vs Atomic |
| --- | ---: | ---: | ---: | ---: |
| Atomic | 23.1096 | 2.90 | 16,777,216 | 1.00x |
| Shared Interleaved | 0.1212 | 553.63 | 65,536 | 190.65x |
| Shared Sequential | 0.0963 | 697.19 | 65,536 | 240.08x |
| Warp Shuffle | 0.0864 | 776.72 | 65,536 | 267.47x |

### 1.3 Root Cause Analysis

The atomic baseline performs one global atomic update per input element,
creating severe contention on a single output location.

Block-local shared-memory aggregation reduces global atomic updates from
16,777,216 to 65,536.

The interleaved shared-memory variant remained slower than sequential
addressing. Nsight Compute reported 6,881,280 excessive shared-memory
wavefronts, accounting for 70% of all shared wavefronts, confirming that the
interleaved access pattern caused substantial shared-memory bank conflicts.

Sequential addressing removed this profiler warning and reduced benchmark
latency from 0.1212 ms to 0.0963 ms.

Warp-level reduction further reduced shared-memory communication and
block-wide synchronization by performing intra-warp aggregation through
register shuffle operations. Latency decreased to 0.0864 ms.

### 1.4 Bottleneck Migration

The warp-level kernel reached 87.68% DRAM throughput while compute throughput
was only 35.20%. Its achieved occupancy was 72.87%, lower than the shared
kernel's 88.57%, yet it remained faster.

This demonstrates that higher occupancy is not inherently better. After
removing atomic contention, shared-memory conflicts, and most block-local
reduction overhead, performance increasingly became constrained by input
memory access.

### 1.5 Conclusion

Reduction optimization was primarily achieved by changing the communication
hierarchy:

```text
Global atomic contention
        ↓
Block-local aggregation
        ↓
Bank-friendly shared-memory access
        ↓
Warp register shuffle
        ↓
Memory-dominated execution
```


Benchmark results determine how fast each version runs, while Nsight Compute
provides evidence explaining why those differences occur