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

## 2. CUDA GEMM

### 2.1 Benchmark

| Variant | N | Latency (ms) | Effective TFLOPS |
| --- | ---: | ---: | ---: |
| Naive | 4096 | 24.4531 | 5.62 |
| Tiled16 | 4096 | 18.7955 | 7.31 |
| Tiled32 | 4096 | 21.2326 | 6.47 |
| PyTorch FP32 IEEE | 4096 | 2.5846 | 53.18 |

Shared-memory tiling improved the native CUDA implementation by approximately
1.30x over the naive kernel.

TILE=32 did not outperform TILE=16 despite providing greater theoretical data
reuse. Nsight Compute showed that TILE=32 was limited to 66.67% theoretical
occupancy, while TILE=16 achieved approximately 100% occupancy.

### 2.2 Profiling Observation

The kernels were not limited by off-chip DRAM bandwidth. Nsight Compute
reported less than 1% DRAM throughput for all three variants.

The naive implementation achieved an 86.20% L1 hit rate, demonstrating that
hardware caching already recovered substantial reuse that is not represented
by a source-level global-load model.

Tiled16 shifted reuse toward explicitly managed shared memory and improved
throughput, but introduced substantial shared-memory/MIO instruction pressure.

The vendor-library FP32 IEEE baseline reached 53.18 TFLOPS, approximately
7.27x the throughput of Tiled16, illustrating that shared-memory tiling alone
is only the first level of modern GEMM optimization.


## 3. Triton GEMM

### 3.1 FP32 IEEE Benchmark

| N | Triton TFLOPS | PyTorch TFLOPS | Triton / PyTorch | Best Triton Config |
| ---: | ---: | ---: | ---: | --- |
| 512 | 18.48 | 21.85 | 84.6% | 32×64×32, 2 warps, 5 stages |
| 1024 | 33.29 | 36.79 | 90.5% | 128×64×32, 4 warps, 4 stages |
| 2048 | 46.67 | 54.65 | 85.4% | 128×64×32, 4 warps, 4 stages |
| 4096 | 41.57 | 48.67 | 85.4% | 128×64×32, 4 warps, 4 stages |

For N=4096, the Triton implementation reached approximately 5.68x
the throughput of the project's hand-written CUDA Tiled16 kernel
(41.57 vs 7.31 TFLOPS).

For N=512, the autotuner selected a smaller 32×64 output tile.
This produces 128 Triton programs for a 512×512 output, matching
the GPU's 128 SMs and avoiding the coarse-grained under-utilization
that would result from larger tiles.

For N>=1024, a 128×64 output tile became viable because the workload
already exposed sufficient program-level parallelism.

The current search space keeps BLOCK_SIZE_K=32 and GROUP_SIZE_M=8
fixed, so the experiment should not be interpreted as a global
optimization over all Triton GEMM parameters.


## 4. Triton BF16 Tensor-Core GEMM

| N | Triton BF16 TFLOPS | PyTorch BF16 TFLOPS | Triton / PyTorch | Best Config |
| ---: | ---: | ---: | ---: | --- |
| 512 | 36.71 | 37.45 | 98.0% | 32×64×32, 2 warps, 5 stages |
| 1024 | 87.38 | 123.36 | 70.8% | 32×64×32, 2 warps, 5 stages |
| 2048 | 164.48 | 169.47 | 97.1% | 128×128×32, 4 warps, 4 stages |
| 4096 | 174.54 | 154.81 | 112.7% | 128×128×32, 4 warps, 4 stages |

Switching from FP32 IEEE to BF16 substantially changes the hardware
execution path. BF16 inputs allow `tl.dot` to use Tensor-Core-oriented
matrix execution while accumulating into FP32.

For N=4096, Triton improved from 41.57 TFLOPS in FP32 IEEE to
174.54 effective TFLOPS in BF16.

The result should not be interpreted as a universal Triton advantage
over vendor libraries. Performance remains strongly shape-dependent:
at N=1024, Triton reached only 70.8% of the PyTorch baseline, while
at N=2048 it reached 97.1%.

Autotuning also selected different configurations by workload size,
demonstrating that tile size, program-level parallelism, warp count,
pipeline depth, and dtype jointly affect kernel performance.


Benchmark results determine how fast each version runs, while Nsight Compute
provides evidence explaining why those differences occur