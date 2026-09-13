你说得对，这是 Markdown 渲染器的问题。我用的 `\[ ... \]` 语法在 GitHub/大部分 Markdown 渲染器里不支持，必须改用 `$$ ... $$`。

我把整份报告中所有数学公式的写法全部修正一遍。你直接复制下面这版即可：

---

# LLMForge M2 实验报告
## Transformer Runtime：Resource Accounting、Autoregressive Generation、KV Cache、Prefill/Decode 与 IO-Aware Attention

> **项目**：LLMForge  
> **阶段**：M2｜Transformer Runtime  
> **定位**：从 M1 的 GPU Kernel 性能工程，提升到 Transformer Model Runtime。  
> **核心问题**：Transformer 的数学结构如何变成 GPU workload？为什么 LLM inference 不是简单的一次 `forward()`？KV Cache、Prefill、Decode 和 FlashAttention 分别解决什么问题？

**M2 性能工程闭环**：

```text
Mathematical Model
  ↓
Resource Accounting
  ↓
Naive Execution
  ↓
KV Cache
  ↓
Prefill / Decode
  ↓
Benchmark
  ↓
Operator Profiling
  ↓
Root Cause
  ↓
SDPA / FlashAttention
  ↓
Correctness + Before/After Evidence
```

---

## 目录

1. [M2 在 LLMForge 中的位置](#1-m2-在-llmforge-中的位置)
2. [实验环境与工程约束](#2-实验环境与工程约束)
3. [Transformer Runtime 基础结构](#3-transformer-runtime-基础结构)
4. [MHA 与 GQA](#4-mha-与-gqa)
5. [Resource Accounting](#5-resource-accounting)
6. [Naive Autoregressive Generation](#6-naive-autoregressive-generation)
7. [KV Cache](#7-kv-cache)
8. [KV Cache 是 Compute-Memory Trade-off](#8-kv-cache-是-compute-memory-trade-off)
9. [Prefill 与 Decode](#9-prefill-与-decode)
10. [Prefill / Decode Benchmark 配置](#10-prefill--decode-benchmark-配置)
11. [Benchmark Methodology](#11-benchmark-methodology)
12. [Naive Attention Benchmark](#12-naive-attention-benchmark)
13. [PyTorch Profiler](#13-pytorch-profiler)
14. [长 Context Prefill 的 Root Cause](#14-长-context-prefill-的-root-cause)
15. [SxS Intermediate Memory Evidence](#15-sxs-intermediate-memory-evidence)
16. [Naive Attention 的核心问题](#16-naive-attention-的核心问题)
17. [Decode Runtime 特征](#17-decode-runtime-特征)
18. [SDPA / FlashAttention 优化](#18-sdpa--flashattention-优化)
19. [Naive vs SDPA Before/After](#19-naive-vs-sdpa-beforeafter)
20. [Profiler 确认 FlashAttention](#20-profiler-确认-flashattention)
21. [为什么 SDPA Decode 没有 E2E 加速](#21-为什么-sdpa-decode-没有-e2e-加速)
22. [Split-KV / FlashDecoding 线索](#22-split-kv--flashdecoding-线索)
23. [SDPA Output Sweep](#23-sdpa-output-sweep)
24. [工程与方法论教训](#24-工程与方法论教训)
25. [工程结构与实验产物](#25-工程结构与实验产物)
26. [M2 核心结论](#26-m2-核心结论)
27. [Interview Check](#27-interview-check)
28. [公式速查](#28-公式速查)
29. [局限性](#29-局限性)
30. [M2 Exit Criteria](#30-m2-exit-criteria)
31. [从 M2 到 M3](#31-从-m2-到-m3)

---

## 1. M2 在 LLMForge 中的位置

LLMForge 主线：

```text
M0 Infra Engineering Foundation
        ↓
M1 GPU Execution & CUDA/Triton Performance
        ↓
M2 Transformer Runtime                     ← 本报告
        ↓
M3 High-Performance Serving
        ↓
M4 vLLM Runtime Internals
        ↓
M5 Distributed Inference
```

M1 回答：一个 CUDA/Triton Kernel 为什么快、为什么慢？

M2 回答：一个 Transformer 请求为什么会产生这些 Tensor、Kernel、显存压力和 Runtime 行为？

M2 不再以"继续写更多 CUDA Kernel"为目标，而是将 M1 的知识放回真实 LLM execution path：

```text
Transformer
├── Linear / GEMM
├── RMSNorm
├── Attention
├── Softmax
├── SwiGLU
└── LM Head
      ↓
Autoregressive Generation
      ↓
Prefill / Decode
      ↓
KV Cache
      ↓
Operator / Kernel Workload
```

最终建立：

```text
Model Structure
→ Resource Model
→ Runtime Path
→ Benchmark
→ Profiler
→ Root Cause
→ Runtime Optimization
```

---

## 2. 实验环境与工程约束

主要实验环境：

- GPU：NVIDIA GeForce RTX 4090
- 正式实验：物理 GPU3
- `CUDA_VISIBLE_DEVICES=3`
- Ada Lovelace / SM89
- 显存：24 GB
- Python：3.12.14
- 正式性能实验 dtype：BF16
- 单 GPU
- Git/GitHub：Source of Truth
- Windows：Development Plane
- Linux GPU Server：Execution / Performance Plane

M2 继续遵守 M0/M1 形成的工程纪律：

```text
Correctness before Performance
Measurement before Explanation
Profiler is Evidence, not Benchmark
One Controlled Variable at a Time
```

同时新增一个关键原则：

```text
Kernel Microbenchmark
≠ Model Runtime
≠ Serving System
```

因此当前的：

```text
Prefill Forward Latency
Decode Forward Latency
```

不能提前称为完整 Serving TTFT / TPOT。

---

## 3. Transformer Runtime 基础结构

本阶段实现教育型 Llama-style Decoder-only Transformer：

```text
Input Token IDs
      ↓
Token Embedding
+
Position Embedding
      ↓
Decoder Block × L
├── RMSNorm
├── Q / K / V Projection
├── Causal Self-Attention
├── O Projection
├── Residual
├── RMSNorm
├── SwiGLU MLP
└── Residual
      ↓
Final RMSNorm
      ↓
LM Head
      ↓
Logits
```

本阶段为了集中研究 Runtime control flow，MiniDecoder 使用 Learned Position Embedding，没有同时引入 RoPE 的额外复杂度。

定义：

$$
d = \text{hidden\_size}, \quad d_{ff} = \text{intermediate\_size}
$$

$$
L = \text{num\_layers}, \quad H = \text{num\_attention\_heads}
$$

$$
H_{kv} = \text{num\_key\_value\_heads}
$$

Head dimension：

$$
d_h = \frac{d}{H}
$$

KV width：

$$
d_{kv} = H_{kv} \cdot d_h
$$

---

## 4. MHA 与 GQA

MHA：

$$
H_{kv} = H
$$

GQA：

$$
H_{kv} < H
$$

例如：

$$
H = 8, \quad H_{kv} = 2
$$

即 4 个 Query Heads 共享一组 KV Head。

对 Inference Infra 而言，GQA 的重要性之一是：

$$
M_{KV} \propto H_{kv}
$$

所以 GQA 直接影响：

```text
KV footprint
→ KV bandwidth
→ Max concurrency
→ Serving capacity
```

它不是单纯的模型算法名词。

---

## 5. Resource Accounting

### 5.1 参数量

Bias-free Llama-style Attention：

$$
P_{attn} = 2d^2 + 2d \cdot d_{kv}
$$

SwiGLU MLP：

$$
P_{MLP} = 3d \cdot d_{ff}
$$

每层主要参数：

$$
P_{layer} \approx 2d^2 + 2d \cdot d_{kv} + 3d \cdot d_{ff}
$$

Parameter memory：

$$
M_{weights} = P \times \text{bytes/weight}
$$

BF16：

$$
\text{bytes/weight} = 2
$$

---

### 5.2 KV Cache

一个 token、一个 layer：

$$
K : H_{kv} \cdot d_h
$$

$$
V : H_{kv} \cdot d_h
$$

因此：

$$
M_{KV/token/layer} = 2 H_{kv} d_h b_{kv}
$$

整个模型：

$$
M_{KV} = B \cdot S \cdot L \cdot 2 \cdot H_{kv} \cdot d_h \cdot b_{kv}
$$

其中：

- $$B$$：batch / concurrent sequence count
- $$S$$：sequence length
- $$L$$：layer count
- `2`：K + V
- $$b_{kv}$$：KV element bytes

---

### 5.3 Resource Model 示例

示例架构：

```text
hidden_size         = 4096
intermediate_size   = 11008
num_layers          = 32
num_attention_heads = 32
num_key_value_heads = 8
vocab_size          = 32000
dtype               = BF16
```

实验输出：

```text
parameters=5.802 B
bf16_weight_memory=10.807 GiB
kv_per_token=128.0 KiB
kv_4096=512.0 MiB
prefill_block_flops=55.25 TFLOPs
decode_step_block_flops=13.49 GFLOPs
```

所以：

$$
4096 \times 128\,\text{KiB} = 512\,\text{MiB}
$$

一个请求、4096 context，仅 KV 即达到约 512 MiB。

---

### 5.4 FLOP Model

每 token、每 layer 的主要 Linear FLOPs：

$$
F_{linear} = 2 \left( 2d^2 + 2d \cdot d_{kv} + 3d \cdot d_{ff} \right)
$$

Prefill Attention：

$$
QK^T + PV \approx 4 S^2 d
$$

所以：

$$
F_{prefill} \approx B \cdot L \cdot \left( S \cdot F_{linear} + 4 S^2 d \right)
$$

Decode 单步：

$$
F_{decode} \approx B \cdot L \cdot \left( F_{linear} + 4 S d \right)
$$

这是 first-order Transformer-block model，不包含全部 RMSNorm、RoPE、Softmax elementwise、Sampling 与 LM Head。

Resource Model 加入测试后：

```text
25 passed
```

---

## 6. Naive Autoregressive Generation

Autoregressive LM：

$$
P(x_1, \ldots, x_n) = \prod_i P(x_i \mid x_{<i})
$$

生成 $$T$$ 个新 token 必须进行 $$T$$ 次 autoregressive decision。

Prompt 长度 $$P$$ 时，Naive Generation：

```text
Forward(P)     → y1
Forward(P+1)   → y2
Forward(P+2)   → y3
...
```

累计 model-token evaluations：

$$
N_{naive} = B \left[ T \cdot P + \frac{T(T-1)}{2} \right]
$$

### P=8, T=4

```text
sequence_lengths=(8, 9, 10, 11)
model_token_evaluations=38
final_sequence_tokens=12
evaluation_to_final_token_ratio=3.17x
```

### P=128, T=32

```text
sequence_lengths=(128,129,...,159)
model_token_evaluations=4592
final_sequence_tokens=160
evaluation_to_final_token_ratio=28.70x
```

这里的 `28.70x` 是累计执行 token positions 与最终 sequence length 的结构性比值，**不是 latency speedup**。

Naive full-sequence Attention 还需要累计：

$$
\sum_{t=0}^{T-1} (P+t)^2
$$

所以历史 token 之间的 Attention 也会被重复计算。

---

## 7. KV Cache

Attention：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

历史 token 的：

$$
K_{past}, V_{past}
$$

不会因未来 token 到来而改变。

所以 Decode 新 token 时只需要：

$$
Q_{new}, K_{new}, V_{new}
$$

然后：

$$
Q_{new} [K_{past}; K_{new}]^T
$$

即可。

历史 Query 不需要缓存，因为以后不会再次使用。

因此：

$$
\boxed{\text{KV Cache} = \text{Historical K} + \text{Historical V}}
$$

---

### 7.1 Cache Layout

教学实现保存 compact GQA representation：

$$
\boxed{K, V : [B, H_{kv}, S, d_h]}
$$

Persistent KV 不提前扩展成 $$H$$ 个 heads。

---

### 7.2 Cached Generation

执行由：

```text
P → P+1 → P+2 → ...
```

变成：

```text
Prefill(P)
→ Decode(1)
→ Decode(1)
→ ...
```

累计 model-token evaluations：

$$
N_{cached} = B(P + T - 1)
$$

最后一个生成 token 不需要再次送入模型，除非还要继续生成。

### P=8, T=4

```text
outputs_equal=True
naive_input_lengths=(8, 9, 10, 11)
cached_input_lengths=(8, 1, 1, 1)
cache_lengths=(8, 9, 10, 11)
naive_token_evaluations=38
cached_token_evaluations=11
evaluation_reduction=3.45x
```

### P=128, T=32

```text
outputs_equal=True
naive_token_evaluations=4592
cached_token_evaluations=159
evaluation_reduction=28.88x
```

测试：

```text
31 passed
```

由此验证：

$$
\text{Full Recompute} \rightarrow \text{Prefill} + \text{Incremental Decode}
$$

并保持输出一致。

---

## 8. KV Cache 是 Compute-Memory Trade-off

Naive：

```text
Persistent KV memory ↓
Repeated compute ↑↑
```

KV Cache：

```text
Persistent KV memory ↑
Repeated historical compute ↓↓
```

所以：

$$
\boxed{\text{KV Cache} = \text{用显存换历史计算复用}}
$$

这也是后面 Serving Engine 必须管理 KV Capacity 的根本原因。

---

## 9. Prefill 与 Decode

对于 GEMM：

$$
[M, K] \times [K, N]
$$

FLOPs：

$$
2 M K N
$$

BF16 weight bytes 约：

$$
2 K N
$$

仅考虑 Weight traffic：

$$
AI_{weight} \approx \frac{2 M K N}{2 K N} = M \ \text{FLOP/B}
$$

### Prefill

$$
M = S
$$

执行类似：

$$
[S, d] \times [d, N]
$$

特点：

- large GEMM
- Weight reuse 高
- Tensor Core 更容易被有效利用
- Attention 具有 $$S^2$$ component

### Decode

Batch=1：

$$
M = 1
$$

近似：

$$
[1, d] \times [d, N]
$$

特点：

- skinny GEMM / GEMV
- 持续读取 Weight
- 读取 Historical KV
- 大量 tiny operators/kernels
- framework / launch overhead 更容易显著

这也是 Continuous Batching 后续能够提高 Decode efficiency 的硬件直觉：当 $$M$$ 从 1 提升到更大的 active-token batch 时，同一份 Weight Load 可以服务更多 token。

---

## 10. Prefill / Decode Benchmark 配置

正式 MiniDecoder 性能配置：

```text
vocab_size          = 1024
hidden_size         = 512
intermediate_size   = 1024
num_layers          = 4
num_attention_heads = 8
num_key_value_heads = 2
dtype               = BF16
batch_size          = 1
```

$$
d_h = 64
$$

每 token KV：

$$
4 \times 2 \times 2 \times 64 \times 2 = 2048 \ \text{B} = 2 \ \text{KiB/token}
$$

所以理论：

| Context | KV |
|---:|---:|
| 128 | 0.25 MiB |
| 512 | 1 MiB |
| 2048 | 4 MiB |
| 4096 | 8 MiB |

实测完全一致。

---

## 11. Benchmark Methodology

Prefill / Decode：

- `torch.inference_mode()`
- CUDA Event
- Model Forward only

因此字段是：

```text
PrefillFwd(ms)
DecodeFwd(ms)
```

而不是 TTFT / TPOT。

Output Sweep：

- synchronized wall-clock
- 包含 Python loop
- model forward
- argmax
- cache orchestration

这是 Runtime-level E2E measurement。

---

### 11.1 Benchmark 修复：`eval()` 不等于 Inference Mode

早期脚本存在 execution contract 不一致：

```text
Prompt/Decode Sweep → Autograd enabled
Output Sweep        → @torch.inference_mode()
```

导致同 workload 结果不能公平比较。

修复后统一：

```python
@torch.inference_mode()
```

重要结论：

> `model.eval()` 主要改变 Dropout/BatchNorm 等行为，不会自动关闭 Autograd。

---

### 11.2 Order Sensitivity

升序实验：

```text
Prompt  PrefillFwd  DecodeFwd
128       3.5808      3.4925
512       3.5860      3.5005
2048      3.5820      2.3512
4096     13.1860      2.3409
```

倒序实验：

```text
4096     13.4318      2.3214
2048      3.4956      2.3301
512       2.4013      2.3349
128       2.3873      2.3248
```

小 workload 对运行顺序 / device state 更敏感。

可能因素包括：

- GPU clocks / power state
- allocator/cache state
- library state

但本阶段没有 clock trace，因此不做单因果过度解释。

正式分析使用第二组 warmed/descending baseline。

---

## 12. Naive Attention Benchmark

### 12.1 Prompt / Context Sweep

| Context | Prefill Fwd | Decode Fwd | KV |
|---:|---:|---:|---:|
| 128 | 2.3873 ms | 2.3248 ms | 0.25 MiB |
| 512 | 2.4013 ms | 2.3349 ms | 1 MiB |
| 2048 | 3.4956 ms | 2.3301 ms | 4 MiB |
| 4096 | 13.4318 ms | 2.3214 ms | 8 MiB |

Artifact：

```text
artifacts/benchmarks/prefill_decode/
20260913T083126Z_cb4f2c0/result.json
```

2048 → 4096：

$$
3.4956 \rightarrow 13.4318 \ \text{ms}
$$

约：

$$
\boxed{3.84\times}
$$

sequence length 只翻倍，因此形成强 hypothesis：

> 长 Context 下 Naive Attention 的 $$S^2$$ compute + intermediate traffic 开始主导。

---

### 12.2 Output Sweep

Prompt=512：

| Output | E2E | Output Tok/s |
|---:|---:|---:|
| 1 | 2.4331 ms | 411.01 |
| 8 | 18.7431 ms | 426.82 |
| 32 | 74.1324 ms | 431.66 |

对 8 tokens：

$$
\frac{18.7431 - 2.4331}{7} \approx 2.330 \ \text{ms}
$$

对 32 tokens：

$$
\frac{74.1324 - 2.4331}{31} \approx 2.313 \ \text{ms}
$$

与独立 Decode Forward 的约 2.33 ms 高度一致。

所以 Naive runtime 很好地满足：

$$
\boxed{T_{generation} \approx T_{prefill} + (T-1) T_{decode}}
$$

---

## 13. PyTorch Profiler

对四组 workload：

```text
Prefill S=512
Prefill S=4096
Decode Context=512
Decode Context=4096
```

进行：

- CPU/CUDA profiling
- input shape
- CUDA memory
- FLOPs
- Chrome Trace

分析。

Profiler 用于回答：WHY

普通 benchmark 用于：HOW FAST

---

### 13.1 Naive Profiler Artifacts

```text
artifacts/profiling/prefill_decode_prefill_s512/
20260913T091429Z_1e34778

artifacts/profiling/prefill_decode_prefill_s4096/
20260913T091446Z_1e34778

artifacts/profiling/prefill_decode_decode_s512/
20260913T091502Z_1e34778

artifacts/profiling/prefill_decode_decode_s4096/
20260913T091518Z_1e34778
```

注意：

`record_function()` 产生的 `prefill_s* / decode_s*` 是父范围，与 child ATen / CUDA Kernel attribution 会重叠，不能机械相加。

---

## 14. 长 Context Prefill 的 Root Cause

Naive Prefill S=4096，3 个 profiler iterations：

```text
Self CUDA total = 40.572 ms
```

约：

$$
13.524 \ \text{ms/forward}
$$

关键 Operator：

```text
aten::_softmax      8.129 ms   20.04%
aten::masked_fill_  7.628 ms   18.80%
aten::div           6.994 ms   17.24%
aten::copy_         6.985 ms   17.22%

QK bmm              3.400 ms    8.38%
PV bmm              4.095 ms   10.09%
```

六项：

$$
37.231 \ \text{ms}
$$

占：

$$
\frac{37.231}{40.572} \approx \boxed{91.8\%}
$$

因此长 Context Prefill 的 root cause 非常明确：

$$
\boxed{\text{Naive Attention Pipeline dominates GPU execution}}
$$

---

## 15. SxS Intermediate Memory Evidence

S=4096：

$$
[B, H, S, S] = [1, 8, 4096, 4096]
$$

BF16 score tensor：

$$
1 \times 8 \times 4096^2 \times 2\text{B} = 256 \ \text{MiB}
$$

4 layers：

$$
1 \ \text{GiB}
$$

Profiler 3 iterations：

$$
3 \ \text{GiB}
$$

实际：

```text
aten::_softmax  CUDA Mem = 3.00 GB
aten::div       CUDA Mem = 3.00 GB
QK bmm          CUDA Mem = 3.00 GB
```

完全对应。

注意：

> 这里 3GB 是 profiler 对多层、多 iteration 的累计 operator allocation，不应直接称为单时刻 peak GPU memory。

---

## 16. Naive Attention 的核心问题

Naive pipeline：

```text
QKᵀ
 ↓
write S×S to HBM

Scale / div
 ↓
read + write

Mask
 ↓
read + write

Softmax
 ↓
read + write

P·V
 ↓
read probabilities
```

因此：

$$
\boxed{\text{Long-context bottleneck} = S^2 \ \text{compute} + S^2 \ \text{intermediate multi-pass HBM IO}}
$$

这成为引入 IO-aware / fused Attention 的直接证据。

---

## 17. Decode Runtime 特征

Naive Decode：

```text
Context 512:
Self CUDA ≈ 851.7 us / 3
≈ 284 us/forward

Context 4096:
Self CUDA ≈ 1.080 ms / 3
≈ 360 us/forward
```

GPU work 实际增长：

$$
\approx 1.27\times
$$

但 normal benchmark 仍约：

$$
2.33 \ \text{ms}
$$

说明当前 toy model 中 fixed/runtime overhead 占比很大。

---

### 17.1 Tiny Kernel 与 Launch Fragmentation

Decode：

```text
cudaLaunchKernel calls = 483 / 3
```

所以：

$$
\boxed{161 \ \text{launches/forward}}
$$

许多 Linear 已变成：

```text
[1,512] × [512,N]
```

Profiler 出现：

```text
internal::gemvx...
```

即 GEMV-like execution。

---

### 17.2 Context 增长确实带来成本

`aten::cat`：

```text
512  : 64.5 us
4096 : 115.2 us
```

GQA `repeat_interleave` underlying copy：

```text
512  : 48.8 us
4096 : 126.8 us
```

Attention BMM：

```text
QK:
512  ≈ 24.2 us
4096 ≈ 58.1 us

PV:
512  ≈ 31.6 us
4096 ≈ 73.2 us
```

Context 成本存在，只是未突破 toy runtime 的 framework/launch floor。

---

## 18. SDPA / FlashAttention 优化

Profiler 已经给出真实 root cause，因此此时才引入：

```text
attention_backend:
- naive
- sdpa
```

而不是为了展示"热门优化"。

---

### 18.1 Correctness

MHA：

```text
[H_kv=8] Prefill parity: PASSED
[H_kv=8] Cached Decode parity: PASSED
```

GQA：

```text
[H_kv=2] Prefill parity: PASSED
[H_kv=2] Cached Decode parity: PASSED
```

最终：

```text
All attention backend parity checks passed.
```

说明 SDPA 在 Prefill 与 Cached Decode 都保持语义一致。

---

### 18.2 Compact GQA

SDPA Prefill Profiler：

```text
Q: [1, 4096, 8, 64]
K: [1, 4096, 2, 64]
V: [1, 4096, 2, 64]
```

说明 fused backend 直接接受 compact GQA K/V。

与 naive 教学版 `repeat_interleave()` 相比，它避免显式 materialize expanded KV heads。

---

## 19. Naive vs SDPA Before/After

SDPA：

| Context | SDPA Prefill | SDPA Decode | KV |
|---:|---:|---:|---:|
| 128 | 2.6278 ms | 2.6186 ms | 0.25 MiB |
| 512 | 2.6547 ms | 2.7070 ms | 1 MiB |
| 2048 | 2.5681 ms | 2.6849 ms | 4 MiB |
| 4096 | 2.5595 ms | 2.6819 ms | 8 MiB |

Artifact：

```text
artifacts/benchmarks/prefill_decode/
20260913T093430Z_a9c9cfd/result.json
```

Prefill 对比：

| Context | Naive | SDPA | Naive / SDPA |
|---:|---:|---:|---:|
| 128 | 2.3873 | 2.6278 | 0.91× |
| 512 | 2.4013 | 2.6547 | 0.90× |
| 2048 | 3.4956 | 2.5681 | **1.36×** |
| 4096 | 13.4318 | 2.5595 | **5.25×** |

所以：

$$
\boxed{\text{Optimization benefit is workload-dependent}}
$$

- 小 Prompt：SDPA 略慢
- 中等 Prompt：开始获益
- 长 Prompt：

$$
\boxed{13.43 \ \text{ms} \rightarrow 2.56 \ \text{ms} = 5.25\times}
$$

---

## 20. Profiler 确认 FlashAttention

SDPA Prefill S=4096：

```text
aten::scaled_dot_product_attention
↓
aten::_scaled_dot_product_flash_attention
↓
aten::_flash_attention_forward
↓
pytorch_flash::flash_fwd_kernel
```

因此不是因为 API 名字叫 SDPA 就假设用了 Flash，而是 Profiler 提供 backend evidence。

Artifact：

```text
artifacts/profiling/prefill_decode_prefill_s4096/
20260913T093445Z_a9c9cfd
```

---

### 20.1 GPU Work Reduction

Naive：

$$
40.572 / 3 = 13.524 \ \text{ms/forward}
$$

SDPA：

$$
6.053 / 3 = 2.018 \ \text{ms/forward}
$$

GPU Self CUDA reduction：

$$
\boxed{6.70\times}
$$

普通 benchmark：

$$
13.432 \rightarrow 2.560 \ \text{ms}
$$

即：

$$
\boxed{5.25\times}
$$

Profiler 与 benchmark 的绝对时间不可混用，但两者趋势一致。

---

### 20.2 SxS Intermediate 消失

Naive：

```text
softmax / div / QK
出现 3 GB 级累计 operator allocation
```

SDPA：

```text
scaled_dot_product_attention ≈ 48 MB
_flash_attention_forward    ≈ 49.5 MB
```

Attention output 单层：

$$
1 \times 4096 \times 8 \times 64 \times 2\text{B} = 4 \ \text{MiB}
$$

4 layers × 3 iterations：

$$
48 \ \text{MiB}
$$

说明 execution 从：

```text
完整 S×S intermediate
```

转向：

```text
tile / online softmax / fused accumulation
```

避免完整 score matrix 反复写回 HBM。

---

### 20.3 Kernel Launch 减少

Naive Prefill：

$$
471 / 3 = 157
$$

SDPA：

$$
387 / 3 = 129
$$

减少：28 次 launch / forward。

Fusion 同时改变：

```text
HBM traffic
+
kernel boundaries
+
execution graph fragmentation
```

---

## 21. 为什么 SDPA Decode 没有 E2E 加速

这是 M2 最有价值的反例之一。

4096 Decode：

$$
\text{Naive} = 2.3214 \ \text{ms}
$$

$$
\text{SDPA} = 2.6819 \ \text{ms}
$$

SDPA E2E 反而慢约 15.5%。

但 GPU Self CUDA：

Naive：

$$
1.080 / 3 = 0.360 \ \text{ms}
$$

SDPA：

$$
0.865 / 3 = 0.288 \ \text{ms}
$$

GPU work 减少约 20%。

Launch：

```text
Naive: 483/3 = 161
SDPA : 423/3 = 141
```

也减少。

所以：

$$
\boxed{\text{GPU Kernel Win} \not\Rightarrow \text{Runtime E2E Win}}
$$

当前 toy Decode 中 GPU work 只有约 0.3ms，而整个 model-forward 在 2～3ms 级。

host-side dispatch、backend setup、framework orchestration 和 many tiny ops 的固定成本能够压过 GPU savings。

这再次证明：

> 优化的是 bottleneck，而不是"最热门的算子"。

---

## 22. Split-KV / FlashDecoding 线索

SDPA Decode 4096 Profiler 出现：

```text
flash_fwd_splitkv_kernel
flash_fwd_splitkv_combine_kernel
```

这提供了 Split-KV / FlashDecoding 的执行直觉：

```text
Q length = 1
KV length = very long
        ↓
Partition long KV
        ↓
Parallel partial attention
        ↓
Combine
```

M2 不继续实现 FlashDecoding，但已经从真实 backend 中观察到了这种策略。

Artifact：

```text
artifacts/profiling/prefill_decode_decode_s4096/
20260913T093458Z_a9c9cfd
```

---

## 23. SDPA Output Sweep

| Output | E2E | Tok/s |
|---:|---:|---:|
| 1 | 2.7048 ms | 369.72 |
| 8 | 21.4513 ms | 372.94 |
| 32 | 56.9227 ms | 562.17 |

32-token amortized throughput 明显高于 isolated decode-step inverse latency。

本阶段不把它解释成"Decode 单步突然变快"。

可能涉及：

- longer-run device state
- CPU/GPU enqueue overlap
- runtime amortization

但缺少 Nsight Systems timeline，因此保留到后续 System/Serving profiling 阶段验证。

正确结论是：

$$
\boxed{\text{Microbenchmark} \neq \text{Runtime} \neq \text{Serving}}
$$

---

## 24. 工程与方法论教训

### 24.1 `eval()` 不等于关闭 Autograd
正式 benchmark 需要统一 `torch.inference_mode()`。

### 24.2 Position Offset 是 Cached Decode Correctness 的关键
Decode 输入长度虽然只有 1，但它的 logical position 应为 `past_length`，而不是重新从 0 开始。

### 24.3 Cached Causal Mask 与 Full-sequence Mask 不同
Prefill: `Qlen = Klen = S`；Decode: `Qlen = 1, Klen = S+1`。Mask 必须基于 absolute query/key position 构造。

### 24.4 Boolean Mask 语义需检查 API Contract
自定义 naive mask：True = forbidden；SDPA boolean mask：True = allowed。所以 generic cached path 需要反转 mask。

### 24.5 Profiler Parent Range 不能与 Child Operator 相加
`prefill_s4096` 是父范围，不是 GPU utilization。

### 24.6 Profiler Allocation 不等于 Peak GPU Memory
3GB 是多层、多 iteration 累计 allocation，不是单时刻 peak。

### 24.7 Educational KV Cache 不是 Production KV Manager
当前 `torch.cat()` 每步都会重新分配与复制历史 KV。生产系统会继续发展到 Preallocated KV → Paged KV → Block Table → KV Cache Manager。

### 24.8 Educational GQA Expansion 不是 Production 实现
Naive path 的 `repeat_interleave()` 是教学实现。SDPA 已验证 compact GQA K/V 可以直接交给 fused backend。

---

## 25. 工程结构与实验产物

M2 核心代码：

```text
src/llmforge/
├── resource_model/
│   ├── __init__.py
│   └── transformer.py
│
└── model_execution/
    ├── __init__.py
    └── mini_decoder.py
```

测试：

```text
tests/
├── test_transformer_resource_model.py
└── test_mini_decoder.py
```

脚本：

```text
scripts/
├── demo_naive_generation.py
├── demo_kv_cache_generation.py
├── benchmark_prefill_decode.py
├── profile_prefill_decode.py
└── check_attention_backend_parity.py
```

报告：

```text
reports/
└── prefill_decode_analysis.md
```

Artifacts：

```text
artifacts/
├── benchmarks/prefill_decode/
└── profiling/prefill_decode_*/
```

---

## 26. M2 核心结论

1. **Autoregressive dependency 不会被 KV Cache 消除。** KV Cache 消除的是历史 K/V 与历史 hidden-state 的重复计算。

2. **KV Cache 是 Compute-Memory Trade-off。** 通过持久化 K/V 减少重复计算，但占用随 $$B \times S \times L \times H_{kv}$$ 增长的 GPU memory。

3. **GQA 是 Serving Resource Design。** 减少 $$H_{kv}$$ 会直接降低 KV footprint 与 KV bandwidth pressure。

4. **Prefill 与 Decode 是两种不同 GPU workload。** Prefill 更容易形成大 GEMM 和长序列 Attention；Decode 更接近 skinny GEMM/GEMV + KV read + many tiny kernels。

5. **长 Context Naive Attention 的问题不只是 FLOPs。** 完整 $$S \times S$$ intermediates 的多次 HBM materialization 是关键瓶颈。

6. **FlashAttention 的核心是 IO-aware execution。** Exact Attention 仍大体是 $$O(S^2 d)$$ compute，但通过 tiling、online softmax 和 fusion 避免完整 score matrix 写回 HBM。

7. **Optimization Benefit 是 workload-dependent 的。** 小 shape SDPA 可以更慢；S=4096 Prefill 则获得约 5.25× speedup。

8. **Kernel-level win 不等于 Runtime-level win。** Decode SDPA GPU-side work 更少，但 toy E2E forward 反而更慢。

9. **性能工程必须跨层看问题。** Model → Operator → Kernel → Runtime 的 bottleneck 可能完全不同。

---

## 27. Interview Check

**为什么需要 KV Cache？**
历史 token 的 K/V 在后续 Decode 中保持不变，可以复用。KV Cache 避免每生成一个 token 都重新计算完整历史 sequence。

**为什么不缓存 Q？**
历史 Query 后续不会再使用；历史 K/V 会被未来 Query 持续读取。

**KV Cache 怎么计算？**

$$
M_{KV} = B \cdot S \cdot L \cdot 2 \cdot H_{kv} \cdot d_h \cdot \text{bytes}
$$

**为什么 GQA 对 Serving 重要？**
因为 KV memory 与 $$H_{kv}$$ 近似成正比，直接影响并发容量与 memory bandwidth pressure。

**Prefill 与 Decode 为什么不同？**
Prefill 是 large-token-batch compute，Linear 更接近大 GEMM；Decode 一步通常只有一个 token，Linear 更接近 GEMV，同时持续读 KV，并承担大量 tiny-op launch/runtime overhead。

**为什么 Naive Prefill 4096 很慢？**
Profiler 显示 Attention pipeline 占约 91.8% Self CUDA time，并产生多 GB 累计 $$S \times S$$ operator allocations。

**FlashAttention 优化了什么？**
主要优化 HBM IO，不是将 exact attention 从 $$O(S^2)$$ 变为线性复杂度。它通过 tiling + online softmax + fused output accumulation 避免完整 attention matrix materialization。

**为什么 SDPA 小 shape 可能慢？**
Fused backend 也有 fixed overhead。小序列时 $$S^2$$ intermediate 不大，IO savings 可能不足以抵消 backend/dispatch 固定成本。

**为什么 Decode GPU 更快但 E2E 更慢？**
Toy Decode 的 GPU work 本身很小，framework/backend/dispatch 固定成本能够压过 kernel-level savings。

**PyTorch Profiler 与 Nsight Compute 区别？**
PyTorch Profiler 更适合 Model/Operator 层；Nsight Compute 更适合单 CUDA kernel 的 SM/DRAM/cache/occupancy/warp-stall 级分析。

---

## 28. 公式速查

Parameter memory：

$$
M_{weights} = P \times \text{bytes/weight}
$$

KV Cache：

$$
M_{KV} = B \cdot S \cdot L \cdot 2 \cdot H_{kv} \cdot d_h \cdot b_{kv}
$$

Naive generation：

$$
N_{naive} = B \left[ T \cdot P + \frac{T(T-1)}{2} \right]
$$

Cached generation：

$$
N_{cached} = B(P + T - 1)
$$

Prefill Attention：

$$
F_{attn, prefill} \approx 4 S^2 d
$$

Decode Attention：

$$
F_{attn, decode} \approx 4 S d
$$

BF16 Linear weight-only AI approximation：

$$
AI_{weight} \approx M
$$

Generation runtime approximation：

$$
T_{generation} \approx T_{prefill} + (T-1) T_{decode}
$$

---

## 29. 局限性

**Educational Model**：MiniDecoder 不是真实 7B/8B production model，不能把绝对 latency 直接推广到真实模型。

**Learned Position Embedding**：本阶段没有完整实现 Llama-style RoPE。

**`torch.cat()` KV Cache**：语义正确，但不是 production KV memory management。

**Naive GQA Expansion**：`repeat_interleave()` 是教学实现，会引入额外 temporary memory/copy。

**没有 Scheduler / Queue / Concurrency**：当前是 single request、batch=1、single GPU，还没有 Continuous Batching、Paged KV、Prefix Cache、Online Arrival、Scheduler。

**当前不是完整 TTFT / TPOT**：只测 Model Runtime，而不是 Server request path。

**Long-run CPU/GPU overlap 尚未用 Nsight Systems 系统验证**：保留到后续 Serving/System Profiling。

---

## 30. M2 Exit Criteria

| M2 Requirement | Evidence | 状态 |
|---|---|---|
| Transformer Components | MiniDecoder | ✅ |
| RMSNorm / Attention / MLP / LM Head | 实现 | ✅ |
| MHA / GQA | 实现 + parity | ✅ |
| Resource Accounting | Parameter / KV / FLOPs | ✅ |
| Naive Generation | 38 / 4592 evaluations | ✅ |
| KV Cache | 11 / 159 evaluations | ✅ |
| Naive vs Cached Correctness | outputs_equal | ✅ |
| Prefill / Decode Split | execution trace | ✅ |
| Prompt Sweep | 128 / 512 / 2048 / 4096 | ✅ |
| Output Sweep | 1 / 8 / 32 | ✅ |
| KV Scaling | 0.25 / 1 / 4 / 8 MiB | ✅ |
| PyTorch Profiler | Prefill + Decode | ✅ |
| Operator Breakdown | Complete | ✅ |
| Long-context Root Cause | S² Attention IO | ✅ |
| SDPA / Flash Backend | Profiler verified | ✅ |
| MHA/GQA SDPA Correctness | Prefill + Decode | ✅ |
| Before / After | 4096 Prefill 5.25× | ✅ |
| Runtime Trade-off | Decode kernel win ≠ E2E win | ✅ |
| Code + Data + Document + Knowledge | Complete | ✅ |

因此：

$$
\boxed{\text{M2 Transformer Runtime} = \text{COMPLETED}}
$$

---

## 31. 从 M2 到 M3

M2 已经理解：

```text
Token
→ Transformer
→ Prefill
→ KV Cache
→ Decode
→ Attention Backend
→ GPU Operators
```

M3 将回答：

> 多个用户同时请求时，谁先执行？如何组成 batch？KV 不够怎么办？怎样平衡 TTFT、TPOT、throughput 与 tail latency？

抽象层继续提升：

```text
M1  Kernel Performance
        ↓
M2  Model Runtime
        ↓
M3  Serving System
```

M2 的概念会继续映射为：

```text
Prefill              → TTFT
Decode               → TPOT / ITL
KV Cache             → Serving Capacity / Paged KV
Different Lengths    → Continuous Batching
Long Prompt          → Chunked Prefill
Concurrency          → Latency / Throughput Trade-off
```

因此 M2 真正完成的不是"会调用 SDPA"，而是形成了一套可以迁移到真实 Serving Engine 的性能工程能力：

```text
观察 Runtime 现象
    ↓
理解 Model Shape
    ↓
建立 Hypothesis
    ↓
用 Profiler 找 Evidence
    ↓
定位 Root Cause
    ↓
选择与 Root Cause 对应的设计变化
    ↓
Correctness
    ↓
Benchmark Before / After
    ↓
解释收益、边界与 Trade-off
```

下一阶段将正式从教育型 Runtime 进入：

$$
\boxed{\text{Real Model} + \text{Real Serving Engine} + \text{Real Workload}}
$$

也就是 LLMForge 的 M3：High-Performance Serving。

