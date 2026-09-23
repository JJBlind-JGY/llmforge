# Transformer Runtime Performance Study

## 1. Objective

This report studies the runtime behavior of a Transformer decoder from mathematical structure to GPU workload. The main questions are:

1. How does Transformer math become GPU work?
2. Why is LLM inference not just a single `forward()`?
3. What do KV Cache, Prefill, Decode, and FlashAttention actually solve?
4. Why does a kernel-level win not always become a runtime-level or end-to-end win?

The experimental loop is:

```text
Model Structure
  -> Resource Accounting
  -> Naive Execution
  -> KV Cache
  -> Prefill / Decode
  -> Benchmark
  -> Operator Profiling
  -> Root Cause
  -> SDPA / FlashAttention
  -> Correctness + Before/After Evidence
```

Measurement discipline:

```text
Correctness before Performance
Measurement before Explanation
Profiler is Evidence, not Benchmark
One Controlled Variable at a Time
Kernel Microbenchmark != Model Runtime != Serving System
```

`PrefillFwd(ms)` and `DecodeFwd(ms)` in this report mean model-forward latency only. They are not full serving TTFT / TPOT.

---

## 2. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Device | Physical GPU3 |
| `CUDA_VISIBLE_DEVICES` | 3 |
| Architecture | Ada Lovelace / SM89 |
| Memory | 24 GB |
| Python | 3.12.14 |
| dtype | BF16 |
| GPU count | 1 |
| Source of truth | Git / GitHub |
| Development plane | Windows |
| Execution / performance plane | Linux GPU server |

MiniDecoder performance configuration:

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

Head dimension:

$$
d_h = 64
$$

KV per token:

$$
4 \times 2 \times 2 \times 64 \times 2 = 2048 \ \text{B} = 2 \ \text{KiB/token}
$$

Theoretical KV scaling for MiniDecoder:

| Context | KV |
|---:|---:|
| 128 | 0.25 MiB |
| 512 | 1 MiB |
| 2048 | 4 MiB |
| 4096 | 8 MiB |

Measured KV usage matched theory.

---

## 3. Resource Model

### 3.1 Parameter Memory

Bias-free Llama-style Attention:

$$
P_{attn} = 2d^2 + 2d \cdot d_{kv}
$$

SwiGLU MLP:

$$
P_{MLP} = 3d \cdot d_{ff}
$$

Per-layer parameters:

$$
P_{layer} \approx 2d^2 + 2d \cdot d_{kv} + 3d \cdot d_{ff}
$$

Parameter memory:

$$
M_{weights} = P \times \text{bytes/weight}
$$

For BF16:

$$
\text{bytes/weight} = 2
$$

### 3.2 KV Cache

Per token, per layer:

$$
K : H_{kv} \cdot d_h
$$

$$
V : H_{kv} \cdot d_h
$$

Therefore:

$$
M_{KV/token/layer} = 2 H_{kv} d_h b_{kv}
$$

Full model:

$$
M_{KV} = B \cdot S \cdot L \cdot 2 \cdot H_{kv} \cdot d_h \cdot b_{kv}
$$

where:

- $B$: batch / concurrent sequence count
- $S$: sequence length
- $L$: layer count
- $2$: K + V
- $b_{kv}$: KV element bytes

### 3.3 FLOP Model

Per token, per layer linear FLOPs:

$$
F_{linear} = 2 \left( 2d^2 + 2d \cdot d_{kv} + 3d \cdot d_{ff} \right)
$$

Prefill Attention:

$$
QK^T + PV \approx 4 S^2 d
$$

Therefore:

$$
F_{prefill} \approx B \cdot L \cdot \left( S \cdot F_{linear} + 4 S^2 d \right)
$$

Decode single step:

$$
F_{decode} \approx B \cdot L \cdot \left( F_{linear} + 4 S d \right)
$$

This is a first-order Transformer-block model. It does not include all RMSNorm, RoPE, Softmax elementwise, sampling, or LM Head costs.

### 3.4 Resource Model Example

Example architecture:

```text
hidden_size         = 4096
intermediate_size   = 11008
num_layers          = 32
num_attention_heads = 32
num_key_value_heads = 8
vocab_size          = 32000
dtype               = BF16
```

Measured resource model output:

```text
parameters=5.802 B
bf16_weight_memory=10.807 GiB
kv_per_token=128.0 KiB
kv_4096=512.0 MiB
prefill_block_flops=55.25 TFLOPs
decode_step_block_flops=13.49 GFLOPs
```

Therefore:

$$
4096 \times 128 \ \text{KiB} = 512 \ \text{MiB}
$$

A single request with 4096 context uses about 512 MiB of KV cache alone.

---

## 4. Autoregressive Generation

Autoregressive LM:

$$
P(x_1, \ldots, x_n) = \prod_i P(x_i \mid x_{<i})
$$

Generating $T$ new tokens requires $T$ autoregressive decisions.

For prompt length $P$, naive generation runs:

```text
Forward(P)     -> y1
Forward(P+1)   -> y2
Forward(P+2)   -> y3
...
```

Cumulative model-token evaluations:

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

`28.70x` is a structural ratio between cumulative executed token positions and final sequence length. It is not a latency speedup.

With KV Cache, cumulative model-token evaluations become:

$$
N_{cached} = B(P + T - 1)
$$

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

Core verified result:

$$
\text{Full Recompute} \rightarrow \text{Prefill} + \text{Incremental Decode}
$$

with identical outputs.

---

## 5. KV Cache

Attention:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

Historical token K/V:

$$
K_{past}, V_{past}
$$

do not change when future tokens arrive.

Therefore, when decoding a new token, only:

$$
Q_{new}, K_{new}, V_{new}
$$

are needed, followed by:

$$
Q_{new} [K_{past}; K_{new}]^T
$$

Historical queries do not need to be cached because they are never reused.

Therefore:

$$
\boxed{\text{KV Cache} = \text{Historical K} + \text{Historical V}}
$$

The educational implementation stores compact GQA representation:

$$
\boxed{K, V : [B, H_{kv}, S, d_h]}
$$

Persistent KV is not expanded to $H$ heads ahead of time.

KV Cache is a compute-memory trade-off:

```text
Naive:
Persistent KV memory down
Repeated compute up

KV Cache:
Persistent KV memory up
Repeated historical compute down
```

That is:

$$
\boxed{\text{KV Cache} = \text{persist K/V to reuse historical computation}}
$$

This is the fundamental reason why serving engines must manage KV capacity.

---

## 6. Prefill vs Decode

For GEMM:

$$
[M, K] \times [K, N]
$$

FLOPs:

$$
2 M K N
$$

BF16 weight bytes are approximately:

$$
2 K N
$$

Ignoring non-weight traffic:

$$
AI_{weight} \approx \frac{2 M K N}{2 K N} = M \ \text{FLOP/B}
$$

### Prefill

$$
M = S
$$

Execution resembles:

$$
[S, d] \times [d, N]
$$

Characteristics:

- large GEMM
- high weight reuse
- Tensor Core easier to utilize
- Attention has an $S^2$ component

### Decode

For batch size 1:

$$
M = 1
$$

Execution approximates:

$$
[1, d] \times [d, N]
$$

Characteristics:

- skinny GEMM / GEMV
- continuous weight reads
- historical KV reads
- many tiny operators / kernels
- framework / launch overhead more visible

Benchmark method:

- Prefill / Decode: `torch.inference_mode()`, CUDA events, model forward only.
- Therefore fields are `PrefillFwd(ms)` and `DecodeFwd(ms)`, not TTFT / TPOT.
- Output Sweep: synchronized wall-clock, including Python loop, model forward, argmax, and cache orchestration.
- Early scripts had inconsistent execution contracts: Prompt/Decode Sweep had Autograd enabled; Output Sweep used `@torch.inference_mode()`. After the fix, both use `@torch.inference_mode()`.
- `model.eval()` mainly changes Dropout / BatchNorm behavior. It does not automatically disable Autograd.

---

## 7. Attention Benchmark

### 7.1 Naive Attention Context Sweep

| Context | Prefill Fwd | Decode Fwd | KV |
|---:|---:|---:|---:|
| 128 | 2.3873 ms | 2.3248 ms | 0.25 MiB |
| 512 | 2.4013 ms | 2.3349 ms | 1 MiB |
| 2048 | 3.4956 ms | 2.3301 ms | 4 MiB |
| 4096 | 13.4318 ms | 2.3214 ms | 8 MiB |

From 2048 to 4096:

$$
3.4956 \rightarrow 13.4318 \ \text{ms}
$$

Approximately:

$$
\boxed{3.84\times}
$$

Sequence length only doubled, which gives a strong hypothesis:

> At long context, naive attention's $S^2$ compute plus intermediate traffic begins to dominate.

### 7.2 Output Sweep

Prompt = 512:

| Output | E2E | Output Tok/s |
|---:|---:|---:|
| 1 | 2.4331 ms | 411.01 |
| 8 | 18.7431 ms | 426.82 |
| 32 | 74.1324 ms | 431.66 |

For 8 tokens:

$$
\frac{18.7431 - 2.4331}{7} \approx 2.330 \ \text{ms}
$$

For 32 tokens:

$$
\frac{74.1324 - 2.4331}{31} \approx 2.313 \ \text{ms}
$$

This matches the independent decode forward of about 2.33 ms.

Naive runtime follows:

$$
\boxed{T_{generation} \approx T_{prefill} + (T-1) T_{decode}}
$$

### 7.3 Decode Runtime Characteristics

Naive Decode:

```text
Context 512:
Self CUDA ~= 851.7 us / 3
~= 284 us/forward

Context 4096:
Self CUDA ~= 1.080 ms / 3
~= 360 us/forward
```

GPU work growth:

$$
\approx 1.27\times
$$

But normal benchmark remains about:

$$
2.33 \ \text{ms}
$$

This indicates that fixed/runtime overhead is a large fraction of the toy model runtime.

Decode launch fragmentation:

```text
cudaLaunchKernel calls = 483 / 3
```

Therefore:

$$
\boxed{161 \ \text{launches/forward}}
$$

Many linear layers become:

```text
[1,512] x [512,N]
```

Profiler shows:

```text
internal::gemvx...
```

i.e. GEMV-like execution.

Context growth does add cost:

| Operator | Context 512 | Context 4096 |
|---|---:|---:|
| `aten::cat` | 64.5 us | 115.2 us |
| GQA `repeat_interleave` underlying copy | 48.8 us | 126.8 us |
| QK BMM | ~24.2 us | ~58.1 us |
| PV BMM | ~31.6 us | ~73.2 us |

Context cost exists, but it does not break through the toy runtime's framework / launch floor.

### 7.4 SDPA Benchmark

| Context | SDPA Prefill | SDPA Decode | KV |
|---:|---:|---:|---:|
| 128 | 2.6278 ms | 2.6186 ms | 0.25 MiB |
| 512 | 2.6547 ms | 2.7070 ms | 1 MiB |
| 2048 | 2.5681 ms | 2.6849 ms | 4 MiB |
| 4096 | 2.5595 ms | 2.6819 ms | 8 MiB |

Prefill comparison:

| Context | Naive | SDPA | Naive / SDPA |
|---:|---:|---:|---:|
| 128 | 2.3873 | 2.6278 | 0.91x |
| 512 | 2.4013 | 2.6547 | 0.90x |
| 2048 | 3.4956 | 2.5681 | **1.36x** |
| 4096 | 13.4318 | 2.5595 | **5.25x** |

Therefore:

$$
\boxed{\text{Optimization benefit is workload-dependent}}
$$

- Small prompt: SDPA is slightly slower.
- Medium prompt: SDPA begins to win.
- Long prompt:

$$
\boxed{13.43 \ \text{ms} \rightarrow 2.56 \ \text{ms} = 5.25\times}
$$

### 7.5 Why SDPA Decode Has No E2E Speedup

Decode 4096:

$$
\text{Naive} = 2.3214 \ \text{ms}
$$

$$
\text{SDPA} = 2.6819 \ \text{ms}
$$

SDPA is about 15.5% slower end-to-end.

But GPU Self CUDA:

Naive:

$$
1.080 / 3 = 0.360 \ \text{ms}
$$

SDPA:

$$
0.865 / 3 = 0.288 \ \text{ms}
$$

GPU work decreases by about 20%.

Launch count:

```text
Naive: 483/3 = 161
SDPA : 423/3 = 141
```

Launch count also decreases.

Therefore:

$$
\boxed{\text{GPU Kernel Win} \not\Rightarrow \text{Runtime E2E Win}}
$$

In toy Decode, GPU work is only about 0.3 ms, while total model-forward is on the order of 2–3 ms. Host-side dispatch, backend setup, framework orchestration, and many tiny ops can dominate GPU savings.

This confirms:

> Optimize the bottleneck, not the most popular operator.

---

## 8. Profiler Evidence

Profiler answers WHY. Normal benchmarking answers HOW FAST.

### 8.1 Naive Prefill S=4096

Naive Prefill S=4096, 3 profiler iterations:

```text
Self CUDA total = 40.572 ms
```

Per forward:

$$
13.524 \ \text{ms/forward}
$$

Key operators:

| Operator | Time | Share |
|---|---:|---:|
| `aten::_softmax` | 8.129 ms | 20.04% |
| `aten::masked_fill_` | 7.628 ms | 18.80% |
| `aten::div` | 6.994 ms | 17.24% |
| `aten::copy_` | 6.985 ms | 17.22% |
| QK bmm | 3.400 ms | 8.38% |
| PV bmm | 4.095 ms | 10.09% |

Sum of six:

$$
37.231 \ \text{ms}
$$

Share:

$$
\frac{37.231}{40.572} \approx \boxed{91.8\%}
$$

Root cause:

$$
\boxed{\text{Naive Attention Pipeline dominates GPU execution}}
$$

### 8.2 S x S Intermediate Memory Evidence

S = 4096:

$$
[B, H, S, S] = [1, 8, 4096, 4096]
$$

BF16 score tensor:

$$
1 \times 8 \times 4096^2 \times 2\text{B} = 256 \ \text{MiB}
$$

4 layers:

$$
1 \ \text{GiB}
$$

Profiler 3 iterations:

$$
3 \ \text{GiB}
$$

Measured:

```text
aten::_softmax  CUDA Mem = 3.00 GB
aten::div       CUDA Mem = 3.00 GB
QK bmm          CUDA Mem = 3.00 GB
```

Note:

> The 3 GB figure is cumulative operator allocation across layers and iterations in the profiler. It is not a single-moment peak GPU memory measurement.

### 8.3 Naive Attention Core Problem

Naive pipeline:

```text
QK^T
 ↓
write S x S to HBM

Scale / div
 ↓
read + write

Mask
 ↓
read + write

Softmax
 ↓
read + write

P V
 ↓
read probabilities
```

Therefore:

$$
\boxed{\text{Long-context bottleneck} = S^2 \ \text{compute} + S^2 \ \text{intermediate multi-pass HBM IO}}
$$

This is the direct evidence for IO-aware / fused attention.

### 8.4 SDPA / FlashAttention Backend Evidence

SDPA Prefill S=4096:

```text
aten::scaled_dot_product_attention
↓
aten::_scaled_dot_product_flash_attention
↓
aten::_flash_attention_forward
↓
pytorch_flash::flash_fwd_kernel
```

The backend is verified by profiler evidence, not by assuming the SDPA API name implies FlashAttention.

SDPA Prefill S=4096:

```text
Self CUDA total = 6.053 ms / 3
~= 2.018 ms/forward
```

Naive:

```text
Self CUDA total = 40.572 ms / 3
~= 13.524 ms/forward
```

GPU Self CUDA reduction:

$$
\boxed{6.70\times}
$$

SDPA removes S x S intermediates:

```text
scaled_dot_product_attention ~= 48 MB
_flash_attention_forward    ~= 49.5 MB
```

Attention output per layer:

$$
1 \times 4096 \times 8 \times 64 \times 2\text{B} = 4 \ \text{MiB}
$$

4 layers x 3 iterations:

$$
48 \ \text{MiB}
$$

Execution moves from:

```text
full S x S intermediate
```

to:

```text
tile / online softmax / fused accumulation
```

avoiding repeated full score-matrix writes to HBM.

Kernel launch:

```text
Naive Prefill: 471 / 3 = 157
SDPA Prefill : 387 / 3 = 129
```

Reduction: 28 launches per forward.

### 8.5 Split-KV / FlashDecoding Clue

SDPA Decode 4096 profiler shows:

```text
flash_fwd_splitkv_kernel
flash_fwd_splitkv_combine_kernel
```

This gives the execution intuition for Split-KV / FlashDecoding:

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

M2 does not implement FlashDecoding, but the strategy is observed in the real backend.

---

## 9. Naive vs SDPA

Profiler first identified the real root cause, so SDPA was introduced after that:

```text
attention_backend:
- naive
- sdpa
```

### 9.1 Correctness

MHA:

```text
[H_kv=8] Prefill parity: PASSED
[H_kv=8] Cached Decode parity: PASSED
```

GQA:

```text
[H_kv=2] Prefill parity: PASSED
[H_kv=2] Cached Decode parity: PASSED
```

Final:

```text
All attention backend parity checks passed.
```

SDPA preserves semantics for both Prefill and Cached Decode.

Compact GQA in SDPA Prefill profiler:

```text
Q: [1, 4096, 8, 64]
K: [1, 4096, 2, 64]
V: [1, 4096, 2, 64]
```

The fused backend directly accepts compact GQA K/V, avoiding explicit materialization of expanded KV heads.

### 9.2 Before / After

Prefill comparison:

| Context | Naive | SDPA | Naive / SDPA |
|---:|---:|---:|---:|
| 128 | 2.3873 | 2.6278 | 0.91x |
| 512 | 2.4013 | 2.6547 | 0.90x |
| 2048 | 3.4956 | 2.5681 | **1.36x** |
| 4096 | 13.4318 | 2.5595 | **5.25x** |

S=4096 Prefill:

$$
13.43 \ \text{ms} \rightarrow 2.56 \ \text{ms}
$$

$$
\boxed{5.25\times \text{ wall-time reduction}}
$$

GPU Self CUDA:

Naive:

$$
\approx 13.524 \ \text{ms}
$$

SDPA:

$$
\approx 2.018 \ \text{ms}
$$

GPU work reduction:

$$
\boxed{6.70\times}
$$

The absolute values from profiler and benchmark should not be mixed, but the trend is consistent.

### 9.3 Decode Trade-off

Decode 4096:

| Metric | Naive | SDPA |
|---|---:|---:|
| E2E forward | 2.3214 ms | 2.6819 ms |
| GPU Self CUDA | 0.360 ms | 0.288 ms |
| Kernel launches | 161 | 141 |

SDPA reduces GPU work and launches, but E2E is about 15.5% slower.

Therefore:

$$
\boxed{\text{Kernel-level win} \not\Rightarrow \text{Runtime-level win}}
$$

---

## 10. Results

Key verified results:

| Item | Evidence |
|---|---|
| P=128, T=32 Naive token evaluations | 4592 |
| P=128, T=32 Cached token evaluations | 159 |
| Naive vs Cached correctness | `outputs_equal=True` |
| Prefill S=4096 Naive | 13.4318 ms |
| Prefill S=4096 SDPA | 2.5595 ms |
| Prefill S=4096 wall-time reduction | **5.25x** |
| Naive GPU self S=4096 Prefill | ~13.524 ms |
| SDPA GPU self S=4096 Prefill | ~2.018 ms |
| GPU self reduction | ~6.70x |
| Naive S x S intermediate allocation | ~3 GB cumulative operator allocation |
| SDPA attention allocation | ~48 MB / 49.5 MB |
| Naive Prefill launch | 157 / forward |
| SDPA Prefill launch | 129 / forward |
| Decode SDPA GPU work | ~20% lower |
| Decode SDPA E2E | ~15.5% slower |

Summary:

| Requirement | Evidence | Status |
|---|---|---|
| Transformer components | MiniDecoder | Pass |
| RMSNorm / Attention / MLP / LM Head | Implemented | Pass |
| MHA / GQA | Implementation + parity | Pass |
| Resource accounting | Parameter / KV / FLOPs | Pass |
| Naive generation | 38 / 4592 evaluations | Pass |
| KV Cache | 11 / 159 evaluations | Pass |
| Naive vs Cached correctness | `outputs_equal` | Pass |
| Prefill / Decode split | Execution trace | Pass |
| Prompt sweep | 128 / 512 / 2048 / 4096 | Pass |
| Output sweep | 1 / 8 / 32 | Pass |
| KV scaling | 0.25 / 1 / 4 / 8 MiB | Pass |
| PyTorch profiler | Prefill + Decode | Pass |
| Operator breakdown | Complete | Pass |
| Long-context root cause | S^2 attention IO | Pass |
| SDPA / Flash backend | Profiler verified | Pass |
| MHA/GQA SDPA correctness | Prefill + Decode | Pass |
| Before / After | 4096 Prefill 5.25x | Pass |
| Runtime trade-off | Decode kernel win != E2E win | Pass |

---

## 11. Limitations

- **Educational model:** MiniDecoder is not a real 7B/8B production model. Absolute latency should not be directly generalized.
- **Learned position embedding:** This stage does not fully implement Llama-style RoPE.
- **`torch.cat()` KV Cache:** Semantically correct, but not production KV memory management.
- **Naive GQA expansion:** `repeat_interleave()` is an educational implementation and introduces extra temporary memory / copies. SDPA has verified that compact GQA K/V can be passed directly to a fused backend.
- **No scheduler / queue / concurrency:** Current setup is single request, batch=1, single GPU. No Continuous Batching, Paged KV, Prefix Cache, online arrival, or scheduler.
- **Not full TTFT / TPOT:** This report measures model runtime, not the full server request path.
- **Long-run CPU/GPU overlap not yet verified with Nsight Systems:** Reserved for a later serving / system profiling stage.
- **Profiler allocation is not peak GPU memory:** The 3 GB figure is cumulative allocation across layers and iterations, not a single-moment peak.
- **GPU self time != E2E:** GPU Self CUDA time from profiler cannot be directly treated as end-to-end runtime latency.
- **FlashAttention removes materialized S x S intermediates, but does not change attention arithmetic asymptotic from S^2.** It mainly optimizes HBM IO. Exact attention remains roughly $O(S^2 d)$ compute, but uses tiling, online softmax, and fused output accumulation to avoid full score-matrix materialization.

---

## 12. Conclusions

1. **Autoregressive dependency is not removed by KV Cache.** KV Cache removes recomputation of historical K/V and historical hidden states.

2. **KV Cache is a compute-memory trade-off.** It persists K/V to avoid repeated computation, but consumes GPU memory scaling with $B \times S \times L \times H_{kv}$.

3. **GQA is a serving resource design.** Reducing $H_{kv}$ directly lowers KV footprint and KV bandwidth pressure.

4. **Prefill and Decode are different GPU workloads.** Prefill forms large GEMMs and long-sequence attention. Decode is closer to skinny GEMM/GEMV plus KV reads and many tiny kernels.

5. **Long-context naive attention is not only about FLOPs.** Multi-pass HBM materialization of full $S \times S$ intermediates is a key bottleneck.

6. **FlashAttention is IO-aware execution.** Exact attention remains roughly $O(S^2 d)$ compute, but tiling, online softmax, and fusion avoid writing the full score matrix back to HBM.

7. **Optimization benefit is workload-dependent.** Small shapes can make SDPA slower. S=4096 Prefill gains about 5.25x.

8. **Kernel-level win does not equal runtime-level win.** Decode SDPA has lower GPU-side work, but toy E2E forward is slower.

9. **Performance engineering must be cross-layer.** Model, operator, kernel, and runtime bottlenecks can be completely different.