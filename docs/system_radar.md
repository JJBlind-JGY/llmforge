# M8 System Radar

M8 does **not** attempt to deploy every modern inference system.

The syllabus requires technical understanding of:

```text
vLLM
SGLang
Mooncake
NVIDIA Dynamo
llm-d
```

while keeping vLLM as the primary backend and using SGLang as the second engine
for cross-engine validation.

## vLLM

Role in LLMForge:

```text
M3 serving baseline
M4 runtime source/instrumentation
M5 distributed inference
M7 optimization target
```

M8 does not replace this primary line.

## SGLang

Role:

```text
second serving backend
+
cross-engine mechanism validation
```

The first M8 experiment uses SGLang's OpenAI-compatible API so that the **same
LLMForge M3 client** can measure both engines.

SGLang's serving benchmark documentation exposes TTFT, ITL, E2E, throughput,
rate control, concurrency control, shared-prefix workloads, and OpenAI-compatible
chat/completion backends. That makes it a useful independent cross-check, but
LLMForge still keeps its own M3 client as the primary measurement boundary.

SGLang's PD-disaggregation documentation explicitly separates compute-intensive
Prefill from memory-intensive Decode and documents Mooncake and NIXL transfer
backends. This is a useful architecture reference even when the local 4×4090
M8 validation does not deploy disaggregation.

## Mooncake

Mooncake is studied as a KV-cache transfer/storage system rather than added to
the default LLMForge runtime.

The current Mooncake documentation contains integrations for both SGLang and
vLLM and separates Transfer Engine, Mooncake Store, and related distributed-KV
components.

M8 records the architectural lesson:

```text
KV state is becoming a distributed systems resource,
not only a tensor allocated inside one inference worker.
```

## NVIDIA Dynamo

Dynamo is studied as a system around the inference engine.

Its current architecture is backend-agnostic and can work with SGLang, vLLM,
and TensorRT-LLM. The architecture separates request, control, and state concerns
and includes disaggregated serving and KV-aware routing.

This gives LLMForge an important system boundary:

```text
Inference Engine
    !=
Distributed Inference Platform
```

LLMForge M3-M7 mostly studies the former and local multi-GPU runtime behavior;
Dynamo demonstrates how routing, control, and KV state are composed around it.

## llm-d

llm-d is studied as a Kubernetes-native distributed-inference architecture.

Its current architecture centers on:

```text
Router
  ├── Proxy
  └── Endpoint Picker (EPP)

InferencePool

Model Server
  └── vLLM / SGLang / other engine
```

Advanced patterns include:

```text
prefix-cache-aware routing
KV-cache indexing
KV offloading
disaggregated serving
predicted-latency routing
autoscaling
```

This is intentionally a radar item rather than a required local deployment,
because the original LLMForge scope prioritizes real single-node GPU/runtime
evidence over a fake Kubernetes cluster.

## What M8 must learn from the radar

The systems form different layers:

```text
CUDA / Triton
     ↓
Model Runtime
     ↓
Serving Engine
(vLLM / SGLang)
     ↓
KV Transfer / State
(Mooncake / NIXL class)
     ↓
Distributed Serving Platform
(Dynamo / llm-d class)
     ↓
Cluster / Deployment Control
```

The project should be able to explain these layers without claiming that every
layer was locally deployed.

## Source policy

Before the final public release, re-check stable official documentation because
these projects change rapidly.

Do not copy API/version assumptions from old blog posts into the release docs.
