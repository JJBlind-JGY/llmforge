# M8 Cross-Engine Validation

## Research question

M8 asks:

> Is an observed serving/runtime behavior caused mainly by one vLLM
> implementation choice, or does the same qualitative behavior appear in a
> second modern serving engine?

The second engine is SGLang.

## Fair comparison boundary

For the first comparison:

```text
same model
same model revision where engine support allows
same GPU
same dtype
same prompt payloads
same output limit
same arrival schedule
same concurrency
same warmup/repetition policy
same LLMForge M3 client
same OpenAI-compatible endpoint semantics
```

The project must record unavoidable differences rather than pretending the two
engines have identical internal defaults.

## Cache-neutral baseline

Engine-specific prefix reuse can dominate a shared-prefix test.

Therefore the first cross-engine baseline uses:

```text
vLLM prefix cache disabled
SGLang radix cache disabled
```

Then a second experiment may intentionally enable each engine's native prefix
reuse to study the mechanism.

Do not mix those two questions.

## Measurement boundary

Primary cross-engine metrics come from the LLMForge M3 client:

```text
TTFT
ITL
TPOT
E2E
request throughput
output token throughput
```

SGLang's own `bench_serving` can be used as a cross-check, not as a replacement
for the measurement boundary.

## Repetitions

At least three runs per engine are required. Five are preferred.

Use raw JSON results:

```text
vLLM × N
SGLang × N
       ↓
analyze_cross_engine.py
       ↓
median comparison
```

## Correctness

Before interpreting performance, send the same deterministic correctness cases
to both engines.

The default cross-engine correctness contract checks:

```text
request success
prompt token count
completion token count
finish reason
```

Exact response hashes are optional because cross-engine floating-point/kernel
differences can create small greedy-decoding differences.

If exact text differs, inspect it before relaxing the criterion.


Run both correctness probes with:

```bash
python scripts/engines/run_engine_correctness_probe.py \
  --engine vllm \
  --base-url http://127.0.0.1:8000 \
  --output artifacts/engines/vllm_correctness.jsonl
```

```bash
python scripts/engines/run_engine_correctness_probe.py \
  --engine sglang \
  --base-url http://127.0.0.1:30000 \
  --output artifacts/engines/sglang_correctness.jsonl
```

Then compare them with `compare_engine_correctness.py`.

## Generality test

Cross-engine validation is not:

```text
SGLang is faster/slower than vLLM
```

The more important question is qualitative behavior.

Example:

```text
Baseline workload
vs
Long-prefill mixed workload
```

If:

```text
vLLM TPOT P99 degrades
and
SGLang TPOT P99 also degrades
```

then the symptom is more consistent with a general serving phenomenon.

If only vLLM shows it, the result may be implementation-specific or the engine
configurations may not yet be comparable.

M8 code records:

```text
same_direction = true / false
```

It does not turn that flag into a causal proof.

## M7 optimization and M8

If the final M7 optimization is tightly coupled to vLLM internals, SGLang does
not need an equivalent implementation.

Instead M8 can validate the **underlying bottleneck/mechanism** across engines.

Example:

```text
M7:
adaptive vLLM mixed-batch budget

M8:
Does long-prefill/decode interference also appear in SGLang?
```

This is stronger and more realistic than forcing the same source patch into two
unrelated runtime implementations.
