# M8 Engineering Notes

M8 closes the original LLMForge syllabus.

Design decisions:

1. vLLM remains the primary engine.
2. SGLang is the only mandatory second engine.
3. Both engines are measured through the same LLMForge M3 client.
4. Engine-native benchmarks are secondary cross-checks, not mixed into the
   primary measurement boundary.
5. The first comparison is cache-neutral.
6. Cross-engine correctness is checked before performance interpretation.
7. Repeated results are aggregated by median.
8. Cross-engine validation focuses on whether a mechanism/symptom generalizes,
   not merely on ranking engines.
9. Mooncake, Dynamo, and llm-d are architecture radar items rather than fake
   local deployments.
10. Public release readiness is executable through a strict repository audit.
11. The audit intentionally blocks release while the M7 final optimization
    report still contains its pending marker.
12. The audit also blocks release until a real upstream issue/patch/PR attempt
    is recorded.
13. README claims must be downstream of raw machine-readable evidence.
14. Existing stronger repository documents/reports should be merged, not
    overwritten mechanically by this pack.
