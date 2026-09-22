# M7 Engineering Notes

This pack follows the original LLMForge M7 syllabus rather than prematurely
declaring one optimization "the answer".

Key decisions:

1. The eight-condition selection gate is executable code.
2. M3/M4/M5/M6 artifacts feed one Observation Sheet.
3. Existing vLLM knobs are explicitly classified as controls where appropriate.
4. The headline candidate cannot be a one-parameter tweak.
5. Baseline/candidate configs are fingerprinted before experiments.
6. Correctness is a hard gate.
7. Repeated runs are required.
8. Objective and guardrails are frozen before candidate results are analyzed.
9. The analysis always reports trade-offs, including regressions.
10. Negative results are preserved.
11. The adaptive mixed-batch budget is a reference implementation, not a
    pre-selected result.
12. It subclasses the M4 TracingScheduler so M7 retains runtime evidence.
13. The candidate emits a separate bounded policy-decision trace.
14. Static token-budget sweeps are mandatory controls for an adaptive-budget
    claim.
15. vLLM 0.29 already contains KV watermark/queue-cap/static scheduler controls;
    those are not presented as original contributions.
16. If evidence points elsewhere, the project must select scheduler fairness,
    CPU/runtime overhead, GPU kernels, or multi-GPU communication instead.
