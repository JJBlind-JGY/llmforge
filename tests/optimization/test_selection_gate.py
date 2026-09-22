from llmforge.optimization import (
    CandidateState,
    OptimizationArea,
    OptimizationCandidate,
    SelectionEvidence,
    evaluate_selection_gate,
)


def make_candidate(
    evidence: SelectionEvidence,
) -> OptimizationCandidate:
    return OptimizationCandidate(
        candidate_id="candidate",
        title="Candidate",
        area=(OptimizationArea.PREFILL_DECODE),
        problem="problem",
        mechanism="mechanism",
        changed_component="scheduler",
        expected_benefit="tail latency",
        expected_tradeoffs=("throughput",),
        state=(CandidateState.EVIDENCE_PENDING),
        evidence=evidence,
    )


def test_gate_rejects_incomplete_candidate() -> None:
    result = evaluate_selection_gate(make_candidate(SelectionEvidence()))

    assert not result.eligible
    assert len(result.checks) == 8


def test_gate_accepts_complete_evidence() -> None:
    evidence = SelectionEvidence(
        reproduction_artifacts=(
            "run1.json",
            "run2.json",
            "run3.json",
        ),
        profiler_artifacts=("trace.json",),
        mechanism_note=(
            "Long prefill consumes the "
            "shared step token budget while "
            "decode requests are active."
        ),
        baseline_artifact=("baseline.json"),
        correctness_plan=("Greedy response hashes must match."),
        regression_metrics=(
            "output_throughput_tok_s",
            "tpot_p99_ms",
        ),
        hardware_feasible=True,
        community_evidence=("upstream-source-audit.md",),
        upstream_overlap=("Static controls exist; candidate is dynamic."),
        parameter_only=False,
    )

    result = evaluate_selection_gate(make_candidate(evidence))

    assert result.eligible
