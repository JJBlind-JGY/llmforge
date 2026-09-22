"""vLLM 0.29 reference candidate for mixed prefill/decode interference.

This module is imported only in the pinned vLLM environment.

It subclasses the M4 TracingScheduler so candidate runs retain the same runtime
instrumentation as the baseline. The candidate changes only the effective
`max_num_scheduled_tokens` during one `schedule()` call, then restores the
upstream value.

This is an experimental implementation, not a performance claim.
"""

from __future__ import annotations

import atexit
import os
import time
from pathlib import Path
from typing import Any

from llmforge.runtime.integrations.vllm.tracing_scheduler import (
    TracingScheduler,
)

from llmforge.optimization.decision_trace import (
    BufferedDecisionRecorder,
)

from .adaptive_budget import (
    AdaptiveBudgetConfig,
    SchedulerPressure,
    choose_budget,
    count_request_phases,
)


class AdaptiveBudgetScheduler(
    TracingScheduler
):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        if int(
            getattr(
                self,
                "num_spec_tokens",
                0,
            )
            or 0
        ) != 0:
            raise RuntimeError(
                "AdaptiveBudgetScheduler "
                "M7 reference implementation "
                "does not support speculative "
                "decoding."
            )

        self._m7_config = (
            AdaptiveBudgetConfig
            .from_environment()
        )

        self._m7_base_budget = int(
            self.max_num_scheduled_tokens
        )

        raw_path = os.environ.get(
            "LLMFORGE_M7_DECISION_TRACE",
            (
                "artifacts/optimization/"
                "adaptive_budget_"
                "{pid}.jsonl"
            ),
        )

        self._m7_decisions = (
            BufferedDecisionRecorder(
                Path(
                    raw_path.format(
                        pid=os.getpid()
                    )
                )
            )
        )

        atexit.register(
            self._m7_close
        )

    def _m7_close(
        self,
    ) -> None:
        recorder = getattr(
            self,
            "_m7_decisions",
            None,
        )

        if recorder is not None:
            recorder.close()

    def _m7_pressure(
        self,
    ) -> SchedulerPressure:
        running_prefill, running_decode = (
            count_request_phases(
                self.running
            )
        )

        waiting_requests = [
            *list(
                self.waiting
            ),
            *list(
                self.skipped_waiting
            ),
        ]

        waiting_prefill, waiting_decode = (
            count_request_phases(
                waiting_requests
            )
        )

        return SchedulerPressure(
            running_prefill=(
                running_prefill
            ),
            running_decode=(
                running_decode
            ),
            waiting_prefill=(
                waiting_prefill
            ),
            waiting_decode=(
                waiting_decode
            ),
            kv_usage_ratio=float(
                self.get_kv_cache_usage()
            ),
        )

    def schedule(
        self,
        throttle_prefills: bool = False,
    ):
        pressure = (
            self._m7_pressure()
        )

        decision = choose_budget(
            config=self._m7_config,
            base_budget=(
                self._m7_base_budget
            ),
            pressure=pressure,
        )

        next_step_id = int(
            getattr(
                self,
                "_llmforge_step_id",
                0,
            )
        ) + 1

        self._m7_decisions.emit(
            {
                "schema_version": 1,
                "wall_time_ns": (
                    time.time_ns()
                ),
                "monotonic_ns": (
                    time.perf_counter_ns()
                ),
                "step_id": (
                    next_step_id
                ),
                "candidate_id": (
                    "adaptive_mixed_batch_budget"
                ),
                **decision.to_dict(),
            }
        )

        original_budget = (
            self.max_num_scheduled_tokens
        )

        self.max_num_scheduled_tokens = (
            decision.applied_budget
        )

        try:
            return super().schedule(
                throttle_prefills=(
                    throttle_prefills
                )
            )
        finally:
            self.max_num_scheduled_tokens = (
                original_budget
            )
