"""Non-invasive vLLM scheduler instrumentation.

Use with:

--scheduler-cls llmforge.runtime.integrations.vllm.tracing_scheduler.TracingScheduler
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

from llmforge.runtime.events import (
    RuntimeEvent,
    RuntimeEventKind,
    clock_now,
)
from llmforge.runtime.observations import SchedulerStepObservation
from llmforge.runtime.trace import (
    BufferedJsonlTraceRecorder,
    SyncJsonlTraceRecorder,
)
from llmforge.runtime.trace.base import TraceRecorder

from .config import VLLMTraceConfig
from .helpers import extract_request_schedule_observations


class _NullRecorder:
    def emit(self, event: RuntimeEvent) -> bool:
        return True

    def flush(self) -> None:
        return

    def close(self) -> None:
        return

    def stats(self):
        return None


def _make_recorder(config: VLLMTraceConfig) -> TraceRecorder:
    if config.mode == "off":
        return _NullRecorder()

    if config.mode == "sync":
        return SyncJsonlTraceRecorder(config.path)

    return BufferedJsonlTraceRecorder(
        config.path,
        queue_size=config.queue_size,
        flush_every=config.flush_every,
        drop_on_full=config.drop_on_full,
    )


class TracingScheduler(Scheduler):
    """Observe upstream scheduling without changing its decisions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._llmforge_trace_config = VLLMTraceConfig.from_environment()
        self._llmforge_recorder = _make_recorder(self._llmforge_trace_config)
        self._llmforge_step_id = 0
        self._llmforge_finished_emitted: set[str] = set()

        atexit.register(self._llmforge_close_trace)

    def _llmforge_emit(
        self,
        kind: RuntimeEventKind,
        *,
        step_id: int | None = None,
        request_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        stamp = clock_now()

        self._llmforge_recorder.emit(
            RuntimeEvent(
                schema_version=1,
                kind=kind,
                wall_time_ns=stamp.wall_time_ns,
                monotonic_ns=stamp.monotonic_ns,
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
                step_id=step_id,
                request_id=request_id,
                payload=payload or {},
            )
        )

    def _llmforge_close_trace(self) -> None:
        recorder = getattr(self, "_llmforge_recorder", None)
        if recorder is None:
            return

        recorder.flush()
        stats = recorder.stats()

        if stats is not None and stats.dropped_events > 0:
            self._llmforge_emit(
                RuntimeEventKind.TRACE_DROPPED,
                step_id=self._llmforge_step_id,
                payload={
                    "submitted_events": stats.submitted_events,
                    "written_events": stats.written_events,
                    "dropped_events": stats.dropped_events,
                },
            )
            recorder.flush()

        recorder.close()

    def add_request(self, request) -> None:
        self._llmforge_emit(
            RuntimeEventKind.REQUEST_ENQUEUED,
            request_id=request.request_id,
            payload={
                "num_prompt_tokens": getattr(
                    request,
                    "num_prompt_tokens",
                    None,
                ),
                "max_tokens": getattr(
                    request,
                    "max_tokens",
                    None,
                ),
                "priority": getattr(
                    request,
                    "priority",
                    None,
                ),
            },
        )

        super().add_request(request)

    def schedule(self, throttle_prefills: bool = False):
        self._llmforge_step_id += 1
        step_id = self._llmforge_step_id

        start_ns = time.perf_counter_ns()

        scheduler_output = super().schedule(throttle_prefills=throttle_prefills)

        end_ns = time.perf_counter_ns()

        request_observations = extract_request_schedule_observations(
            step_id=step_id,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            requests=self.requests,
        )

        total_prompt_tokens = sum(
            observation.scheduled_prompt_tokens for observation in request_observations
        )

        total_output_tokens = sum(
            observation.scheduled_output_tokens for observation in request_observations
        )

        for observation in request_observations:
            self._llmforge_emit(
                RuntimeEventKind.REQUEST_SCHEDULED,
                step_id=step_id,
                request_id=observation.request_id,
                payload=observation.to_dict(),
            )

        preempted_req_ids = tuple(
            getattr(
                scheduler_output,
                "preempted_req_ids",
                (),
            )
            or ()
        )

        for request_id in preempted_req_ids:
            self._llmforge_emit(
                RuntimeEventKind.REQUEST_PREEMPTED,
                step_id=step_id,
                request_id=request_id,
            )

        running, waiting = self.get_request_counts()

        observation = SchedulerStepObservation(
            step_id=step_id,
            running_requests=running,
            waiting_requests=waiting,
            scheduled_requests=len(request_observations),
            scheduled_tokens=int(
                getattr(
                    scheduler_output,
                    "total_num_scheduled_tokens",
                    sum(scheduler_output.num_scheduled_tokens.values()),
                )
            ),
            scheduled_prompt_tokens=total_prompt_tokens,
            scheduled_output_tokens=total_output_tokens,
            preempted_requests=len(preempted_req_ids),
            kv_usage_ratio=float(self.get_kv_cache_usage()),
            schedule_duration_us=(end_ns - start_ns) / 1000.0,
        )

        self._llmforge_emit(
            RuntimeEventKind.SCHEDULER_STEP,
            step_id=step_id,
            payload=observation.to_dict(),
        )

        return scheduler_output

    def update_from_output(self, scheduler_output, model_runner_output):
        request_ids_before = set(self.requests)
        start_ns = time.perf_counter_ns()

        result = super().update_from_output(
            scheduler_output,
            model_runner_output,
        )

        end_ns = time.perf_counter_ns()

        self._llmforge_emit(
            RuntimeEventKind.MODEL_OUTPUT_PROCESSED,
            step_id=self._llmforge_step_id,
            payload={
                "duration_us": (end_ns - start_ns) / 1000.0,
                "scheduled_tokens": int(
                    getattr(
                        scheduler_output,
                        "total_num_scheduled_tokens",
                        0,
                    )
                ),
            },
        )

        request_ids_after = set(self.requests)

        for request_id in sorted(request_ids_before - request_ids_after):
            self._llmforge_emit_finished(
                request_id,
                reason="completed_or_released",
            )

        return result

    def finish_requests(self, request_ids, finished_status):
        finished = super().finish_requests(
            request_ids,
            finished_status,
        )

        for request in finished:
            self._llmforge_emit_finished(
                request.request_id,
                reason=str(finished_status),
            )

        return finished

    def _llmforge_emit_finished(self, request_id: str, *, reason: str) -> None:
        if request_id in self._llmforge_finished_emitted:
            return

        self._llmforge_finished_emitted.add(request_id)

        self._llmforge_emit(
            RuntimeEventKind.REQUEST_FINISHED,
            step_id=self._llmforge_step_id,
            request_id=request_id,
            payload={"reason": reason},
        )
