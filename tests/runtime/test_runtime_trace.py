import threading
from pathlib import Path

from llmforge.runtime.events import RuntimeEvent, RuntimeEventKind, clock_now
from llmforge.runtime.trace import (
    BufferedJsonlTraceRecorder,
    JsonlTraceReader,
    SyncJsonlTraceRecorder,
)


def make_event(index: int) -> RuntimeEvent:
    stamp = clock_now()
    return RuntimeEvent(
        schema_version=1,
        kind=RuntimeEventKind.SCHEDULER_STEP,
        wall_time_ns=stamp.wall_time_ns,
        monotonic_ns=stamp.monotonic_ns,
        process_id=1,
        thread_id=threading.get_ident(),
        step_id=index,
        payload={"index": index},
    )


def test_sync_trace_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sync.jsonl"
    with SyncJsonlTraceRecorder(path) as recorder:
        recorder.emit(make_event(1))
        recorder.emit(make_event(2))
    assert [e.step_id for e in JsonlTraceReader(path).read_all()] == [1, 2]


def test_buffered_trace_preserves_events(tmp_path: Path) -> None:
    path = tmp_path / "buffered.jsonl"
    with BufferedJsonlTraceRecorder(
        path,
        queue_size=128,
        flush_every=8,
        drop_on_full=False,
    ) as recorder:
        for index in range(32):
            assert recorder.emit(make_event(index))
        recorder.flush()
        stats = recorder.stats()
        assert stats.submitted_events == 32
        assert stats.written_events == 32
        assert stats.dropped_events == 0

    events = JsonlTraceReader(path).read_all()
    assert len(events) == 32
    assert events[0].step_id == 0
    assert events[-1].step_id == 31
