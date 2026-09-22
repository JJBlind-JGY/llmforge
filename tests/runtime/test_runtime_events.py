import threading

from llmforge.runtime.events import RuntimeEvent, RuntimeEventKind, clock_now


def test_runtime_event_round_trip() -> None:
    stamp = clock_now()
    event = RuntimeEvent(
        schema_version=1,
        kind=RuntimeEventKind.REQUEST_ENQUEUED,
        wall_time_ns=stamp.wall_time_ns,
        monotonic_ns=stamp.monotonic_ns,
        process_id=123,
        thread_id=threading.get_ident(),
        request_id="r1",
        payload={"priority": 0},
    )
    assert RuntimeEvent.from_dict(event.to_dict()) == event


def test_clock_contains_two_time_domains() -> None:
    stamp = clock_now()
    assert stamp.wall_time_ns > 0
    assert stamp.monotonic_ns > 0
