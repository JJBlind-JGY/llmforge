from llmforge.runtime.integrations.vllm.discovery import resolve_first_symbol


def test_symbol_discovery_uses_fallbacks() -> None:
    result = resolve_first_symbol(
        (
            ("this.module.does.not.exist", "Missing"),
            ("json", "JSONDecoder"),
        )
    )
    assert result is not None
    assert result.module_name == "json"
    assert result.symbol_name == "JSONDecoder"


def test_symbol_discovery_can_fail_cleanly() -> None:
    assert resolve_first_symbol((("this.module.does.not.exist", "Missing"),)) is None
