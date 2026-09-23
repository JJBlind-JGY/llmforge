"""Check naive vs SDPA attention parity on GPU (MHA + GQA)."""

from __future__ import annotations

import torch

from llmforge.model_execution.mini_decoder import (
    MiniDecoderConfig,
    MiniDecoderLM,
)


def _make_pair_configs(
    num_kv_heads: int,
) -> tuple[MiniDecoderConfig, MiniDecoderConfig]:
    common = {
        "vocab_size": 256,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": num_kv_heads,
        "max_sequence_length": 512,
    }
    return (
        MiniDecoderConfig(**common, attention_backend="naive"),
        MiniDecoderConfig(**common, attention_backend="sdpa"),
    )


def _check(num_kv_heads: int, device: torch.device) -> None:
    naive_cfg, sdpa_cfg = _make_pair_configs(num_kv_heads)
    naive_model = MiniDecoderLM(naive_cfg).to(device).float().eval()
    sdpa_model = MiniDecoderLM(sdpa_cfg).to(device).float().eval()
    sdpa_model.load_state_dict(naive_model.state_dict())

    # ---- Prefill parity ----
    prompt = torch.randint(0, naive_cfg.vocab_size, (1, 127), device=device)
    with torch.no_grad():
        naive_logits = naive_model(prompt)
        sdpa_logits = sdpa_model(prompt)

    torch.testing.assert_close(
        sdpa_logits,
        naive_logits,
        rtol=1e-4,
        atol=1e-4,
    )
    print(f"  [H_kv={num_kv_heads}] Prefill parity: PASSED")

    # ---- Cached Decode parity ----
    with torch.no_grad():
        _, cache = naive_model.forward_with_cache(prompt)
        next_token = torch.argmax(naive_logits[:, -1, :], dim=-1, keepdim=True)
        naive_step = naive_model.forward_with_cache(
            next_token,
            past_key_values=cache,
        )[0]

        _, sdpa_cache = sdpa_model.forward_with_cache(prompt)
        sdpa_step = sdpa_model.forward_with_cache(
            next_token,
            past_key_values=sdpa_cache,
        )[0]

    torch.testing.assert_close(
        sdpa_step[:, -1, :],
        naive_step[:, -1, :],
        rtol=1e-4,
        atol=1e-4,
    )
    print(f"  [H_kv={num_kv_heads}] Cached Decode parity: PASSED")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print("Checking MHA (H_kv = H):")
    _check(num_kv_heads=8, device=device)

    print("Checking GQA (H_kv = 2, H = 8):")
    _check(num_kv_heads=2, device=device)

    print("\nAll attention backend parity checks passed.")


if __name__ == "__main__":
    main()
