"""Inspect a real Hugging Face causal LM on one GPU."""

from __future__ import annotations

import argparse
import time

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def bytes_to_gib(value: int) -> float:
    return value / 2**30


def bytes_to_mib(value: int) -> float:
    return value / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
    )

    parser.add_argument(
        "--revision",
        required=True,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU.")

    device = torch.device("cuda:0")

    print("=== Environment ===")
    print(f"transformers={transformers.__version__}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    print()
    print("=== Config ===")

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
    )

    head_dim = getattr(
        config,
        "head_dim",
        (config.hidden_size // config.num_attention_heads),
    )

    print(f"architecture={config.architectures}")
    print(f"hidden_size={config.hidden_size}")
    print(f"intermediate_size={config.intermediate_size}")
    print(f"num_layers={config.num_hidden_layers}")
    print(f"attention_heads={config.num_attention_heads}")
    print(f"kv_heads={config.num_key_value_heads}")
    print(f"head_dim={head_dim}")
    print(f"vocab_size={config.vocab_size}")
    print(f"max_position_embeddings={config.max_position_embeddings}")

    kv_bytes_per_token = (
        config.num_hidden_layers * 2 * config.num_key_value_heads * head_dim * 2
    )

    print()
    print("=== Predicted BF16 KV Cache ===")
    print(f"kv_per_token_kib={kv_bytes_per_token / 1024:.2f}")

    for sequence_length in (
        128,
        512,
        2048,
        4096,
    ):
        kv_bytes = kv_bytes_per_token * sequence_length

        print(f"kv_s{sequence_length}_mib={bytes_to_mib(kv_bytes):.2f}")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    free_before, total_memory = torch.cuda.mem_get_info()

    print()
    print("=== Loading Model ===")
    print(f"gpu_total_gib={bytes_to_gib(total_memory):.3f}")
    print(f"gpu_free_before_gib={bytes_to_gib(free_before):.3f}")

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    model.eval()

    torch.cuda.synchronize()

    load_seconds = time.perf_counter() - start

    free_after, _ = torch.cuda.mem_get_info()

    actual_parameters = sum(parameter.numel() for parameter in model.parameters())

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    print(f"load_seconds={load_seconds:.3f}")
    print(f"parameter_count_b={actual_parameters / 1e9:.3f}")
    print(f"parameter_storage_gib={bytes_to_gib(parameter_bytes):.3f}")
    print(
        f"model_memory_footprint_gib={bytes_to_gib(model.get_memory_footprint()):.3f}"
    )
    print(f"cuda_allocated_gib={bytes_to_gib(torch.cuda.memory_allocated()):.3f}")
    print(f"cuda_reserved_gib={bytes_to_gib(torch.cuda.memory_reserved()):.3f}")
    print(f"gpu_free_after_load_gib={bytes_to_gib(free_after):.3f}")

    attention_impl = getattr(
        model.config,
        "_attn_implementation",
        "unknown",
    )

    print(f"attention_implementation={attention_impl}")

    print()
    print("=== Smoke Generation ===")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "In one short sentence, "
                "describe what KV cache does "
                "during LLM inference."
            ),
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            min_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    torch.cuda.synchronize()

    generation_seconds = time.perf_counter() - start

    input_tokens = inputs["input_ids"].shape[1]

    output_tokens = output_ids.shape[1] - input_tokens

    peak_allocated = torch.cuda.max_memory_allocated()

    generated_text = tokenizer.decode(
        output_ids[0, input_tokens:],
        skip_special_tokens=True,
    )

    print(f"input_tokens={input_tokens}")
    print(f"output_tokens={output_tokens}")
    print(f"generation_seconds={generation_seconds:.4f}")
    print(f"peak_allocated_gib={bytes_to_gib(peak_allocated):.3f}")
    print(f"generated_text={generated_text!r}")


if __name__ == "__main__":
    main()
