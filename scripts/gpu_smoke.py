"""Minimal GPU runtime smoke test for LLMForge."""

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")

    x = torch.randn(2048, 2048, device=device)
    y = torch.randn(2048, 2048, device=device)

    z = x @ y

    torch.cuda.synchronize()

    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={tuple(z.shape)}")
    print(f"dtype={z.dtype}")
    print(f"mean={z.mean().item():.6f}")


if __name__ == "__main__":
    main()
