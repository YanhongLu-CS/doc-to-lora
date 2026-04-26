from __future__ import annotations

from contextlib import nullcontext

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def is_npu_available() -> bool:
    return bool(
        hasattr(torch, "npu")
        and getattr(torch.npu, "is_available", None)
        and torch.npu.is_available()
    )


def get_default_device() -> torch.device:
    if is_npu_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return get_default_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def get_device_type(device: str | torch.device | None = None) -> str:
    return normalize_device(device).type


def get_default_dtype(device: str | torch.device | None = None) -> torch.dtype:
    device_type = get_device_type(device)
    if device_type in {"cuda", "npu"}:
        return torch.bfloat16
    return torch.float32


def should_use_flash_attn(
    requested: bool, device: str | torch.device | None = None
) -> bool:
    return requested and get_device_type(device) == "cuda"


def get_autocast_context(
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    device_type = get_device_type(device)
    if device_type in {"cuda", "npu"}:
        return torch.autocast(device_type=device_type, dtype=dtype)
    return nullcontext()
