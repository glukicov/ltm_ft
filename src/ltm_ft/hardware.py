"""Honest timing and memory on whichever accelerator the run lands on (Apple MPS, NVIDIA CUDA or CPU).

GPU kernels run asynchronously, so a wall-clock reading is only meaningful after `synchronize`.
Peak memory is exact on CUDA (`max_memory_allocated`); MPS only reports what its driver currently
holds, so callers sample `memory_gb` after every step and keep the maximum.
"""

from __future__ import annotations

import contextlib
import platform
import subprocess

import torch


def device_type(device: str) -> str:
    return torch.device(device).type


def synchronize(device: str) -> None:
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def accelerator_name(device: str) -> str:
    """e.g. "NVIDIA L4", "Apple M4 (MPS)" or the CPU model."""
    kind = device_type(device)
    if kind == "cuda":
        return torch.cuda.get_device_name(torch.device(device))
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
    return f"{cpu} (MPS)" if kind == "mps" else cpu


def reset_peak_memory(device: str) -> None:
    if device_type(device) == "cuda":
        torch.cuda.reset_peak_memory_stats()


def memory_gb(device: str) -> float | None:
    """Peak allocated memory on CUDA; the MPS driver's current allocation; None on CPU."""
    kind = device_type(device)
    if kind == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if kind == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return None
