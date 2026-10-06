# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""PyTorch memory utilities for garbage collection and XPU cache cleanup."""

import gc
from ctypes import CDLL, util

import torch


def cleanup_memory() -> None:
    """Perform garbage collection and XPU cache cleanup."""
    cpu_cleanup()
    gpu_cleanup()


def gpu_cleanup():
    """Perform XPU cache cleanup."""
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        if hasattr(xpu, "empty_cache"):
            xpu.empty_cache()
        if hasattr(xpu, "synchronize"):
            xpu.synchronize()


def cpu_cleanup():
    """Perform CPU memory cleanup."""
    gc.collect()
    cpu_cleanup_low_level()


def cpu_cleanup_low_level():
    """Perform low-level CPU memory cleanup."""
    import platform

    try:
        if platform.system() == "Linux":
            CDLL(util.find_library("c")).malloc_trim(0)
    except (OSError, AttributeError):
        # Silently ignore if platform-specific cleanup is not available
        pass
