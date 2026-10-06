# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""System monitoring utilities for GPU and CPU usage."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
import torch

from torch_tweak.torch.config import SYSTEM_MONITOR_ENABLE

logger = logging.getLogger(__name__)

DEFAULT_LOGGER_FUNC = logger.info if SYSTEM_MONITOR_ENABLE else logger.debug


@dataclass
class GPUStats:
    """Data class containing GPU statistics.

    Only metrics that PyTorch XPU can actually report are exposed. Utilization and
    power draw would require a sysman backend (xpu-smi or Level Zero) which this
    module does not use.

    Allocated and reserved memory come from the PyTorch caching allocator and cover
    the current process only. Total and free memory are device wide and include
    every process using the GPU; they are None on builds without
    torch.xpu.mem_get_info.
    """

    device_index: int
    device_name: str
    memory_allocated_bytes: int
    memory_reserved_bytes: int
    memory_free_bytes: int | None
    memory_total_bytes: int | None

    @property
    def memory_allocated_mb(self) -> float:
        """Memory allocated by the PyTorch caching allocator in MB."""
        return self.memory_allocated_bytes / 1024 / 1024

    @property
    def memory_reserved_mb(self) -> float:
        """Memory reserved by the PyTorch caching allocator in MB."""
        return self.memory_reserved_bytes / 1024 / 1024

    @property
    def memory_used_bytes(self) -> int | None:
        """Device wide memory in use, or None when unavailable."""
        if self.memory_total_bytes is None or self.memory_free_bytes is None:
            return None
        return self.memory_total_bytes - self.memory_free_bytes

    @property
    def memory_used_mb(self) -> float | None:
        """Device wide memory in use in MB, or None when unavailable."""
        used = self.memory_used_bytes
        return None if used is None else used / 1024 / 1024

    @property
    def memory_free_mb(self) -> float | None:
        """Device wide free memory in MB, or None when unavailable."""
        return None if self.memory_free_bytes is None else self.memory_free_bytes / 1024 / 1024

    @property
    def memory_total_mb(self) -> float | None:
        """Total device memory in MB, or None when unavailable."""
        return None if self.memory_total_bytes is None else self.memory_total_bytes / 1024 / 1024

    @property
    def memory_free_percent(self) -> float | None:
        """Percentage of free device memory, or None when unavailable."""
        if not self.memory_total_bytes or self.memory_free_bytes is None:
            return None
        return (self.memory_free_bytes / self.memory_total_bytes) * 100

    @property
    def memory_used_percent(self) -> float | None:
        """Percentage of device memory in use, or None when unavailable."""
        used = self.memory_used_bytes
        if not self.memory_total_bytes or used is None:
            return None
        return (used / self.memory_total_bytes) * 100


@dataclass
class CPUStats:
    """Data class containing CPU statistics."""

    utilization_percent: float
    memory_used_bytes: int
    memory_free_bytes: int
    memory_total_bytes: int

    @property
    def memory_used_mb(self) -> float:
        """Memory used in MB."""
        return self.memory_used_bytes / 1024 / 1024

    @property
    def memory_free_mb(self) -> float:
        """Memory free in MB."""
        return self.memory_free_bytes / 1024 / 1024

    @property
    def memory_total_mb(self) -> float:
        """Total memory in MB."""
        return self.memory_total_bytes / 1024 / 1024

    @property
    def memory_free_percent(self) -> float:
        """Percentage of free memory."""
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_free_bytes / self.memory_total_bytes) * 100

    @property
    def memory_used_percent(self) -> float:
        """Percentage of used memory."""
        if self.memory_total_bytes == 0:
            return 0.0
        return (self.memory_used_bytes / self.memory_total_bytes) * 100


class SystemMonitor:
    """Utility class for monitoring GPU and CPU usage.

    This class collects CPU statistics via psutil and Intel GPU memory statistics
    from the PyTorch XPU caching allocator and torch.xpu.mem_get_info (when available).
    It provides properties for accessing current system metrics and logging helpers
    for structured output.
    """

    def __init__(self):
        """Initialize the SystemMonitor."""
        self._gpu_count = 0

        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            DEFAULT_LOGGER_FUNC("PyTorch XPU is not available in this build.")
            return

        try:
            if xpu.is_available():
                self._gpu_count = xpu.device_count()
                DEFAULT_LOGGER_FUNC("Found %d Intel XPU device(s)", self._gpu_count)
            else:
                DEFAULT_LOGGER_FUNC("PyTorch XPU is not available at runtime.")
        except Exception as e:
            logger.debug("Failed to query XPU devices: %s", e)
            self._gpu_count = 0

    @property
    def gpu_count(self) -> int:
        """The number of GPU devices."""
        return self._gpu_count

    def get_gpu_stats(self, device_index: int = 0) -> GPUStats | None:
        """Get current statistics for the specified GPU.

        Args:
            device_index: Index of the GPU device (default: 0)

        Returns:
            GPUStats object with current statistics or None if unavailable
        """
        xpu = getattr(torch, "xpu", None)
        if xpu is None or device_index >= self._gpu_count:
            return None

        try:
            name = xpu.get_device_name(device_index) if hasattr(xpu, "get_device_name") else f"xpu:{device_index}"

            free_bytes: int | None = None
            total_bytes: int | None = None
            try:
                free_bytes, total_bytes = (int(value) for value in xpu.mem_get_info(device_index))
            except (AttributeError, NotImplementedError, RuntimeError) as e:
                logger.debug("Device memory info unavailable for device %s: %s", device_index, e)

            return GPUStats(
                device_index=device_index,
                device_name=name,
                memory_allocated_bytes=int(xpu.memory_allocated(device_index)),
                memory_reserved_bytes=int(xpu.memory_reserved(device_index)),
                memory_free_bytes=free_bytes,
                memory_total_bytes=total_bytes,
            )
        except Exception as e:
            logger.error("Failed to get GPU stats for device %s: %s", device_index, e)
            return None

    def get_all_gpu_stats(self) -> list[GPUStats]:
        """Get statistics for all available GPUs.

        Returns:
            List of GPUStats objects for all available GPUs
        """
        return [stats for stats in (self.get_gpu_stats(i) for i in range(self._gpu_count)) if stats is not None]

    @property
    def cpu_stats(self) -> CPUStats:
        """Current CPU statistics.

        Returns:
            CPUStats object with current CPU statistics
        """
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        return CPUStats(
            utilization_percent=cpu_percent,
            memory_used_bytes=memory.used,
            memory_free_bytes=memory.available,
            memory_total_bytes=memory.total,
        )

    def log_gpu_stats(self, device_index: int | None = None, logger_func=None):
        """Log GPU statistics in a structured format.

        Args:
            device_index: GPU device index to log, or None to log all GPUs
            logger_func: Optional custom logging function, defaults to logger.debug
        """
        if logger_func is None:
            logger_func = DEFAULT_LOGGER_FUNC

        if not self._gpu_count:
            logger_func("GPU statistics unavailable - no XPU GPUs found")
            return

        if device_index is not None:
            stats = self.get_gpu_stats(device_index)
            if stats:
                self._log_single_gpu_stats(stats, logger_func)
            else:
                logger_func("GPU statistics unavailable for device %s", device_index)
        else:
            for i in range(self._gpu_count):
                stats = self.get_gpu_stats(i)
                if stats:
                    self._log_single_gpu_stats(stats, logger_func)

    def _log_single_gpu_stats(self, stats: GPUStats, logger_func):
        """Log statistics for a single GPU.

        Args:
            stats: GPUStats object to log
            logger_func: Logging function to use
        """
        logger_func(f"GPU {stats.device_index} ({stats.device_name}):")

        used_percent = stats.memory_used_percent
        if used_percent is None:
            logger_func("  Device memory: unavailable")
        else:
            logger_func(
                f"  Device memory: {stats.memory_used_mb:.1f} MB used ({used_percent:.1f}%), "
                f"{stats.memory_free_mb:.1f} MB free ({stats.memory_free_percent:.1f}%), "
                f"{stats.memory_total_mb:.1f} MB total"
            )

        logger_func(
            f"  PyTorch allocator: {stats.memory_allocated_mb:.1f} MB allocated, "
            f"{stats.memory_reserved_mb:.1f} MB reserved"
        )

    def log_cpu_stats(self, logger_func=None):
        """Log CPU statistics in a structured format.

        Args:
            logger_func: Optional custom logging function, defaults to logger.debug
        """
        if logger_func is None:
            logger_func = DEFAULT_LOGGER_FUNC

        stats = self.cpu_stats
        logger_func("CPU:")
        logger_func("  Utilization: %.1f%%", stats.utilization_percent)
        logger_func(
            "  Memory: %.1f MB used (%.1f%%), %.1f MB free (%.1f%%), %.1f MB total",
            stats.memory_used_mb,
            stats.memory_used_percent,
            stats.memory_free_mb,
            stats.memory_free_percent,
            stats.memory_total_mb,
        )

    def log_system_stats(self, gpu_indices: list[int] | None = None, logger_func=None, log_label: str | None = None):
        """Log both CPU and GPU statistics in a structured format.

        Args:
            gpu_indices: List of GPU indices to log, or None to log all GPUs
            logger_func: Optional custom logging function, defaults to logger.debug
            log_label: Optional label to identify what stage/operation these stats are for
        """
        if logger_func is None:
            logger_func = DEFAULT_LOGGER_FUNC

        header = "=== System Resource Usage ==="
        if log_label:
            header = f"=== System Resource Usage: {log_label} ==="

        logger_func(header)
        self.log_cpu_stats(logger_func)

        if gpu_indices is not None:
            for idx in gpu_indices:
                self.log_gpu_stats(idx, logger_func)
        else:
            self.log_gpu_stats(None, logger_func)

        footer = "============================="
        if log_label:
            footer = "=" * len(header)

        logger_func(footer)

    @contextmanager
    def system_stats_context(
        self, gpu_indices: list[int] | None = None, logger_func=None, log_label: str | None = None
    ):
        """Context manager that logs system statistics on enter and exit.

        Args:
            gpu_indices: List of GPU indices to log, or None to log all GPUs
            logger_func: Optional custom logging function, defaults to logger.debug
            log_label: Optional label to identify what stage/operation these stats are for

        Yields:
            None
        """
        entry_label = f"{log_label} - Start" if log_label else "Start"
        self.log_system_stats(gpu_indices, logger_func, entry_label)

        try:
            yield
        finally:
            exit_label = f"{log_label} - End" if log_label else "End"
            self.log_system_stats(gpu_indices, logger_func, exit_label)


def system_resource_monitor(name: str | None = None, parent_monitor: SystemMonitor | None = None, logger_func=None):
    """Decorator that logs system statistics before and after function execution.

    This decorator wraps a function to log system resource usage before and after
    the function is called, providing insights into resource consumption.

    Args:
        name: Optional name to identify the decorated function in logs
                If None, the function's name will be used
        parent_monitor: Optional parent monitor to use for logging
        logger_func: Optional custom logging function, defaults to logger.debug
    Returns:
        Decorator function

    Example:
        >>> @system_resource_monitor("my_operation")
        ... def resource_intensive_function():
        ...     pass
        >>> resource_intensive_function()

        >>> @system_resource_monitor()  # Uses function name in logs
        ... def another_function():
        ...     pass
        >>> another_function()
    """

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name if no name provided
            label = name if name is not None else func.__name__

            # Get the SystemMonitor instance if not provided
            monitor = parent_monitor or SystemMonitor()

            with monitor.system_stats_context(log_label=label, logger_func=logger_func):
                return func(*args, **kwargs)

        return wrapper

    return decorator
