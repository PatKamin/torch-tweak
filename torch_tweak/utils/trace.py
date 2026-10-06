# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Lightweight tracing utilities.

This project originally used NVTX annotations. Intel's equivalent is ITT, exposed by PyTorch as
`torch.profiler.itt` and readable by tools such as VTune, so the `annotate` decorator emits ITT
ranges instead.

ITT has no notion of domains, so `domain` is folded into the task label as a prefix.

Availability is resolved once, at decoration time. When ITT is missing the decorator returns the
function untouched, so instrumentation never becomes a hard dependency and adds no per-call cost
on hot inference paths.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from torch.profiler import itt

T = TypeVar("T", bound=Callable[..., Any])


def _label(message: str | None, domain: str | None, func: Callable[..., Any]) -> str:
    name = message or getattr(func, "__qualname__", None) or getattr(func, "__name__", "<unknown>")
    return f"{domain}/{name}" if domain else name


def annotate(*, message: str | None = None, domain: str | None = None) -> Callable[[T], T]:
    """Decorate a function so its execution is reported as an ITT task.

    Args:
        message: Task label. Defaults to the qualified name of the decorated function.
        domain: Optional prefix prepended to the label as ``domain/label``, since ITT has no
            separate domain concept.

    Returns:
        A decorator emitting an ITT range around each call, or the identity decorator when ITT
        is unavailable.
    """

    def _decorator(func: T) -> T:
        if not itt.is_available():
            return func

        label = _label(message, domain, func)

        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            itt.range_push(label)
            try:
                return func(*args, **kwargs)
            finally:
                itt.range_pop()

        return _wrapper  # type: ignore[return-value]

    return _decorator
