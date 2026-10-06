# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Device utilities."""

import re

import torch


def get_device(device: str | torch.device) -> torch.device:
    """Normalize a device string/object to torch.device.

    Returns:
        torch.device: Normalized device.
    """
    device_str = None
    if isinstance(device, torch.device):
        if device.type == "xpu":
            device_str = f"{device.type}:{device.index or 0}"
        elif device.type in ["cpu", "meta"]:
            device_str = device.type
        else:
            raise ValueError(f"Invalid device: {device}. Expected 'xpu', 'cpu' or 'meta'")
    elif isinstance(device, str):
        if device in ["cpu", "meta"]:
            device_str = device
        else:
            kind = "xpu"
            pattern = rf"^{kind}:(\d+)$"
            match = re.match(pattern, device)
            if match:
                index = int(match.group(1))
                if index < 0 or index > 127:
                    raise ValueError(f"Invalid device index: {index}. Expected 0-127.")
                device_str = device

            pattern = rf"^{kind}$"
            match = re.match(pattern, device)
            if match:
                device_str = f"{kind}:0"

    if device_str is None:
        raise ValueError(f"Invalid device: {device}. Expected 'cpu', 'meta', 'xpu' or 'xpu:<id>' (id 0-127)")

    return torch.device(device_str)
