# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Model utilities for Stable Diffusion pipeline."""

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline


def get_pipeline(model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", device: str = "xpu"):
    """Get a pretrained Stable Diffusion model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on (default: xpu if available, else cpu)

    Returns:
        DiffusionPipeline: The loaded Stable Diffusion pipeline
    """
    if device is None:
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    if "stable-diffusion-v1-5" in model_name or model_name.endswith("stable-diffusion-v1-5"):
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to(device, dtype=torch.float16)
    return pipe
