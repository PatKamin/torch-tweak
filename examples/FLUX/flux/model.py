# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Model utilities for Flux pipeline."""

import torch
from diffusers import DiffusionPipeline


def get_pipeline(model_name: str = "black-forest-labs/FLUX.2-klein-4B", device: str = "xpu"):
    """Get a pretrained Flux model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        DiffusionPipeline: The loaded pipeline class declared by the model
    """
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device, dtype=torch.float16)
    torch.xpu.empty_cache()
    return pipe
