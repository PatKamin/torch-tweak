# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Hugging Face integrations."""

try:
    from transformers import DynamicCache, StaticCache, StaticLayer, StaticSlidingWindowLayer

    from torch_tweak.torch.module.locator import Locator

    # enable static cache support
    Locator.register_user_type(StaticCache, only_tensors=True)
    Locator.register_user_type(StaticLayer, only_tensors=True)
    Locator.register_user_type(StaticSlidingWindowLayer, only_tensors=True)
    # disable inspection of dynamic cache as it is not supported by the backends
    # the only exception is TorchEagerBackend as it is basically passthrough with some dtype casts
    Locator.ignore_type(DynamicCache)
except ImportError:
    pass
