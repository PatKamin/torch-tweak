# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Test for passhrough module."""

import pytest

from tests.toy_models import ToyTorchModel
from torch_tweak.torch.module.passthrough_module import PassthroughModule
from torch_tweak.torch.utils.xpu import is_available as is_xpu_available

devices = ["cpu", "xpu"] if is_xpu_available() else ["cpu"]


@pytest.mark.parametrize("device", devices)
def test_call(device):
    model = ToyTorchModel()
    x = model.inputs()[0]
    module = PassthroughModule(model, device=device)

    module(x)
