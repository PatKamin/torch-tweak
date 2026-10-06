# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torch_tweak.torch.backend import (
    TorchEagerBackend,
    TorchInductorBackend,
)
from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.backend.torchao_backend import TorchAOBackend
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata


class Latch:
    def __init__(self):
        self.latched = False

    def signal(self):
        if self.latched:
            raise RuntimeError("Latch already latched - should be called only once.")
        self.latched = True


class ToyModel(nn.Module):
    def forward(self, x: torch.Tensor, latch: Latch):
        latch.signal()
        return x


def build_backend(backend: Backend, torch_device: torch.device, tmp_path: Path):
    model = ToyModel()
    x = torch.randn(1, 2)
    args = (x, Latch())
    kwargs = {}
    input_spec = SampleMetadata.from_inputs(args, kwargs)
    output_spec = SampleMetadata.from_outputs(x)
    graph_spec = GraphSpec(name="test_model", input_spec=input_spec, output_spec=output_spec)
    backend.build(model, graph_spec, [(args, kwargs)], device=torch_device, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "backend_class",
    [
        TorchEagerBackend,
        TorchAOBackend,
        TorchInductorBackend,
        # OpenVINOBackend, - not supported yet
    ],
)
def test_backend_build_is_idempotent(backend_class, torch_device, tmp_path):
    backend = backend_class()
    build_backend(backend, torch_device, tmp_path)
