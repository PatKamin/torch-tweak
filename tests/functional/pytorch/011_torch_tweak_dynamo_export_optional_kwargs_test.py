# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

from logging import getLogger
from pathlib import Path

import openvino as ov
import pytest
import torch
import torch.nn as nn

from torch_tweak.torch.backend.openvino.ov_model_converter import OpenVINOModelConverter
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata

logger = getLogger(Path(__file__).stem)


class ModelWithOptionalKwargs(nn.Module):
    """Test model with optional kwargs including None defaults."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x, optional_flag=None, optional_value=None):
        """Forward with optional kwargs.

        Args:
            x: Input tensor
            optional_flag: Optional flag (None or tensor)
            optional_value: Optional value (None or tensor)
        """
        out = self.linear(x)
        if optional_flag is not None:
            out = out * optional_flag
        if optional_value is not None:
            out = out + optional_value
        return out


def _input_names_from_ir(xml_path: Path) -> list[str]:
    model = ov.Core().read_model(xml_path)
    return [port.get_any_name() for port in model.inputs]


@pytest.mark.functional
def test_export_with_none_kwargs(tmp_path: Path, torch_device: str):
    """Test dynamo export with None-valued kwargs - should filter them out."""
    logger.info("Test 1: Export with None kwargs")

    model = ModelWithOptionalKwargs().eval().to(torch_device)
    x = torch.randn(2, 10, device=torch_device)

    sample = ((x,), {})
    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    out_dir = tmp_path / "none_kwargs"
    converter = OpenVINOModelConverter(output_dir=out_dir, use_dynamo=True)
    xml_path = converter.convert(module=model, sample=sample, graph_spec=graph_spec)

    input_names = _input_names_from_ir(xml_path)
    assert len(input_names) == 1, f"Expected 1 input, got {len(input_names)}"
    assert input_names[0] == "args_0"

    logger.info("✓ Test 1 passed: None kwargs filtered correctly")


@pytest.mark.functional
def test_export_with_kwargs_wrong_order(tmp_path: Path, torch_device: str):
    """Test dynamo export with kwargs in wrong order - should reorder to match forward signature."""
    logger.info("Test 2: Export with kwargs in wrong order")

    model = ModelWithOptionalKwargs().eval().to(torch_device)
    x = torch.randn(2, 10, device=torch_device)
    flag = torch.tensor(2.0, device=torch_device)
    value = torch.tensor(1.0, device=torch_device)

    sample = ((x,), {"optional_value": value, "optional_flag": flag})
    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    out_dir = tmp_path / "wrong_order"
    converter = OpenVINOModelConverter(output_dir=out_dir, use_dynamo=True)
    xml_path = converter.convert(module=model, sample=sample, graph_spec=graph_spec)

    input_names = _input_names_from_ir(xml_path)
    assert len(input_names) == 3, f"Expected 3 inputs, got {len(input_names)}"
    assert input_names[0] == "args_0"
    assert input_names[1] == "kwargs_optional_flag"
    assert input_names[2] == "kwargs_optional_value"

    logger.info("✓ Test 2 passed: Kwargs reordered correctly")


@pytest.mark.functional
def test_export_with_mixed_none_and_non_none_kwargs(tmp_path: Path, torch_device: str):
    """Test dynamo export with mixed None and non-None kwargs."""
    logger.info("Test 3: Export with mixed None and non-None kwargs")

    model = ModelWithOptionalKwargs().eval().to(torch_device)
    x = torch.randn(2, 10, device=torch_device)
    flag = torch.tensor(2.0, device=torch_device)

    sample = ((x,), {"optional_flag": flag, "optional_value": None})
    args, kwargs = sample
    output = model(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    out_dir = tmp_path / "mixed_kwargs"
    converter = OpenVINOModelConverter(output_dir=out_dir, use_dynamo=True)
    xml_path = converter.convert(module=model, sample=sample, graph_spec=graph_spec)

    input_names = _input_names_from_ir(xml_path)
    assert len(input_names) == 2, f"Expected 2 inputs, got {len(input_names)}"
    assert input_names[0] == "args_0"
    assert input_names[1] == "kwargs_optional_flag"

    logger.info("✓ Test 3 passed: Mixed None and non-None kwargs handled correctly")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
