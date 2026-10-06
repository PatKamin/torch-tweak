# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Test for device utilities."""

import pytest
import torch

from torch_tweak.torch.utils.device import get_device


def test_get_device_cpu_string():
    """Test get_device with CPU string input."""
    result = get_device("cpu")
    assert isinstance(result, torch.device)
    assert result.type == "cpu"
    assert result.index is None


def test_get_device_cpu_torch_device():
    """Test get_device with CPU torch.device input."""
    cpu_device = torch.device("cpu")
    result = get_device(cpu_device)
    assert isinstance(result, torch.device)
    assert result.type == "cpu"
    assert result.index is None


def test_get_device_xpu_string_no_index():
    """Test get_device with 'xpu' string (should default to xpu:0)."""
    result = get_device("xpu")
    assert isinstance(result, torch.device)
    assert result.type == "xpu"
    assert result.index == 0


def test_get_device_xpu_string_with_index():
    """Test get_device with 'xpu:N' string."""
    for device_id in [0, 1, 2, 10]:
        result = get_device(f"xpu:{device_id}")
        assert isinstance(result, torch.device)
        assert result.type == "xpu"
        assert result.index == device_id


def test_get_device_xpu_torch_device_no_index():
    """Test get_device with xpu torch.device without index."""
    xpu_device = torch.device("xpu")
    result = get_device(xpu_device)
    assert isinstance(result, torch.device)
    assert result.type == "xpu"
    assert result.index == 0


def test_get_device_xpu_torch_device_with_index():
    """Test get_device with xpu torch.device with index."""
    for device_id in [0, 1, 2, 5]:
        xpu_device = torch.device(f"xpu:{device_id}")
        result = get_device(xpu_device)
        assert isinstance(result, torch.device)
        assert result.type == "xpu"
        assert result.index == device_id


def test_get_device_invalid_string_inputs():
    """Test get_device with various invalid string inputs."""
    invalid_inputs = [
        "",
        "gpu",
        "cuda",
        "cuda:0",
        "xpu:",
        "xpu:a",
        "xpu:-1",
        "xpu:1.5",
        "xpu: 0",
        " xpu:0",
        "xpu:0 ",
        "XPU:0",
        "mps",
        "tpu",
        "invalid",
        "xpu:0:0",
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError, match="Invalid device"):
            get_device(invalid_input)


def test_get_device_invalid_torch_device_inputs():
    """Test get_device with invalid torch.device inputs."""
    # Test with unsupported device types
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        with pytest.raises(ValueError, match="Invalid device"):
            get_device(mps_device)

    # We can't easily create torch.device with invalid types without causing errors,
    # so we'll test with a mock or patch if needed


def test_get_device_none_input():
    """Test get_device with None input."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device(None)


def test_get_device_integer_input():
    """Test get_device with integer input."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device(0)


def test_get_device_list_input():
    """Test get_device with list input."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device(["cuda:0"])


def test_get_device_return_type_consistency():
    """Test that get_device always returns torch.device objects."""
    test_cases = [
        "cpu",
        "xpu",
        "xpu:0",
        "xpu:1",
        torch.device("cpu"),
        torch.device("xpu:0"),
        torch.device("xpu:1"),
    ]

    for test_case in test_cases:
        result = get_device(test_case)
        assert isinstance(result, torch.device)


def test_get_device_large_device_indices():
    """Test get_device with large device indices."""
    large_indices = [99, 100, 127]

    for index in large_indices:
        result = get_device(f"xpu:{index}")
        assert isinstance(result, torch.device)
        assert result.type == "xpu"
        assert result.index == index


def test_get_device_device_indices_from_out_of_range():
    """Test get_device with device indices from out of range."""
    large_indices = [128, 512, 999]

    for index in large_indices:
        with pytest.raises(ValueError, match=f"Invalid device index: {index}. Expected 0-127"):
            get_device(f"xpu:{index}")


def test_get_device_edge_case_zero_index():
    """Test get_device specifically with zero index."""
    # Test string input
    result_str = get_device("xpu:0")
    assert result_str.type == "xpu"
    assert result_str.index == 0

    # Test torch.device input
    result_device = get_device(torch.device("xpu:0"))
    assert result_device.type == "xpu"
    assert result_device.index == 0

    # Test that "xpu" becomes "xpu:0"
    result_default = get_device("xpu")
    assert result_default.type == "xpu"
    assert result_default.index == 0
