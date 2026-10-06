# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO backend components."""

from torch_tweak.torch.backend.openvino.openvino_backend import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.backend.openvino.ov_model_converter import OpenVINOModelConverter

__all__ = ["OpenVINOModelConverter", "OpenVINOBackend", "OpenVINOBackendConfig"]
