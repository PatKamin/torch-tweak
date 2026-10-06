# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
from importlib.metadata import version

try:
    __version__ = version("torch_tweak")
except Exception:
    __version__ = "0.0.0+unknown"
