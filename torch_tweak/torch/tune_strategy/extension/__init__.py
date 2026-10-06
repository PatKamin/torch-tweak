# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Strategy extensions for tune strategy.

Extensions should augment tune method instead of implementing _tune method.
"""

from torch_tweak.torch.tune_strategy.extension.find_max_batch_size_extension import (
    FindMaxBatchSizeExtensionConfig,
    TuneStrategyFindMaxBatchSizeExtension,
)

__all__ = [
    "TuneStrategyFindMaxBatchSizeExtension",
    "FindMaxBatchSizeExtensionConfig",
]
