# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
from parakeet_ctc._dependency_patches import patch_nemo_exp_manager, patch_nv_one_logger

patch_nv_one_logger()
patch_nemo_exp_manager()
