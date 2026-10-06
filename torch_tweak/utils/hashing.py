# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Hashing utilities for the Torch Tweak package."""

import hashlib


def hash_string(s: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
