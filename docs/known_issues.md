<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
NOTE: This file has been modified by Intel Corporation.
-->

# Known Issues and Limitations

- Torch Tweak currently only supports single-GPU configurations - no multi-GPU support.
- OpenVINO conversion or compilation may fail for models with unsupported operators,
complex dynamic control flow, symbolic shape constraints, or memory requirements beyond available resources.
- Torch Inductor may encounter graph breaks on unsupported Python constructs or operations,
resulting in partial or failed compilation.
- These backend-specific limitations depend on the model.
See [backend limitations](guides/tune_strategies/tune_strategies.md#why-backends-can-fail).
