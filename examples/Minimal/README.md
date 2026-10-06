<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Minimal backend selection

Tunes a small MLP and prints the faster of Torch Inductor and Torch Eager. Uses an Intel XPU when one is available, and CPU otherwise.

## Run from the repository

From the repository root, after `uv sync --extra dev`:

```bash
uv run python examples/Minimal/demo.py
```

## Run this example on its own

From this directory. PyTorch comes from the `xpu` extra (Intel XPU wheels).

```bash
uv sync
uv run demo
```
