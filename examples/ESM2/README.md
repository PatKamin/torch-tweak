<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# ESM2 Model Tuning with Torch Tweak

> Evolutionary-scale prediction of atomic level protein structure with a language model

This example demonstrates how to use Torch Tweak to tune the ESM2 transformer protein language model - `facebook/esm2_t33_650M_UR50D` - from Hugging Face's transformer library.

## Environment Setup

You can use either of the following options to set up the environment:

### Option 1 - virtual environment managed by you

Activate your virtual environment and install the dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/xpu \
        --extra-index-url https://pypi.org/simple \
        .
```

### Option 2 - virtual environment managed by `uv`

Install dependencies:

```bash
uv sync
```

## Usage

### Tuning the model

To tune the ESM2 model, run:

```bash
tune
```

or

```bash
uv run tune
```

After tuning, run inference

```bash
inference
```

or

```bash
uv run inference
```

## Model Details

ESM-2 (Evolutionary Scale Modeling-2) is a state-of-the-art protein language model developed by Facebook AI, designed to analyze and interpret protein sequences using deep learning techniques. It is trained on a masked language modeling objective, meaning it predicts missing amino acids in protein sequences, which enables it to learn patterns relevant for understanding structure and function.

## Links

* [Hugging Face Model](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
* [Research Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2)
