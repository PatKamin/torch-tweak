# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Common command line arguments for Stable Diffusion."""

import argparse

DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def parse_sizes(sizes_str: str) -> list[tuple[int, int]]:
    """Parse sizes string into list of width and height tuples.

    Args:
        sizes_str: String with multiple size combinations separated by spaces
                  (e.g., "128,128 256,256" or "512,512")

    Returns:
        List of (width, height) tuples as integers

    Raises:
        ValueError: If the format is invalid or values are not positive integers
    """
    try:
        size_combinations = sizes_str.split()
        result = []

        for combo in size_combinations:
            parts = combo.split(",")
            if len(parts) != 2:
                raise ValueError(f"Size combination '{combo}' must be in format 'width,height' (e.g., '512,512')")

            width = int(parts[0].strip())
            height = int(parts[1].strip())

            if width <= 0 or height <= 0:
                raise ValueError(f"Width and height in '{combo}' must be positive integers")

            result.append((width, height))

        return result
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError("Width and height must be valid integers") from e
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tuned Stable Diffusion model")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape with mountains and a lake",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=[(512, 512)],
        help="Image dimensions as space-separated width,height pairs (e.g., '512,512' or '512,512 768,768')",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--tuned-model-path",
        type=str,
        default="stable_diffusion.tt",
        help="Path to save the tuned model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for generated images (inference only; overridden by TORCH_TWEAK_OUTPUT_DIR if set)",
    )
    args = parser.parse_args()
    return args
