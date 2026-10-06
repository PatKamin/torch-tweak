# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Inference script for Stable Diffusion model."""

import os
from logging import basicConfig, getLogger
from pathlib import Path

from stable_diffusion.cmd_args import parse_args
from stable_diffusion.model import get_pipeline
from torch_tweak.torch import load

logger = getLogger(__name__)


def _save_generation(pipe, prompt: str, width: int, height: int, steps: int, image_path: Path) -> None:
    result = pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps)
    result.images[0].save(image_path)
    logger.info("Generated image saved to: %s", image_path)


def main():
    """Entry point for the script."""
    basicConfig(level="INFO", format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()
    output_dir = os.environ.get("TORCH_TWEAK_OUTPUT_DIR", args.output_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    slug = args.prompt[:20].replace(" ", "_")

    pipe = get_pipeline(model_name=args.model_name)

    logger.info("Generating images on original pipeline")
    for width, height in args.sizes:
        path = output_path / f"stable_diffusion_output_orig_{slug}_{width}x{height}.jpg"
        _save_generation(pipe, args.prompt, width, height, args.steps, path)

    logger.info("Loading tuned pipeline from %s", args.tuned_model_path)
    pipe = load(pipe, args.tuned_model_path)

    logger.info("Generating images on tuned pipeline")
    for width, height in args.sizes:
        path = output_path / f"stable_diffusion_output_{slug}_{width}x{height}.jpg"
        _save_generation(pipe, args.prompt, width, height, args.steps, path)


if __name__ == "__main__":
    main()
