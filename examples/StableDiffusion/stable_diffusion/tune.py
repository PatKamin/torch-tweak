# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Tune Stable Diffusion model."""

import os
from logging import basicConfig, getLogger

from stable_diffusion.cmd_args import parse_args
from stable_diffusion.model import get_pipeline
from stable_diffusion.tuning_strategy import hybrid_strategy
from torch_tweak.torch import inspect, save, tune, wrap

logger = getLogger(__name__)


def main():
    """Entry point for the script."""
    log_level = os.environ.get("TORCH_TWEAK_LOG_LEVEL", "INFO")
    basicConfig(level=log_level, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S", force=True)
    args = parse_args()

    pipeline = get_pipeline(model_name=args.model_name)
    input_data = [{"prompt": args.prompt}]
    modules_info = inspect(pipeline, input_data)
    modules = modules_info.get_modules(min_execution_percentage=0.05)
    pipeline = wrap(pipeline, modules, strategy=hybrid_strategy())

    def call_wrapper(*wrapper_args, **wrapper_kwargs):
        for width, height in args.sizes:
            print(f"Generating image with width={width} and height={height}")  # noqa: T201
            pipeline(
                *wrapper_args,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                **wrapper_kwargs,
            )

    logger.info("Tuning module: %s", args.model_name)
    tune(call_wrapper, input_data, batch_sizes=[1])
    logger.info("Tuning completed.")

    save(pipeline, args.tuned_model_path)
    logger.info("Model saved to %s", args.tuned_model_path)


if __name__ == "__main__":
    main()
