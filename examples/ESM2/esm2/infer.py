# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Inference for ESM2 model."""

import logging
import os

import torch
from transformers import AutoTokenizer

import torch_tweak.torch as tt
from esm2.tune import DEVICE, LOG_LEVEL, MODEL_NAME, SAMPLE_SEQUENCE, get_model

logger = logging.getLogger(__name__)


def infer(model_path: str = "esm2_tuned"):
    sample_sequence = os.environ.get("ESM2_SEQUENCE", SAMPLE_SEQUENCE)

    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    with torch.no_grad():
        logger.info("Loading model...")
        model = get_model()
        pipe = tt.load(model, model_path)

        logger.info("Preparing input data...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision="08e4846e537177426273712802403f7ba8261b6c",
        )
        input_data = tokenizer([sample_sequence], return_tensors="pt")

        input_data = {
            "input_ids": input_data["input_ids"].to(DEVICE),
            "attention_mask": input_data["attention_mask"].to(DEVICE),
        }

        logger.info("Running inference...")
        output = pipe(**input_data)
        results = decode(input_data, output, tokenizer)

        logger.info("Inference completed.")

        print("sequence:", sample_sequence)
        print("prediction:", results[0])


def decode(inputs, outputs, tokenizer, k=2):
    logits = outputs.logits
    print("logits shape:", logits.shape)
    mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    results = []
    for batch_idx in range(logits.shape[0]):
        # Find all mask positions for this batch element
        mask_positions = mask_token_indices[mask_token_indices[:, 0] == batch_idx][:, 1]
        if len(mask_positions) == 0:
            results.append([])
            continue
        # For each mask position, get topk predictions
        batch_results = []
        for pos in mask_positions:
            top_k = torch.topk(logits[batch_idx, pos], k=k, dim=-1)
            decoded = tokenizer.decode(top_k.indices)
            batch_results.append(decoded)
        results.append(batch_results)

    return results


if __name__ == "__main__":
    infer()
