# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphSpec."""

import copy

import pytest
import torch

from tests.utilities.helpers import assert_only_primitives, save_and_load_weights_only
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata


@pytest.fixture
def graph_spec():
    return GraphSpec(
        name="graph_0",
        input_spec=SampleMetadata.from_inputs(
            args=(torch.randn(2, 3), 1),
            kwargs={"t": torch.zeros(4, dtype=torch.int64), "flag": True},
            strict=True,
        ),
        output_spec=SampleMetadata.from_outputs(torch.randn(2, 3), strict=True),
    )


def test_graph_spec_to_dict_holds_only_primitives(graph_spec):
    assert_only_primitives(graph_spec.to_dict())


def test_graph_spec_to_from_dict(graph_spec):
    result = GraphSpec.from_dict(graph_spec.to_dict())

    assert result.name == graph_spec.name
    assert result.input_spec == graph_spec.input_spec
    assert result.output_spec == graph_spec.output_spec


def test_graph_spec_to_from_dict_preserves_tensor_details(graph_spec):
    result = GraphSpec.from_dict(graph_spec.to_dict())

    for spec_name in ("input_spec", "output_spec"):
        expected_specs = getattr(graph_spec, spec_name).tensor_specs
        actual_specs = getattr(result, spec_name).tensor_specs
        for expected, actual in zip(expected_specs, actual_specs, strict=True):
            assert expected.name == actual.name
            assert expected.shape == actual.shape
            assert expected.dtype == actual.dtype

    assert [(str(locator), name, value) for locator, name, value in result.input_spec.other_data] == [
        (str(locator), name, value) for locator, name, value in graph_spec.input_spec.other_data
    ]


def test_graph_spec_survives_weights_only_load(graph_spec, tmp_path):
    """A serialized graph spec must load without reconstructing arbitrary objects."""
    result = GraphSpec.from_dict(save_and_load_weights_only(graph_spec.to_dict(), tmp_path))

    assert result.name == graph_spec.name
    assert result.input_spec == graph_spec.input_spec
    assert result.output_spec == graph_spec.output_spec


def test_graph_spec_from_dict_rejects_invalid_data():
    with pytest.raises(ValueError, match="Invalid dictionary format"):
        GraphSpec.from_dict({"name": "graph_0"})


def test_graph_spec_from_dict_does_not_mutate_input(graph_spec):
    data = graph_spec.to_dict()
    before = copy.deepcopy(data)

    GraphSpec.from_dict(data)

    assert data == before
