# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the header of ``.tt`` checkpoint files."""

import io
import json
import logging
import shutil

import pytest
import torch

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.checkpoint import storage_tasks
from torch_tweak.torch.checkpoint.checkpoint_header import (
    IncompatibleCheckpointError,
    read_header,
    tuning_environment,
    write_header,
)
from torch_tweak.torch.checkpoint.local_torch_storage import LocalTorchStorage
from torch_tweak.torch.checkpoint.storage_tasks import UnzipLoadTask, ZipSaveTask

HEADER_OF_CPU_A = b"\x01\x00\x00\x00" + b"\x0c\x00\x00\x00" + b'{"cpu": "a"}'


def test_header_is_the_format_version_then_the_environment_record():
    fp = io.BytesIO()

    write_header(fp, {"cpu": "a"})

    assert fp.getvalue() == HEADER_OF_CPU_A


def test_reading_the_header_leaves_the_file_at_the_archive():
    fp = io.BytesIO(HEADER_OF_CPU_A + b"PK\x03\x04")

    read_header(fp, {"cpu": "a"})

    assert fp.read() == b"PK\x03\x04"


def test_reading_the_header_returns_the_recorded_environment():
    fp = io.BytesIO(HEADER_OF_CPU_A)

    assert read_header(fp, {"cpu": "a"}) == {"cpu": "a"}


def test_another_format_version_is_rejected_before_the_environment_is_read():
    fp = io.BytesIO(b"\x02\x00\x00\x00" + b"not a header of version 1")

    with pytest.raises(IncompatibleCheckpointError, match="format version 2 is not 1"):
        read_header(fp, {"cpu": "a"})


def test_a_checkpoint_saved_before_the_format_version_is_rejected():
    fp = io.BytesIO(b"PK\x03\x04")

    with pytest.raises(IncompatibleCheckpointError, match="saved by an older torch-tweak"):
        read_header(fp, {"cpu": "a"})


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"\x01\x00",
        b"PK\x03",
        b"\x01\x00\x00\x00",
        b"\x01\x00\x00\x00" + b"\x0c\x00",
        b"\x01\x00\x00\x00" + b"\x0c\x00\x00\x00" + b'{"cpu":',
    ],
)
def test_a_truncated_checkpoint_header_is_rejected(header):
    fp = io.BytesIO(header)

    with pytest.raises(IncompatibleCheckpointError, match="truncated"):
        read_header(fp, {"cpu": "a"})


def test_an_environment_record_over_the_limit_is_rejected():
    fp = io.BytesIO(b"\x01\x00\x00\x00" + b"\x01\x00\x01\x00")

    with pytest.raises(IncompatibleCheckpointError, match="65537 bytes, the limit is 65536"):
        read_header(fp, {"cpu": "a"})


def test_an_environment_record_that_is_not_json_keeps_the_decode_error():
    fp = io.BytesIO(b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"{")

    with pytest.raises(IncompatibleCheckpointError, match="not UTF-8 JSON") as raised:
        read_header(fp, {"cpu": "a"})

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


def test_an_environment_record_that_is_not_utf8_keeps_the_decode_error():
    fp = io.BytesIO(b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\xff")

    with pytest.raises(IncompatibleCheckpointError, match="not UTF-8 JSON") as raised:
        read_header(fp, {"cpu": "a"})

    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    "header",
    [
        b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"[]",
        b"\x01\x00\x00\x00" + b"\x04\x00\x00\x00" + b"null",
        b"\x01\x00\x00\x00" + b"\x0a\x00\x00\x00" + b'{"cpu": 1}',
    ],
)
def test_an_environment_record_that_is_not_an_object_of_strings_is_rejected(header):
    fp = io.BytesIO(header)

    with pytest.raises(IncompatibleCheckpointError, match="not an object of strings"):
        read_header(fp, {"cpu": "a"})


def test_a_different_cpu_is_rejected_without_strict():
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with pytest.raises(IncompatibleCheckpointError, match="cpu: tuned with 'a', running 'b'"):
        read_header(fp, {"cpu": "b"})


def test_a_checkpoint_tuned_in_the_same_environment_logs_no_warning(caplog):
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with caplog.at_level(logging.WARNING):
        read_header(fp, {"cpu": "a"})

    assert caplog.records == []


def test_strict_reading_rejects_a_checkpoint_tuned_in_another_environment():
    fp = io.BytesIO()
    write_header(fp, {"cpu": "a", "torch": "1"})
    fp.seek(0)

    with pytest.raises(IncompatibleCheckpointError, match="torch: tuned with '1', running '2'"):
        read_header(fp, {"cpu": "a", "torch": "2"}, strict=True)


def test_strict_reading_rejects_a_checkpoint_tuned_without_a_library_now_installed():
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with pytest.raises(IncompatibleCheckpointError, match="openvino: tuned with None, running '2026.1.0'"):
        read_header(fp, {"cpu": "a", "openvino": "2026.1.0"}, strict=True)


def test_a_different_machine_is_not_extracted(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"machine": "x86_64", "cpu": "a"}).save(folder, {})
    shutil.rmtree(folder)

    with pytest.raises(IncompatibleCheckpointError, match="machine: tuned with 'x86_64', running 'aarch64'"):
        UnzipLoadTask(environment={"machine": "aarch64", "cpu": "a"}).load(folder)

    assert not folder.exists()


def test_strict_local_storage_rejects_a_checkpoint_tuned_in_another_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_tasks, "tuning_environment", lambda: {"cpu": "a", "torch": "1"})
    LocalTorchStorage(base_folder=tmp_path).save("model.tt", {"weight": torch.tensor([1.0])})
    monkeypatch.setattr(storage_tasks, "tuning_environment", lambda: {"cpu": "a", "torch": "2"})

    with pytest.raises(IncompatibleCheckpointError, match="torch: tuned with '1', running '2'"):
        LocalTorchStorage(base_folder=tmp_path, strict_environment=True).load("model.tt")


def test_tuning_environment_records_the_torch_version():
    environment = tuning_environment()

    assert environment["torch"] == torch.__version__


@pytest.mark.parametrize(
    ("recorded", "running"),
    [
        ({"accelerator": "xpu: Arc A770", "driver": "1"}, {"accelerator": "xpu: Arc B70", "driver": "1"}),
        ({"cpu": "a"}, {"cpu": "a", "accelerator": "xpu: Arc B70"}),
        ({"accelerator": "xpu: Arc B70", "driver": "1"}, {"driver": "1"}),
    ],
)
def test_a_checkpoint_tuned_for_another_accelerator_is_not_extracted(tmp_path, recorded, running):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment=recorded).save(folder, {})
    shutil.rmtree(folder)

    with pytest.raises(IncompatibleCheckpointError, match="other hardware"):
        UnzipLoadTask(environment=running).load(folder)

    assert not folder.exists()


def test_a_checkpoint_tuned_with_another_driver_is_extracted(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"accelerator": "xpu: Arc B70", "driver": "1"}).save(folder, {})
    shutil.rmtree(folder)

    UnzipLoadTask(environment={"accelerator": "xpu: Arc B70", "driver": "2"}).load(folder)

    assert (folder / "state.txt").read_text() == "tuned"


def test_another_driver_is_reported_in_a_warning(caplog):
    fp = io.BytesIO()
    write_header(fp, {"accelerator": "xpu: Arc B70", "driver": "1"})
    fp.seek(0)

    with caplog.at_level(logging.WARNING):
        read_header(fp, {"accelerator": "xpu: Arc B70", "driver": "2"})

    assert "driver: tuned with '1', running '2'" in caplog.text


def test_a_new_environment_field_is_extracted(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"accelerator": "xpu: Arc B70", "driver": "1"}).save(folder, {})
    shutil.rmtree(folder)

    UnzipLoadTask(environment={"accelerator": "xpu: Arc B70", "driver": "1", "kernel": "6.8"}).load(folder)

    assert (folder / "state.txt").read_text() == "tuned"


def test_strict_loading_does_not_extract_a_checkpoint_with_a_new_environment_field(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"accelerator": "xpu: Arc B70", "driver": "1"}).save(folder, {})
    shutil.rmtree(folder)

    with pytest.raises(IncompatibleCheckpointError, match="kernel: tuned with None, running '6.8'"):
        UnzipLoadTask(
            environment={"accelerator": "xpu: Arc B70", "driver": "1", "kernel": "6.8"}, strict_environment=True
        ).load(folder)

    assert not folder.exists()


def test_strict_loading_does_not_extract_a_checkpoint_tuned_with_another_driver(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"accelerator": "xpu: Arc B70", "driver": "1"}).save(folder, {})
    shutil.rmtree(folder)

    with pytest.raises(IncompatibleCheckpointError, match="driver: tuned with '1', running '2'"):
        UnzipLoadTask(environment={"accelerator": "xpu: Arc B70", "driver": "2"}, strict_environment=True).load(folder)

    assert not folder.exists()


@requires_xpu
def test_tuning_environment_records_the_accelerator_name_apart_from_the_driver():
    environment = tuning_environment()

    assert environment["accelerator"] == f"xpu: {torch.xpu.get_device_properties(0).name}"
    assert environment["driver"] == torch.xpu.get_device_properties(0).driver_version
