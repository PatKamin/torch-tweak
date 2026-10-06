# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Runtime dependency patches for the Parakeet RNNT example."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

_NV_ONE_LOGGER_MODULE = "nv_one_logger.training_telemetry.integration.pytorch_lightning"
_NV_ONE_LOGGER_OLD_SIGNATURE = (
    "    def save_checkpoint(self, filepath: Union[str, Path], weights_only: bool = False, "
    "storage_options: Optional[Any] = None) -> None:"
)
_NV_ONE_LOGGER_NEW_SIGNATURE = (
    "    def save_checkpoint(\n"
    "        self, filepath: Union[str, Path], weights_only: Optional[bool] = None, "
    "storage_options: Optional[Any] = None\n"
    "    ) -> None:"
)
_NEMO_EXP_MANAGER_MODULE = "nemo.utils.exp_manager"
_NEMO_EXP_MANAGER_OLD_IMPORT = (
    "from lightning.pytorch.loggers import MLFlowLogger, NeptuneLogger, TensorBoardLogger, WandbLogger"
)
_NEMO_EXP_MANAGER_NEW_IMPORT = (
    "from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger\n"
    "try:\n"
    "    from lightning.pytorch.loggers import NeptuneLogger\n"
    "except ImportError:\n"
    "    NeptuneLogger = None"
)
_NEMO_EXP_MANAGER_OLD_NEPTUNE_BLOCK = "    if create_neptune_logger:\n        if neptune_kwargs is None:"
_NEMO_EXP_MANAGER_NEW_NEPTUNE_BLOCK = (
    "    if create_neptune_logger:\n"
    "        if NeptuneLogger is None:\n"
    '            raise ImportError("NeptuneLogger is not available in this Lightning version")\n'
    "        if neptune_kwargs is None:"
)


def _nv_one_logger_module_path() -> Path:
    spec = find_spec(_NV_ONE_LOGGER_MODULE)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Unable to locate {_NV_ONE_LOGGER_MODULE}")
    return Path(spec.origin)


def _nemo_exp_manager_module_path() -> Path:
    spec = find_spec(_NEMO_EXP_MANAGER_MODULE)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Unable to locate {_NEMO_EXP_MANAGER_MODULE}")
    return Path(spec.origin)


def patch_nv_one_logger() -> bool:
    """Patch the installed nv-one-logger integration if needed.

    Returns True when the file was modified and False when it was already patched.
    """
    path = _nv_one_logger_module_path()
    source = path.read_text()

    if _NV_ONE_LOGGER_NEW_SIGNATURE in source:
        return False
    if _NV_ONE_LOGGER_OLD_SIGNATURE not in source:
        raise RuntimeError(f"Unexpected {_NV_ONE_LOGGER_MODULE} save_checkpoint signature in {path}")

    path.write_text(source.replace(_NV_ONE_LOGGER_OLD_SIGNATURE, _NV_ONE_LOGGER_NEW_SIGNATURE, 1))
    return True


def nv_one_logger_patch_status() -> str:
    """Return the installed nv-one-logger integration path and patch status."""
    path = _nv_one_logger_module_path()
    modified = patch_nv_one_logger()
    status = "patched" if modified else "already patched"
    return f"{path}: {status}"


def patch_nemo_exp_manager() -> bool:
    """Patch NeMo exp_manager for Lightning 2.6.6 logger compatibility if needed."""
    path = _nemo_exp_manager_module_path()
    source = path.read_text()

    modified = False
    if _NEMO_EXP_MANAGER_NEW_IMPORT not in source:
        if _NEMO_EXP_MANAGER_OLD_IMPORT not in source:
            raise RuntimeError(f"Unexpected {_NEMO_EXP_MANAGER_MODULE} logger import in {path}")
        source = source.replace(_NEMO_EXP_MANAGER_OLD_IMPORT, _NEMO_EXP_MANAGER_NEW_IMPORT, 1)
        modified = True

    if _NEMO_EXP_MANAGER_NEW_NEPTUNE_BLOCK not in source:
        if _NEMO_EXP_MANAGER_OLD_NEPTUNE_BLOCK not in source:
            raise RuntimeError(f"Unexpected {_NEMO_EXP_MANAGER_MODULE} Neptune logger block in {path}")
        source = source.replace(_NEMO_EXP_MANAGER_OLD_NEPTUNE_BLOCK, _NEMO_EXP_MANAGER_NEW_NEPTUNE_BLOCK, 1)
        modified = True

    if modified:
        path.write_text(source)
    return modified


def nemo_exp_manager_patch_status() -> str:
    """Return the installed NeMo exp_manager path and patch status."""
    path = _nemo_exp_manager_module_path()
    modified = patch_nemo_exp_manager()
    status = "patched" if modified else "already patched"
    return f"{path}: {status}"


def patch_dependencies() -> list[str]:
    """Apply all runtime dependency patches and return their statuses."""
    return [nv_one_logger_patch_status(), nemo_exp_manager_patch_status()]
