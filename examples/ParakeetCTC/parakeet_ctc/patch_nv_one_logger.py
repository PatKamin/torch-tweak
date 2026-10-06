# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Patch nv-one-logger for Lightning 2.6.6 compatibility."""

from parakeet_ctc._dependency_patches import patch_dependencies


def main() -> None:
    """Apply dependency patches from the command line."""
    print("\n".join(patch_dependencies()))


if __name__ == "__main__":
    main()
