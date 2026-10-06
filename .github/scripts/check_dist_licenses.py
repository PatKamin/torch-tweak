# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check that built distributions ship the required license files.

Build backends stay silent when those files go missing, so the artifacts are
inspected here instead of trusting the build.
"""

import argparse
import logging
import sys
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO, format="%(message)s")

REQUIRED_FILES = ("LICENSE", "third-party-programs.txt")

# Location that PEP 639 reserves for license files inside a wheel.
CANONICAL_WHEEL_DIR = ".dist-info/licenses/"


def _members(dist: Path) -> list[str]:
    """Return the member names stored inside a wheel or an sdist."""
    if dist.suffix == ".whl":
        with zipfile.ZipFile(dist) as archive:
            return archive.namelist()
    with tarfile.open(dist) as archive:
        return archive.getnames()


def _problems(dist: Path, canonical: bool) -> list[str]:
    """Report every required file that is missing from dist or stored in the wrong place."""
    members = _members(dist)
    problems = []

    for filename in REQUIRED_FILES:
        found = [name for name in members if name.rsplit("/", 1)[-1] == filename]
        if not found:
            problems.append(f"missing {filename}")
        elif canonical and dist.suffix == ".whl" and not any(CANONICAL_WHEEL_DIR in name for name in found):
            problems.append(f"{filename} is not under <name>{CANONICAL_WHEEL_DIR}")

    return problems


def main() -> int:
    """Inspect every distribution given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dists", nargs="+", type=Path, help="wheel or sdist files to inspect")
    parser.add_argument("--canonical", action="store_true", help="also require the PEP 639 location in wheels")
    args = parser.parse_args()

    failed = False
    for dist in sorted(args.dists):
        problems = _problems(dist, args.canonical)
        failed = failed or bool(problems)
        for problem in problems:
            logger.error("FAIL  %s: %s", dist.name, problem)
        if not problems:
            logger.info("OK    %s", dist.name)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
