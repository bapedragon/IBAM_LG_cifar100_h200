#!/usr/bin/env python3
"""Run Ours on Flowers-102."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.Ours.core import cli_main


PROTOCOL_DEFAULTS = (
    ("--protocol-name", "flowers102_deit_ti_common_kd_v1"),
    ("--student-epochs", "200"),
    ("--batch-size", "64"),
    ("--lr", "0.0005"),
    ("--weight-decay", "0.05"),
    ("--warmup-epochs", "5"),
    ("--label-smoothing", "0.1"),
)


def has_option(option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


if __name__ == "__main__":
    if has_option("--dataset"):
        raise SystemExit("This wrapper fixes --dataset flowers102; remove --dataset.")
    sys.argv[1:1] = ["--dataset", "flowers102"]
    for option, value in reversed(PROTOCOL_DEFAULTS):
        if not has_option(option):
            sys.argv[1:1] = [option, value]
    cli_main()
