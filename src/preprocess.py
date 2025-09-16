"""Pre-processing stub.

The real ICI-Diff pipeline uses sophisticated data discovery and diffusion
editing.  Until that code is provided, we merely verify that the configured
`dataset_root` exists – if it does not, the script raises a clear error so
that users immediately know which prerequisite is missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def preprocess(cfg: Dict[str, Any]) -> None:
    data_root = Path(cfg["dataset_root"]).expanduser()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Configured dataset_root={data_root} does not exist. "
            "Please download the dataset before running the experiment."
        )
    # nothing else to do in the stub
