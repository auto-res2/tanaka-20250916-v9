"""Main orchestration script for ICI-Diff experiments.

Run either a quick smoke test or the full experiment:

uv run python -m src.main --smoke-test
uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import json
from typing import Dict, Any

import yaml

# local, *relative* imports mandated by the spec
from .preprocess import preprocess
from .train import Trainer
from .evaluate import evaluate


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    # inject runtime derived fields
    cfg.setdefault("output_root", ".research/iteration2")
    cfg.setdefault("seed", 0)
    return cfg


def _phase(name: str, cfg: Dict[str, Any]) -> None:
    print(f"===== {name.upper()} PHASE =====")
    preprocess(cfg)
    trainer = Trainer(cfg)
    trainer.fit()
    evaluate(cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run ICI-Diff experiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="run quick sanity run")
    group.add_argument("--full-experiment", action="store_true", help="run full experiment")
    args = parser.parse_args(argv)

    root_cfg_dir = Path(__file__).parent.parent / "config"
    smoke_cfg = root_cfg_dir / "smoke_test.yaml"
    full_cfg = root_cfg_dir / "full_experiment.yaml"

    if args.smoke_test:
        cfg = _load_cfg(smoke_cfg)
        _phase("smoke", cfg)
    elif args.full_experiment:
        # mandatory two-phase: first smoke, then full
        cfg_smoke = _load_cfg(smoke_cfg)
        _phase("smoke", cfg_smoke)
        cfg_full = _load_cfg(full_cfg)
        _phase("full", cfg_full)


if __name__ == "__main__":
    main()
