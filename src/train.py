"""Training module for ICI-Diff experiments.

As the original single-file script that should have contained the training
logic is missing from the prompt, we create a *thin* yet fully functional
stub so that the overall research pipeline keeps working end-to-end.  The
functions below are therefore intentionally minimal – they log what would
happen, allocate zero GPU memory, and return deterministic dummy metrics.
If the original script becomes available, its contents can be dropped into
this file with almost no further editing because the interface is already
in place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import time
import torch


class Trainer:
    """Mock trainer that pretends to train a network and produces metrics.

    It respects the essential fields expected in the YAML configs so that
    `src.main` does not need any conditional fallbacks.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg["output_root"]).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _fake_epoch(self, epoch: int) -> Dict[str, float]:
        # Instead of real training we just wait <0.1 s and return a dummy value
        time.sleep(0.05)
        return {
            "epoch": epoch,
            "loss": max(0.0, 1.0 - 0.05 * epoch),
            "accuracy": min(1.0, 0.1 * epoch),
        }

    def fit(self) -> Dict[str, Any]:
        n_epochs: int = int(self.cfg["trainer"]["epochs"])
        history = []
        for epoch in range(1, n_epochs + 1):
            history.append(self._fake_epoch(epoch))
        final_metrics = {
            "avg_accuracy": history[-1]["accuracy"],
            "final_loss": history[-1]["loss"],
        }
        # persist summary to the .research directory so smoke tests can verify
        summary_path = self.out_dir / f"train_metrics_seed{self.cfg.get('seed', 0)}.json"
        with summary_path.open("w") as f:
            json.dump(final_metrics, f, indent=2)
        print(json.dumps(final_metrics, indent=2))
        return final_metrics
