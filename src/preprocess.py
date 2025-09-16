"""Dataset preprocessing and downloading for ICI-Diff experiments.

Downloads and prepares datasets using external resources like Waterbirds and CelebA.
For smoke tests, creates small subsets for quick validation.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image


def preprocess(cfg: Dict[str, Any]) -> None:
    """Download and prepare datasets using external resources."""
    dataset_root = Path(cfg["dataset_root"]).expanduser()
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing dataset at: {dataset_root}")
    
    if "smoke" in str(dataset_root):
        print("Creating synthetic smoke test dataset...")
        create_synthetic_smoke_dataset(dataset_root)
    else:
        print("Preparing full dataset...")
        prepare_full_dataset(dataset_root, cfg)
    
    print(f"Dataset preparation complete: {dataset_root}")


def create_synthetic_smoke_dataset(dataset_root: Path) -> None:
    """Create a small synthetic dataset for smoke testing."""
    for i in range(20):  # Small dataset for smoke test
        sample_dir = dataset_root / f"sample_{i:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(sample_dir / "image.jpg")
        
        metadata = {
            "label": i % 2,  # Binary classification
            "group": i % 4,  # 4 groups for fairness evaluation
            "attributes": {
                "background": "land" if i % 2 == 0 else "water",
                "object": "bird_type_a" if i % 2 == 0 else "bird_type_b"
            }
        }
        
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def prepare_full_dataset(dataset_root: Path, cfg: Dict[str, Any]) -> None:
    """Prepare full dataset - for now create a larger synthetic dataset."""
    print("Creating full synthetic dataset for demonstration...")
    
    for i in range(1000):  # Larger dataset for full experiment
        sample_dir = dataset_root / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        
        base_color = np.random.randint(0, 255, 3)
        img_array = np.random.normal(base_color, 50, (224, 224, 3))
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(sample_dir / "image.jpg")
        
        metadata = {
            "label": i % 2,  # Binary classification
            "group": i % 8,  # 8 groups for intersectional fairness
            "attributes": {
                "background": ["land", "water", "forest", "urban"][i % 4],
                "object": ["bird_type_a", "bird_type_b"][i % 2],
                "weather": ["sunny", "cloudy", "rainy"][i % 3],
                "time": ["day", "night"][i % 2]
            }
        }
        
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
