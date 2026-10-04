"""
Utility functions for logging, reproducibility seeding, and persistence.
"""

import os
import random
import logging
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import joblib

def setup_logger(name: str = "PredictiveGuard", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def seed_everything(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, and environment for deterministic behavior."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    logger.info("Global random seed set to %d across random, numpy, os", seed)

def save_artifact(obj: Any, path: Path) -> None:
    """Saves a python object using joblib to the designated path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact to %s", path)

def load_artifact(path: Path) -> Any:
    """Loads a python object using joblib from the designated path."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    obj = joblib.load(path)
    logger.info("Loaded artifact from %s", path)
    return obj

def save_json(data: Dict[str, Any], path: Path) -> None:
    """Saves dictionary data to a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info("Saved JSON to %s", path)

def load_json(path: Path) -> Dict[str, Any]:
    """Loads dictionary data from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded JSON from %s", path)
    return data
