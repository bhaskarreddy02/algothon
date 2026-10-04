"""
Global configuration module for PredictiveGuard (ALGOTHON'26).

Contains all constants, feature lists, model hyperparameters, random seeds,
anti-leakage rules, and file paths for full pipeline reproducibility.
"""

from pathlib import Path
from typing import Dict, List, Any

# ==============================================================================
# BASE DIRECTORIES & FILE PATHS
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR
DATA_PATH = DATA_DIR / "ai4i2020.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTIONS_DIR = ROOT_DIR / "predictions"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Ensure runtime directories exist
for directory in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Frozen artifact file paths
FROZEN_MODEL_PATH = MODELS_DIR / "final_pipeline.joblib"
METRICS_REPORT_PATH = REPORTS_DIR / "model_comparison.csv"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.md"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"
THRESHOLD_CONFIG_PATH = MODELS_DIR / "threshold_config.json"
TEST_DATA_PATH = DATA_DIR / "holdout_test.csv"
TRAIN_DATA_PATH = DATA_DIR / "train_development.csv"

# ==============================================================================
# REPRODUCIBILITY & SPLIT SETTINGS
# ==============================================================================
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
CV_REPEATS: int = 3

# ==============================================================================
# SCHEMA & ANTI-LEAKAGE DEFINITIONS
# ==============================================================================
TARGET_COLUMN: str = "Machine failure"

# Identifiers to drop unconditionally before feature extraction
DROP_COLUMNS: List[str] = [
    "UDI",
    "Product ID",
]

# Underlying failure mode flags.
# STRICT ANTI-LEAKAGE RULE: NEVER use these as input features for the main model.
# Must only be used for post-hoc analysis, EDA, error analysis, or auxiliary targets.
LEAK_COLUMNS: List[str] = [
    "TWF",  # Tool Wear Failure
    "HDF",  # Heat Dissipation Failure
    "PWF",  # Power Failure
    "OSF",  # Overstrain Failure
    "RNF",  # Random Failure
]

# Canonical raw sensor and metadata features available at inference time
CANONICAL_NUMERIC_FEATURES: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CANONICAL_CATEGORICAL_FEATURES: List[str] = [
    "Type",
]

ALL_INFERENCE_FEATURES: List[str] = CANONICAL_CATEGORICAL_FEATURES + CANONICAL_NUMERIC_FEATURES

# ==============================================================================
# ROBUST SCHEMA NORMALIZATION / ALIASES
# ==============================================================================
COLUMN_ALIASES: Dict[str, str] = {
    # UDI / Identifier variants
    "udi": "UDI",
    "uid": "UDI",
    "id": "UDI",
    "index": "UDI",
    "row_id": "UDI",
    
    # Product ID variants
    "product id": "Product ID",
    "product_id": "Product ID",
    "productid": "Product ID",
    "prod_id": "Product ID",
    
    # Type variants
    "type": "Type",
    "machine_type": "Type",
    "product_type": "Type",
    
    # Air temperature variants
    "air temperature [k]": "Air temperature [K]",
    "air temperature": "Air temperature [K]",
    "air_temperature": "Air temperature [K]",
    "air_temp": "Air temperature [K]",
    "air_temperature_[k]": "Air temperature [K]",
    "air temp [k]": "Air temperature [K]",
    "airtemp": "Air temperature [K]",
    
    # Process temperature variants
    "process temperature [k]": "Process temperature [K]",
    "process temperature": "Process temperature [K]",
    "process_temperature": "Process temperature [K]",
    "process_temp": "Process temperature [K]",
    "process_temperature_[k]": "Process temperature [K]",
    "process temp [k]": "Process temperature [K]",
    "processtemp": "Process temperature [K]",
    
    # Rotational speed variants
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "rotational speed": "Rotational speed [rpm]",
    "rotational_speed": "Rotational speed [rpm]",
    "rotational_speed_[rpm]": "Rotational speed [rpm]",
    "rpm": "Rotational speed [rpm]",
    "speed": "Rotational speed [rpm]",
    
    # Torque variants
    "torque [nm]": "Torque [Nm]",
    "torque": "Torque [Nm]",
    "torque_[nm]": "Torque [Nm]",
    "torque_nm": "Torque [Nm]",
    
    # Tool wear variants
    "tool wear [min]": "Tool wear [min]",
    "tool wear": "Tool wear [min]",
    "tool_wear": "Tool wear [min]",
    "tool_wear_[min]": "Tool wear [min]",
    "wear": "Tool wear [min]",
    "tool_wear_min": "Tool wear [min]",
    
    # Target variants
    "machine failure": "Machine failure",
    "machine_failure": "Machine failure",
    "target": "Machine failure",
    "failure": "Machine failure",
    "fail": "Machine failure",
    
    # Failure mode labels (for auxiliary tasks/EDA only)
    "twf": "TWF",
    "hdf": "HDF",
    "pwf": "PWF",
    "osf": "OSF",
    "rnf": "RNF",
}

# ==============================================================================
# PHYSICS-INFORMED DOMAIN CONSTANTS
# ==============================================================================
HDF_TEMP_DIFF_THRESHOLD: float = 8.6       # Kelvin
HDF_ROTATIONAL_SPEED_THRESHOLD: float = 1380.0 # rpm

PWF_LOWER_POWER_LIMIT: float = 3500.0     # Watts
PWF_UPPER_POWER_LIMIT: float = 9000.0     # Watts

OSF_OVERSTRAIN_THRESHOLDS: Dict[str, float] = {
    "L": 11000.0,
    "M": 12000.0,
    "H": 13000.0,
}

# Conservative industrial engineering fallback for uncharacterized machine variants.
# Defaulting to minimum known structural capacity (Type L = 11000 min*Nm) adheres to
# conservative machinery safety standards (failsafe design) to prevent underestimating failure risk.
OSF_CONSERVATIVE_CAPACITY_FALLBACK: float = 11000.0

TWF_MIN_WEAR: float = 200.0               # minutes
TWF_MAX_WEAR: float = 240.0               # minutes

TYPE_ADDITIONAL_WEAR: Dict[str, float] = {
    "L": 2.0,
    "M": 3.0,
    "H": 5.0,
}

# ==============================================================================
# THRESHOLDING & EVALUATION
# ==============================================================================
DEFAULT_DECISION_THRESHOLD: float = 0.50
OPTIMIZATION_METRIC: str = "f1"           # F1-score optimization on OOF predictions
HIGH_RECALL_MIN_RECALL: float = 0.85      # Predictive maintenance target recall

# ==============================================================================
# MODEL HYPERPARAMETERS (DEFAULTS)
# ==============================================================================
MODEL_SEEDS: Dict[str, int] = {
    "logistic_regression": RANDOM_SEED,
    "random_forest": RANDOM_SEED,
    "extra_trees": RANDOM_SEED,
    "xgboost": RANDOM_SEED,
    "lightgbm": RANDOM_SEED,
}
