"""
Initial smoke tests verifying module imports, configuration integrity, and schema aliases.
"""

import pytest
import pandas as pd
from src.config import (
    RANDOM_SEED,
    TARGET_COLUMN,
    DROP_COLUMNS,
    LEAK_COLUMNS,
    CANONICAL_NUMERIC_FEATURES,
    CANONICAL_CATEGORICAL_FEATURES,
    COLUMN_ALIASES,
    DATA_PATH,
)
from src.data import normalize_column_names, validate_schema
from src.utils import seed_everything

def test_config_constants():
    """Verify core anti-leakage and configuration constants."""
    assert RANDOM_SEED == 42
    assert TARGET_COLUMN == "Machine failure"
    assert "UDI" in DROP_COLUMNS
    assert "Product ID" in DROP_COLUMNS
    assert set(LEAK_COLUMNS) == {"TWF", "HDF", "PWF", "OSF", "RNF"}
    assert len(CANONICAL_NUMERIC_FEATURES) == 5
    assert "Type" in CANONICAL_CATEGORICAL_FEATURES

def test_data_path_exists():
    """Verify that dataset exists in the workspace root."""
    assert DATA_PATH.exists(), f"Expected dataset at {DATA_PATH}"

def test_column_alias_normalization():
    """Verify alias mapping correctly handles common case/formatting variations."""
    df_raw = pd.DataFrame(columns=[
        "udi", "Product_ID", "type", "air_temperature_[k]",
        "process_temp", "rpm", "TORQUE", "wear", "machine_failure", "TWF"
    ])
    df_norm = normalize_column_names(df_raw)
    
    assert "UDI" in df_norm.columns
    assert "Product ID" in df_norm.columns
    assert "Type" in df_norm.columns
    assert "Air temperature [K]" in df_norm.columns
    assert "Process temperature [K]" in df_norm.columns
    assert "Rotational speed [rpm]" in df_norm.columns
    assert "Torque [Nm]" in df_norm.columns
    assert "Tool wear [min]" in df_norm.columns
    assert "Machine failure" in df_norm.columns
    assert "TWF" in df_norm.columns

def test_schema_validation_failure():
    """Verify validate_schema raises ValueError when mandatory columns are missing."""
    df_incomplete = pd.DataFrame({"Type": ["L"], "Air temperature [K]": [300.0]})
    with pytest.raises(ValueError, match="Missing required inference columns"):
        validate_schema(df_incomplete)

def test_seed_everything():
    """Verify random seeding runs without error."""
    seed_everything(42)
