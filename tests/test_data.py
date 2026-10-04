"""
Unit tests for data loading, schema validation, alias mapping, and anti-leakage isolation.
"""

import pytest
import pandas as pd
import numpy as np
from src.data import (
    normalize_column_names,
    validate_schema,
    remove_leakage_columns,
    remove_identifiers,
    handle_missing_values,
    load_data,
)
from src.config import TARGET_COLUMN, LEAK_COLUMNS, DROP_COLUMNS, ALL_INFERENCE_FEATURES, DATA_PATH

def test_alias_normalization_comprehensive():
    """Test alias normalization across diverse column variations."""
    df_messy = pd.DataFrame(columns=[
        "uid", "product_id", "machine_type",
        "air_temperature", "process_temp", "rpm", "torque_nm", "tool_wear",
        "machine_failure", "twf", "hdf", "pwf", "osf", "rnf"
    ])
    df_norm = normalize_column_names(df_messy)
    
    assert "UDI" in df_norm.columns
    assert "Product ID" in df_norm.columns
    assert "Type" in df_norm.columns
    assert "Air temperature [K]" in df_norm.columns
    assert "Process temperature [K]" in df_norm.columns
    assert "Rotational speed [rpm]" in df_norm.columns
    assert "Torque [Nm]" in df_norm.columns
    assert "Tool wear [min]" in df_norm.columns
    assert "Machine failure" in df_norm.columns
    for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert m in df_norm.columns

def test_validate_schema_success():
    """Test validate_schema with valid dataframe."""
    df = pd.DataFrame(columns=ALL_INFERENCE_FEATURES + [TARGET_COLUMN])
    validate_schema(df, require_target=True)
    validate_schema(df, require_target=False)

def test_validate_schema_missing_column():
    """Test validate_schema throws error when essential column missing."""
    df = pd.DataFrame(columns=["Type", "Air temperature [K]"])
    with pytest.raises(ValueError, match="Missing required inference features"):
        validate_schema(df)

def test_validate_schema_missing_target():
    """Test validate_schema throws error when required target is missing."""
    df = pd.DataFrame(columns=ALL_INFERENCE_FEATURES)
    with pytest.raises(ValueError, match="Required target column"):
        validate_schema(df, require_target=True)

def test_remove_leakage_columns():
    """Test that all leakage failure modes are strictly removed."""
    df = pd.DataFrame({
        "Type": ["L"], "Air temperature [K]": [300.0],
        "TWF": [1], "HDF": [0], "PWF": [0], "OSF": [0], "RNF": [0]
    })
    df_clean, df_leaks = remove_leakage_columns(df)
    for leak in LEAK_COLUMNS:
        assert leak not in df_clean.columns
    assert df_leaks is not None
    assert set(df_leaks.columns) == set(LEAK_COLUMNS)

def test_remove_identifiers():
    """Test that identifiers are removed cleanly."""
    df = pd.DataFrame({
        "UDI": [101], "Product ID": ["L47180"], "Type": ["L"]
    })
    df_clean, df_ids = remove_identifiers(df)
    assert "UDI" not in df_clean.columns
    assert "Product ID" not in df_clean.columns
    assert "Type" in df_clean.columns
    assert df_ids is not None

def test_handle_missing_values_median():
    """Test missing value median imputation."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["A", "B", "A", None]
    })
    df_imp = handle_missing_values(df, strategy="impute_median")
    assert df_imp["num"].isnull().sum() == 0
    assert df_imp["num"].iloc[2] == 2.0  # Median of [1, 2, 4] is 2.0
    assert df_imp["cat"].isnull().sum() == 0
    assert df_imp["cat"].iloc[3] == "A"  # Mode of ["A", "B", "A"] is "A"

def test_load_data_full_pipeline():
    """Test loading actual dataset and verify anti-leakage and schema integrity."""
    X, y, leakage_df = load_data(DATA_PATH, require_target=True, drop_leakage=True)
    assert len(X) == 10000
    assert len(y) == 10000
    assert leakage_df is not None
    assert TARGET_COLUMN not in X.columns
    for leak in LEAK_COLUMNS:
        assert leak not in X.columns
    for id_col in DROP_COLUMNS:
        assert id_col not in X.columns
    for feat in ALL_INFERENCE_FEATURES:
        assert feat in X.columns
