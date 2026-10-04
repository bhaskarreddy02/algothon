"""
Data loading, cleaning, schema validation, and alias normalization module.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from src.config import (
    COLUMN_ALIASES,
    TARGET_COLUMN,
    DROP_COLUMNS,
    LEAK_COLUMNS,
    CANONICAL_NUMERIC_FEATURES,
    CANONICAL_CATEGORICAL_FEATURES,
    ALL_INFERENCE_FEATURES,
)
from src.utils import logger

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names by stripping whitespace, converting to lower case for mapping,
    and renaming to canonical project names using COLUMN_ALIASES.
    """
    renamed = {}
    for col in df.columns:
        clean_name = str(col).strip().lower()
        if clean_name in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[clean_name]
    return df.rename(columns=renamed)

def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """
    Validates that required inference features exist in the dataframe.
    Raises ValueError with descriptive message if mandatory columns are absent.
    """
    missing = [c for c in ALL_INFERENCE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing required inference columns: {missing}")
    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Schema validation failed. Missing required target column: '{TARGET_COLUMN}'")

def load_data(
    file_path: Path,
    require_target: bool = True,
    drop_leakage: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Loads raw CSV data, normalizes columns, validates schema, and separates features from target.
    
    Returns:
        X: Feature dataframe (without target, without IDs, without leakage columns if drop_leakage=True)
        y: Target series (Machine failure) if present, else None
        leakage_df: Underlying failure mode dataframe (TWF, HDF, PWF, OSF, RNF) if present, else None
    """
    logger.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path)
    df = normalize_column_names(df)
    validate_schema(df, require_target=require_target)
    
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    
    # Extract leakage dataframe if present (for EDA and auxiliary targets only)
    present_leaks = [c for c in LEAK_COLUMNS if c in df.columns]
    leakage_df = df[present_leaks].copy() if present_leaks else None
    
    # Build clean inference feature matrix X
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if drop_leakage:
        cols_to_drop.extend([c for c in LEAK_COLUMNS if c in df.columns])
    if TARGET_COLUMN in df.columns:
        cols_to_drop.append(TARGET_COLUMN)
        
    X = df.drop(columns=cols_to_drop, errors="ignore")
    return X, y, leakage_df
