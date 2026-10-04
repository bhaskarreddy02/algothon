"""
Data loading, cleaning, schema validation, alias normalization, and anti-leakage isolation module.

Provides resilient data ingestion for training and batch inference, guaranteeing that:
1. Identifier columns (UDI, Product ID) are purged.
2. Failure mode flags (TWF, HDF, PWF, OSF, RNF) are strictly prevented from entering the inference matrix.
3. Schema variations (casing, brackets, underscores, common aliases) are seamlessly canonicalized.
4. Missing values and arbitrary column orderings are handled gracefully.
"""

from pathlib import Path
from typing import Tuple, Optional, List, Union
import re
import pandas as pd
import numpy as np

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    LEAK_COLUMNS,
    CANONICAL_NUMERIC_FEATURES,
    CANONICAL_CATEGORICAL_FEATURES,
    ALL_INFERENCE_FEATURES,
    COLUMN_ALIASES,
)
from src.utils import logger

def _clean_column_string(col_name: str) -> str:
    """Standardizes a column name string for robust alias matching."""
    return str(col_name).strip().lower()

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names by converting to lower case, stripping whitespace/underscores,
    and renaming to canonical project names using COLUMN_ALIASES.
    """
    renamed = {}
    for col in df.columns:
        cleaned = _clean_column_string(col)
        cleaned_spaces = re.sub(r'[\s_]+', ' ', cleaned)
        cleaned_underscores = re.sub(r'[\s_]+', '_', cleaned)
        
        # Direct lookup across variations
        if cleaned in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[cleaned]
            continue
        elif cleaned_spaces in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[cleaned_spaces]
            continue
        elif cleaned_underscores in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[cleaned_underscores]
            continue
        
        # Soft regex fallback matching
        if "type" in cleaned_spaces:
            renamed[col] = "Type"
        elif "air" in cleaned_spaces and "temp" in cleaned_spaces:
            renamed[col] = "Air temperature [K]"
        elif "process" in cleaned_spaces and "temp" in cleaned_spaces:
            renamed[col] = "Process temperature [K]"
        elif "speed" in cleaned_spaces or "rpm" in cleaned_spaces or "rotat" in cleaned_spaces:
            renamed[col] = "Rotational speed [rpm]"
        elif "torque" in cleaned_spaces:
            renamed[col] = "Torque [Nm]"
        elif "wear" in cleaned_spaces:
            renamed[col] = "Tool wear [min]"
        elif "machine" in cleaned_spaces and "fail" in cleaned_spaces:
            renamed[col] = "Machine failure"
        elif cleaned in ["twf", "hdf", "pwf", "osf", "rnf"]:
            renamed[col] = cleaned.upper()
            
    df_out = df.rename(columns=renamed)
    return df_out

def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """
    Validates that essential inference features exist in the dataframe.
    Raises ValueError with descriptive diagnosis if mandatory features are missing.
    """
    missing_features = [col for col in ALL_INFERENCE_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Schema validation failed. Missing required inference features: {missing_features}. "
            f"Available columns: {list(df.columns)}"
        )
    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Schema validation failed. Required target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

def remove_leakage_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Strict anti-leakage isolation.
    Extracts and strips TWF, HDF, PWF, OSF, and RNF from the main dataframe.
    
    Returns:
        df_clean: Dataframe with all leakage columns removed.
        df_leaks: Extracted leakage dataframe if any leakage columns were present, else None.
    """
    present_leaks = [col for col in LEAK_COLUMNS if col in df.columns]
    df_leaks = df[present_leaks].copy() if present_leaks else None
    df_clean = df.drop(columns=present_leaks, errors="ignore")
    return df_clean, df_leaks

def remove_identifiers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Purges identifier columns (UDI, Product ID) to prevent spurious correlations.
    
    Returns:
        df_clean: Dataframe stripped of IDs.
        df_ids: Extracted identifier dataframe if present, else None.
    """
    present_ids = [col for col in DROP_COLUMNS if col in df.columns]
    df_ids = df[present_ids].copy() if present_ids else None
    df_clean = df.drop(columns=present_ids, errors="ignore")
    return df_clean, df_ids

def handle_missing_values(df: pd.DataFrame, strategy: str = "keep") -> pd.DataFrame:
    """
    Handles missing values in sensor and categorical columns.
    
    Args:
        df: Input dataframe
        strategy: 
            'keep': Leaves NaNs for the pipeline SimpleImputer to process inside CV.
            'impute_median': Imputes numeric columns with column median and categoricals with mode.
            'drop': Drops rows containing missing values.
    """
    if strategy == "keep":
        return df.copy()
    elif strategy == "drop":
        return df.dropna().copy()
    elif strategy == "impute_median":
        df_imputed = df.copy()
        for col in df_imputed.select_dtypes(include=[np.number]).columns:
            if df_imputed[col].isnull().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        for col in df_imputed.select_dtypes(include=['object', 'category', 'string']).columns:
            if df_imputed[col].isnull().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
        return df_imputed
    else:
        raise ValueError(f"Unknown missing value strategy: '{strategy}'")

def load_data(
    file_path: Union[str, Path] = DATA_PATH,
    require_target: bool = True,
    drop_leakage: bool = True,
    drop_ids: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Loads raw CSV data, canonicalizes column aliases, validates schema,
    purges identifiers, isolates leakage failure modes, and separates features from target.
    
    Args:
        file_path: Path to dataset CSV.
        require_target: Whether to mandate target column presence (True for train, False for predict).
        drop_leakage: Whether to remove TWF, HDF, PWF, OSF, RNF.
        drop_ids: Whether to remove UDI, Product ID.
        
    Returns:
        X: Clean feature DataFrame ready for the scikit-learn Pipeline.
        y: Target Series (Machine failure) if present, else None.
        leakage_df: Failure mode DataFrame (TWF, HDF, PWF, OSF, RNF) if present, else None.
    """
    file_path = Path(file_path)
    logger.info("Loading dataset from %s (require_target=%s, drop_leakage=%s)", file_path, require_target, drop_leakage)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    df = normalize_column_names(df)
    validate_schema(df, require_target=require_target)
    
    # Extract target if present
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    
    # Extract leakage dataframe (for diagnostic / mode-aware training only)
    leakage_df = None
    if drop_leakage:
        df, leakage_df = remove_leakage_columns(df)
        
    # Remove identifiers
    if drop_ids:
        df, _ = remove_identifiers(df)
        
    # Drop target from feature matrix X
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
        
    return df, y, leakage_df
