"""
Physics-informed feature engineering module for PredictiveGuard.

Implements domain-driven scikit-learn transformers that encode physical relationships
governing industrial machine failure modes:
1. Heat Dissipation Failure (HDF) - Differential thermal dissipation and convective airflow.
2. Power Failure (PWF) - Motor mechanical power generation bounds [3500 W, 9000 W].
3. Overstrain Failure (OSF) - Product variant-specific mechanical overstrain limits.
4. Tool Wear Failure (TWF) - Tool wear degradation in the critical stochastic band [200, 240] min.
5. Mechanical Operating Ratios - Speed-to-torque dynamics and thermal ratios.
"""

from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import (
    HDF_TEMP_DIFF_THRESHOLD,
    HDF_ROTATIONAL_SPEED_THRESHOLD,
    PWF_LOWER_POWER_LIMIT,
    PWF_UPPER_POWER_LIMIT,
    OSF_OVERSTRAIN_THRESHOLDS,
    OSF_CONSERVATIVE_CAPACITY_FALLBACK,
    TWF_MIN_WEAR,
    TWF_MAX_WEAR,
    CANONICAL_NUMERIC_FEATURES,
    CANONICAL_CATEGORICAL_FEATURES,
    ALL_INFERENCE_FEATURES,
)
from src.data import normalize_column_names
from src.utils import logger


class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms canonical sensory inputs into domain-specific physical interaction features.
    
    Resilient to:
    - Column reordering
    - Presence of extra irrelevant columns
    - Unseen machine type categories
    - Missing sensory values (NaNs)
    - Absence of target or failure mode leakage columns
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit method (stateless transformer)."""
        return self

    def _resolve_column(self, df: pd.DataFrame, canonical_name: str) -> Optional[str]:
        """Resolves column names case-insensitively and through alias normalization."""
        if canonical_name in df.columns:
            return canonical_name
        for col in df.columns:
            if str(col).strip().lower() == canonical_name.lower():
                return col
        return None

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Computes all physics-informed interactions from input data.
        
        Returns:
            pd.DataFrame: Augmented feature dataframe containing raw and engineered features.
        """
        if isinstance(X, np.ndarray):
            # If passed as numpy array, convert to DataFrame using canonical inference features
            if X.shape[1] == len(ALL_INFERENCE_FEATURES):
                df = pd.DataFrame(X, columns=ALL_INFERENCE_FEATURES)
            else:
                # Default numeric features fallback
                df = pd.DataFrame(X, columns=CANONICAL_NUMERIC_FEATURES[:X.shape[1]])
        else:
            df = normalize_column_names(X.copy())

        # Resolve column names
        air_col = self._resolve_column(df, "Air temperature [K]")
        proc_col = self._resolve_column(df, "Process temperature [K]")
        speed_col = self._resolve_column(df, "Rotational speed [rpm]")
        torque_col = self._resolve_column(df, "Torque [Nm]")
        wear_col = self._resolve_column(df, "Tool wear [min]")
        type_col = self._resolve_column(df, "Type")

        out = df.copy()

        # ----------------------------------------------------------------------
        # 1. Temperature Difference (Heat Dissipation Failure Physics)
        # Reason: Convective heat dissipation depends on temperature gradient (T_proc - T_air)
        # ----------------------------------------------------------------------
        if proc_col and air_col:
            out["temp_diff"] = out[proc_col] - out[air_col]
        else:
            out["temp_diff"] = 0.0

        # ----------------------------------------------------------------------
        # 2. Mechanical Power [Watts]
        # Formula: Power = Torque [Nm] * Rotational Speed [rad/s] = Torque * RPM * 2*pi / 60
        # Reason: Motor mechanical energy output governs drive strain and thermal generation
        # ----------------------------------------------------------------------
        if torque_col and speed_col:
            out["power_w"] = out[torque_col] * out[speed_col] * (2.0 * np.pi / 60.0)
        else:
            out["power_w"] = 0.0

        # ----------------------------------------------------------------------
        # 3. Power Below Lower Limit (PWF Low Risk)
        # Reason: Operating below 3500 W indicates mechanical stalling or loss of drive
        # ----------------------------------------------------------------------
        out["power_below_limit"] = np.maximum(0.0, PWF_LOWER_POWER_LIMIT - out["power_w"])

        # ----------------------------------------------------------------------
        # 4. Power Above Upper Limit (PWF High Risk)
        # Reason: Operating above 9000 W indicates severe motor overload / electrical burnout
        # ----------------------------------------------------------------------
        out["power_above_limit"] = np.maximum(0.0, out["power_w"] - PWF_UPPER_POWER_LIMIT)

        # ----------------------------------------------------------------------
        # 5. Continuous Distance to Safe Power Band [3500 W, 9000 W]
        # Reason: Unified risk measure quantifying degree of power excursion
        # ----------------------------------------------------------------------
        out["power_distance_to_safe_band"] = out["power_below_limit"] + out["power_above_limit"]

        # ----------------------------------------------------------------------
        # 6. Wear * Torque (Overstrain Failure Physics)
        # Formula: Overstrain Load = Tool wear [min] * Torque [Nm]
        # Reason: Mechanical strain on the tool tip increases non-linearly with cumulative wear and resistance torque
        # ----------------------------------------------------------------------
        if wear_col and torque_col:
            out["wear_torque"] = out[wear_col] * out[torque_col]
        else:
            out["wear_torque"] = 0.0

        # ----------------------------------------------------------------------
        # 7. Overstrain Ratio (Variant-Specific Threshold Capacity)
        # Thresholds: L = 11,000 min*Nm, M = 12,000 min*Nm, H = 13,000 min*Nm
        # Conservative Fail-Safe: For uncharacterized/unseen types, default to the minimum
        # known structural capacity (Type L = 11,000 min*Nm) per industrial safety standards.
        # ----------------------------------------------------------------------
        if type_col:
            is_known = out[type_col].isin(OSF_OVERSTRAIN_THRESHOLDS.keys()).astype(float)
            thresholds = out[type_col].map(OSF_OVERSTRAIN_THRESHOLDS).fillna(OSF_CONSERVATIVE_CAPACITY_FALLBACK)
        else:
            is_known = pd.Series(0.0, index=out.index)
            thresholds = pd.Series(OSF_CONSERVATIVE_CAPACITY_FALLBACK, index=out.index)
            
        out["overstrain_ratio"] = out["wear_torque"] / (thresholds + self.eps)
        out["is_known_type"] = is_known

        # ----------------------------------------------------------------------
        # 8. Low Speed Low Temp Diff Interaction (HDF Region Indicator)
        # Physics: HDF occurs when temp_diff < 8.6 K AND rotational_speed < 1380 rpm
        # Reason: Combined deficit captures severity of heat entrapment under insufficient convective cooling
        # ----------------------------------------------------------------------
        if speed_col:
            temp_deficit = np.maximum(0.0, HDF_TEMP_DIFF_THRESHOLD - out["temp_diff"])
            speed_deficit = np.maximum(0.0, HDF_ROTATIONAL_SPEED_THRESHOLD - out[speed_col])
            out["low_speed_low_tempdiff"] = temp_deficit * speed_deficit
        else:
            out["low_speed_low_tempdiff"] = 0.0

        # ----------------------------------------------------------------------
        # 9. Tool Wear Critical Band [200, 240] min (TWF Region Proximity)
        # Physics: Tool replacement window occurs at a random wear point between 200 and 240 min
        # Reason: Provides both a strict binary indicator and a smooth exponential proximity metric
        # ----------------------------------------------------------------------
        if wear_col:
            in_band = ((out[wear_col] >= TWF_MIN_WEAR) & (out[wear_col] <= TWF_MAX_WEAR)).astype(float)
            dist_to_band = np.where(
                out[wear_col] < TWF_MIN_WEAR,
                TWF_MIN_WEAR - out[wear_col],
                np.where(out[wear_col] > TWF_MAX_WEAR, out[wear_col] - TWF_MAX_WEAR, 0.0)
            )
            out["tool_wear_in_critical_band"] = in_band
            out["tool_wear_critical_proximity"] = np.exp(-dist_to_band / 20.0)
        else:
            out["tool_wear_in_critical_band"] = 0.0
            out["tool_wear_critical_proximity"] = 0.0

        # ----------------------------------------------------------------------
        # 10. Speed-to-Torque Ratio
        # Formula: RPM / (Torque + eps)
        # Reason: Inverse relationship reflecting mechanical gear/load impedance
        # ----------------------------------------------------------------------
        if speed_col and torque_col:
            out["speed_torque_ratio"] = out[speed_col] / (out[torque_col] + self.eps)
        else:
            out["speed_torque_ratio"] = 0.0

        # ----------------------------------------------------------------------
        # 11. Thermodynamic Temperature Ratio
        # Formula: Process Temperature / Air Temperature
        # Reason: Normalized relative heat accumulation independent of ambient seasonal shifts
        # ----------------------------------------------------------------------
        if proc_col and air_col:
            out["temp_ratio"] = out[proc_col] / (out[air_col] + self.eps)
        else:
            out["temp_ratio"] = 1.0

        self.feature_names_ = list(out.columns)
        return out


def get_feature_lists(include_physics: bool = True) -> Dict[str, List[str]]:
    """
    Returns categorized lists of numeric and categorical feature names.
    """
    categorical = ["Type"]
    numeric = list(CANONICAL_NUMERIC_FEATURES)
    
    if include_physics:
        physics_features = [
            "temp_diff",
            "power_w",
            "power_below_limit",
            "power_above_limit",
            "power_distance_to_safe_band",
            "wear_torque",
            "overstrain_ratio",
            "low_speed_low_tempdiff",
            "tool_wear_in_critical_band",
            "tool_wear_critical_proximity",
            "speed_torque_ratio",
            "temp_ratio",
            "is_known_type",
        ]
        numeric.extend(physics_features)
        
    return {
        "categorical": categorical,
        "numeric": numeric,
        "all": categorical + numeric,
    }


def build_preprocessing_pipeline(
    include_physics: bool = True,
    scale_numeric: bool = False,
) -> Pipeline:
    """
    Constructs an end-to-end, leak-free scikit-learn preprocessing pipeline.
    
    Pipeline Steps:
    1. PhysicsFeatureEngineer (optional): Derives physics features from raw signals.
    2. ColumnTransformer:
       - Numeric Pipeline: SimpleImputer(strategy='median') + optional StandardScaler().
       - Categorical Pipeline: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore').
       
    Ensures safe handling of unseen categories, reordered columns, and missing values.
    """
    feature_info = get_feature_lists(include_physics=include_physics)
    num_cols = feature_info["numeric"]
    cat_cols = feature_info["categorical"]

    # Numeric sub-pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical sub-pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",  # Safely drops any extra/irrelevant columns
    )

    steps = []
    if include_physics:
        steps.append(("physics", PhysicsFeatureEngineer()))
    steps.append(("preprocessor", preprocessor))

    return Pipeline(steps=steps)
