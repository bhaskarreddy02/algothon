"""
Physics-informed feature engineering module.

Implements custom scikit-learn transformers that encode physical relationships
governing tool wear, heat dissipation, power limits, and mechanical overstrain.
"""

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
    TWF_MIN_WEAR,
    TWF_MAX_WEAR,
    CANONICAL_NUMERIC_FEATURES,
    CANONICAL_CATEGORICAL_FEATURES,
)
from src.utils import logger

class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms canonical input sensor variables into domain-specific physics features.
    
    Engineered features:
    1. temp_diff: Process temperature - Air temperature [K]
    2. power_w: Mechanical power (Torque * RPM * 2 * pi / 60) [W]
    3. power_below_limit: Continuous defect metric below 3500 W (PWF risk)
    4. power_above_limit: Continuous defect metric above 9000 W (PWF risk)
    5. power_distance_to_safe_band: Distance to the safe power operational band [3500, 9000]
    6. wear_torque: Tool wear * Torque [min * Nm] (Overstrain indicator)
    7. overstrain_ratio: wear_torque / type-specific threshold
    8. low_speed_low_tempdiff: Soft indicator of HDF region (temp_diff < 8.6 & rpm < 1380)
    9. tool_wear_200_240: Proximity/indicator for TWF critical wear band [200, 240]
    10. speed_torque_ratio: RPM / (Torque + eps)
    """
    
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Ensure column lookup is case/alias tolerant
        col_map = {c.lower(): c for c in X_out.columns}
        
        air_col = col_map.get("air temperature [k]", "Air temperature [K]")
        proc_col = col_map.get("process temperature [k]", "Process temperature [K]")
        speed_col = col_map.get("rotational speed [rpm]", "Rotational speed [rpm]")
        torque_col = col_map.get("torque [nm]", "Torque [Nm]")
        wear_col = col_map.get("tool wear [min]", "Tool wear [min]")
        type_col = col_map.get("type", "Type")
        
        # 1. Temperature difference (Heat Dissipation Failure physics)
        if proc_col in X_out.columns and air_col in X_out.columns:
            X_out["temp_diff"] = X_out[proc_col] - X_out[air_col]
        else:
            X_out["temp_diff"] = 0.0
            
        # 2. Mechanical Power (W)
        if torque_col in X_out.columns and speed_col in X_out.columns:
            X_out["power_w"] = X_out[torque_col] * X_out[speed_col] * (2.0 * np.pi / 60.0)
        else:
            X_out["power_w"] = 0.0
            
        # 3. Power below lower limit (3500 W)
        X_out["power_below_limit"] = np.maximum(0.0, PWF_LOWER_POWER_LIMIT - X_out["power_w"])
        
        # 4. Power above upper limit (9000 W)
        X_out["power_above_limit"] = np.maximum(0.0, X_out["power_w"] - PWF_UPPER_POWER_LIMIT)
        
        # 5. Power distance to safe operational band [3500, 9000]
        X_out["power_distance_to_safe_band"] = X_out["power_below_limit"] + X_out["power_above_limit"]
        
        # 6. Wear * Torque (Overstrain interaction)
        if wear_col in X_out.columns and torque_col in X_out.columns:
            X_out["wear_torque"] = X_out[wear_col] * X_out[torque_col]
        else:
            X_out["wear_torque"] = 0.0
            
        # 7. Overstrain ratio (Threshold: L=11000, M=12000, H=13000)
        if type_col in X_out.columns:
            thresholds = X_out[type_col].map(OSF_OVERSTRAIN_THRESHOLDS).fillna(12000.0)
        else:
            thresholds = 12000.0
        X_out["overstrain_ratio"] = X_out["wear_torque"] / (thresholds + self.eps)
        
        # 8. Low speed low temperature difference (HDF risk region)
        if speed_col in X_out.columns:
            temp_deficit = np.maximum(0.0, HDF_TEMP_DIFF_THRESHOLD - X_out["temp_diff"])
            speed_deficit = np.maximum(0.0, HDF_ROTATIONAL_SPEED_THRESHOLD - X_out[speed_col])
            X_out["low_speed_low_tempdiff"] = temp_deficit * speed_deficit
        else:
            X_out["low_speed_low_tempdiff"] = 0.0
            
        # 9. Tool wear critical band [200, 240] (TWF proximity)
        if wear_col in X_out.columns:
            in_band = ((X_out[wear_col] >= TWF_MIN_WEAR) & (X_out[wear_col] <= TWF_MAX_WEAR)).astype(float)
            dist_to_band = np.minimum(
                np.abs(X_out[wear_col] - TWF_MIN_WEAR),
                np.abs(X_out[wear_col] - TWF_MAX_WEAR)
            )
            dist_to_band = np.where(in_band == 1.0, 0.0, dist_to_band)
            X_out["tool_wear_in_critical_band"] = in_band
            X_out["tool_wear_critical_proximity"] = np.exp(-dist_to_band / 20.0)
        else:
            X_out["tool_wear_in_critical_band"] = 0.0
            X_out["tool_wear_critical_proximity"] = 0.0
            
        # 10. Speed to torque ratio
        if speed_col in X_out.columns and torque_col in X_out.columns:
            X_out["speed_torque_ratio"] = X_out[speed_col] / (X_out[torque_col] + self.eps)
        else:
            X_out["speed_torque_ratio"] = 0.0
            
        return X_out

def build_preprocessing_pipeline(include_physics: bool = True) -> Pipeline:
    """
    Builds an end-to-end sklearn Pipeline with imputation, optional physics engineering,
    one-hot encoding for categorical features, and standard scaling.
    """
    steps = []
    if include_physics:
        steps.append(("physics", PhysicsFeatureEngineer()))
    return Pipeline(steps=steps)
