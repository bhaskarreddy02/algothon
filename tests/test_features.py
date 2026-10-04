"""
Comprehensive unit tests for physics-informed feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import PhysicsFeatureEngineer, build_preprocessing_pipeline, get_feature_lists
from src.config import LEAK_COLUMNS, TARGET_COLUMN

@pytest.fixture
def sample_telemetry_df():
    """Provides a deterministic sample telemetry DataFrame."""
    return pd.DataFrame({
        "Type": ["L", "M", "H", "L"],
        "Air temperature [K]": [298.1, 298.2, 298.3, 300.0],
        "Process temperature [K]": [308.6, 308.7, 308.7, 307.0],  # Last row: diff = 7.0 (< 8.6)
        "Rotational speed [rpm]": [1500, 1400, 1600, 1200],       # Last row: speed = 1200 (< 1380)
        "Torque [Nm]": [40.0, 50.0, 30.0, 70.0],
        "Tool wear [min]": [10, 150, 220, 200],                    # Row 2 & 3 in TWF band
    })

def test_feature_engineering_presence(sample_telemetry_df):
    """Verify that all required physics features are generated and non-null."""
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(sample_telemetry_df)
    
    expected_physics = [
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
    for feat in expected_physics:
        assert feat in out.columns, f"Feature '{feat}' missing from transformer output"
        assert not out[feat].isnull().any(), f"Feature '{feat}' contains unexpected NaNs"

def test_power_formula(sample_telemetry_df):
    """Verify mechanical power calculation matches Torque * RPM * 2*pi / 60."""
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(sample_telemetry_df)
    
    expected_power = 40.0 * 1500.0 * (2.0 * np.pi / 60.0)
    assert np.isclose(out["power_w"].iloc[0], expected_power, atol=1e-3)
    
    # Row 0: power is ~6283.18 W (within [3500, 9000]) -> power_distance_to_safe_band == 0
    assert out["power_distance_to_safe_band"].iloc[0] == 0.0

def test_temperature_difference(sample_telemetry_df):
    """Verify temp_diff = Process temp - Air temp."""
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(sample_telemetry_df)
    
    assert np.isclose(out["temp_diff"].iloc[0], 308.6 - 298.1, atol=1e-4)
    assert np.isclose(out["temp_diff"].iloc[3], 307.0 - 300.0, atol=1e-4)

def test_overstrain_ratio(sample_telemetry_df):
    """Verify overstrain ratio uses L=11000, M=12000, H=13000 thresholds."""
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(sample_telemetry_df)
    
    # Row 0: Type L, wear=10, torque=40 -> wear_torque=400 -> ratio = 400 / 11000
    expected_ratio_l = (10.0 * 40.0) / 11000.0
    assert np.isclose(out["overstrain_ratio"].iloc[0], expected_ratio_l, atol=1e-4)
    
    # Row 1: Type M, wear=150, torque=50 -> wear_torque=7500 -> ratio = 7500 / 12000
    expected_ratio_m = (150.0 * 50.0) / 12000.0
    assert np.isclose(out["overstrain_ratio"].iloc[1], expected_ratio_m, atol=1e-4)
    
    # Row 2: Type H, wear=220, torque=30 -> wear_torque=6600 -> ratio = 6600 / 13000
    expected_ratio_h = (220.0 * 30.0) / 13000.0
    assert np.isclose(out["overstrain_ratio"].iloc[2], expected_ratio_h, atol=1e-4)

def test_hdf_interaction(sample_telemetry_df):
    """Verify low_speed_low_tempdiff activates only when both conditions are violated."""
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(sample_telemetry_df)
    
    # Rows 0-2 do not satisfy HDF condition (speed > 1380 or temp_diff >= 8.6)
    assert out["low_speed_low_tempdiff"].iloc[0] == 0.0
    assert out["low_speed_low_tempdiff"].iloc[1] == 0.0
    
    # Row 3: temp_diff = 7.0 (< 8.6), speed = 1200 (< 1380)
    # Deficit: (8.6 - 7.0) * (1380 - 1200) = 1.6 * 180 = 288.0
    assert np.isclose(out["low_speed_low_tempdiff"].iloc[3], 1.6 * 180.0, atol=1e-3)

def test_tool_wear_critical_band(sample_telemetry_df):
    """Verify tool wear critical band [200, 240] detection and proximity."""
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(sample_telemetry_df)
    
    # Row 0: wear = 10 -> in_band = 0
    assert out["tool_wear_in_critical_band"].iloc[0] == 0.0
    # Row 2: wear = 220 -> in_band = 1, proximity = 1.0
    assert out["tool_wear_in_critical_band"].iloc[2] == 1.0
    assert np.isclose(out["tool_wear_critical_proximity"].iloc[2], 1.0)

def test_column_reordering_resilience(sample_telemetry_df):
    """Verify transformer produces equivalent results regardless of column permutation."""
    engineer = PhysicsFeatureEngineer()
    cols_orig = list(sample_telemetry_df.columns)
    cols_permuted = cols_orig[::-1]
    
    out_orig = engineer.fit_transform(sample_telemetry_df)
    out_perm = engineer.fit_transform(sample_telemetry_df[cols_permuted])
    
    for col in ["temp_diff", "power_w", "wear_torque", "overstrain_ratio"]:
        np.testing.assert_allclose(out_orig[col].values, out_perm[col].values, rtol=1e-5)

def test_extra_columns_resilience(sample_telemetry_df):
    """Verify extra irrelevant columns do not cause failures or leak into features."""
    sample_with_extra = sample_telemetry_df.copy()
    sample_with_extra["irrelevant_sensor_x"] = [99.0, 98.0, 97.0, 96.0]
    sample_with_extra["operator_notes"] = ["Shift A", "Shift B", "Shift A", "Shift C"]
    
    pipe = build_preprocessing_pipeline(include_physics=True)
    transformed = pipe.fit_transform(sample_with_extra)
    assert transformed.shape[0] == 4

def test_target_absent(sample_telemetry_df):
    """Verify pipeline works when target column is completely absent."""
    assert TARGET_COLUMN not in sample_telemetry_df.columns
    pipe = build_preprocessing_pipeline(include_physics=True)
    transformed = pipe.fit_transform(sample_telemetry_df)
    assert transformed.shape[0] == 4

def test_leakage_columns_not_used(sample_telemetry_df):
    """Verify that none of the leakage failure mode columns are required or used."""
    for leak in LEAK_COLUMNS:
        assert leak not in sample_telemetry_df.columns
    pipe = build_preprocessing_pipeline(include_physics=True)
    transformed = pipe.fit_transform(sample_telemetry_df)
    assert transformed.shape[0] == 4

def test_unseen_type_handling(sample_telemetry_df):
    """Verify unseen Type categories (e.g. 'Z') safely use conservative capacity fallback (11000.0)."""
    df_unseen = sample_telemetry_df.copy()
    df_unseen.loc[0, "Type"] = "Z"  # Unseen product variant
    
    engineer = PhysicsFeatureEngineer()
    out = engineer.fit_transform(df_unseen)
    
    # Assert is_known_type is 0 for unseen type and 1 for known types
    assert out["is_known_type"].iloc[0] == 0.0
    assert out["is_known_type"].iloc[1] == 1.0
    
    # Assert conservative fallback (11000.0) is used for unseen variant Z
    # Row 0: wear = 10, torque = 40 -> wear_torque = 400 -> ratio = 400 / 11000
    expected_ratio = 400.0 / 11000.0
    assert np.isclose(out["overstrain_ratio"].iloc[0], expected_ratio, atol=1e-4)
    
    # Also verify full pipeline transforms without error
    pipe = build_preprocessing_pipeline(include_physics=True)
    pipe.fit(sample_telemetry_df)
    transformed = pipe.transform(df_unseen)
    assert transformed.shape[0] == 4
    assert not np.isnan(transformed).any()

def test_missing_values_handling(sample_telemetry_df):
    """Verify that NaNs in sensors are imputed gracefully."""
    df_missing = sample_telemetry_df.copy()
    df_missing.loc[1, "Torque [Nm]"] = np.nan
    df_missing.loc[2, "Type"] = None
    
    pipe = build_preprocessing_pipeline(include_physics=True)
    transformed = pipe.fit_transform(df_missing)
    assert not np.isnan(transformed).any()
