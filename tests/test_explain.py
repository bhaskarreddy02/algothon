"""
Unit tests for Model Interpretability & Explainability Module (src/explain.py).
"""

import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import FROZEN_MODEL_PATH
from src.explain import get_pipeline_feature_names, explain_single_record


def test_get_pipeline_feature_names():
    """Verify feature names extracted from the production pipeline match expectations."""
    pipeline = joblib.load(FROZEN_MODEL_PATH)
    names = get_pipeline_feature_names(pipeline)
    assert len(names) == 21, f"Expected 21 pipeline feature names, got {len(names)}"
    
    # Check essential physics features exist
    expected_physics = [
        "overstrain_ratio",
        "low_speed_low_tempdiff",
        "power_w",
        "speed_torque_ratio",
        "temp_diff",
        "wear_torque",
        "Type_H", "Type_L", "Type_M",
    ]
    for feat in expected_physics:
        assert feat in names, f"Feature '{feat}' missing from pipeline feature names"


def test_explain_single_record_nominal():
    """Verify single-record explainability on nominal telemetry."""
    nominal_input = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 20,
    }
    
    explanation = explain_single_record(nominal_input)
    assert "failure_probability" in explanation
    assert "top_risk_drivers" in explanation
    assert "top_mitigating_factors" in explanation
    assert "recommended_maintenance_action" in explanation
    
    assert 0.0 <= explanation["failure_probability"] <= 1.0
    assert explanation["failure_probability"] < 0.20  # Nominal machine should have low failure probability
    assert "NOMINAL" in explanation["recommended_maintenance_action"]


def test_explain_single_record_overstrain():
    """Verify single-record explainability on overstrained tool telemetry."""
    overstrain_input = {
        "Type": "L",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 309.5,
        "Rotational speed [rpm]": 1350,
        "Torque [Nm]": 58.0,
        "Tool wear [min]": 220,
    }
    
    explanation = explain_single_record(overstrain_input)
    assert explanation["failure_probability"] > 0.80  # Overstrained machine should have high failure probability
    assert "CRITICAL" in explanation["recommended_maintenance_action"] or "overstrain" in explanation["dominant_risk_driver"]
    
    driver_names = [d["feature"] for d in explanation["top_risk_drivers"]]
    assert any("overstrain" in d or "wear" in d for d in driver_names)
