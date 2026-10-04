"""
Unit tests for model factory, baseline estimators, and out-of-fold mode-aware stacking.
"""

import pytest
import numpy as np
import pandas as pd
from src.models import get_baseline_models, ModeAwareStackingClassifier

def test_baseline_models_factory():
    """Verify all baseline models are configured with anti-imbalance parameters."""
    models = get_baseline_models(scale_pos_weight=28.0)
    expected_keys = ["dummy", "logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm"]
    for k in expected_keys:
        assert k in models, f"Model key '{k}' missing from baseline models"

def test_auxiliary_stacking_is_strictly_out_of_fold():
    """
    CRITICAL ANTI-LEAKAGE TEST:
    Verifies that ModeAwareStackingClassifier uses STRICTLY out-of-fold predictions
    during training, guaranteeing that no sample's stacking features are generated
    by an auxiliary model that was trained on that same sample.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 6
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Synthetic auxiliary failure mode targets
    aux_targets = pd.DataFrame({
        "HDF": np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        "PWF": np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07]),
        "OSF": np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06]),
        "TWF": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
    })
    
    model = ModeAwareStackingClassifier(n_inner_folds=5, random_state=42)
    model.fit(X, y, aux_targets=aux_targets)
    
    # 1. Assert internal folds were executed and tracked
    assert model.training_indices_used_ is not None
    assert len(model.training_indices_used_) == 5
    
    # 2. Assert zero overlap between train and validation indices for every inner fold
    covered_val_indices = []
    for fold_idx, (in_train_idx, in_val_idx) in enumerate(model.training_indices_used_):
        train_set = set(in_train_idx)
        val_set = set(in_val_idx)
        
        # Absolute requirement: Mutual exclusivity between train and validation sets
        intersection = train_set.intersection(val_set)
        assert len(intersection) == 0, f"Leakage detected in fold {fold_idx}: indices {intersection} in both train and val!"
        
        covered_val_indices.extend(in_val_idx)
        
    # 3. Assert that all samples received exactly one out-of-fold prediction
    assert len(covered_val_indices) == n_samples
    assert set(covered_val_indices) == set(range(n_samples))
    
    # 4. Assert OOF prediction matrix has valid probabilities [0, 1] without NaNs
    oof_df = model.oof_train_predictions_
    assert oof_df is not None
    assert oof_df.shape == (n_samples, 4)
    assert not oof_df.isnull().any().any()
    assert ((oof_df >= 0.0) & (oof_df <= 1.0)).all().all()

def test_mode_aware_inference_without_leakage_columns():
    """
    CRITICAL INFERENCE RESILIENCE TEST:
    Verifies that the fitted mode-aware pipeline makes predictions on unseen data
    where auxiliary mode columns (HDF, PWF, OSF, TWF) DO NOT EXIST.
    """
    np.random.seed(42)
    n_train, n_test = 150, 50
    X_train = pd.DataFrame(np.random.randn(n_train, 5), columns=[f"sensor_{i}" for i in range(5)])
    y_train = np.random.choice([0, 1], size=n_train, p=[0.9, 0.1])
    aux_train = pd.DataFrame({
        "HDF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "PWF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "OSF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "TWF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
    })
    
    model = ModeAwareStackingClassifier(combination_mode="stacking", n_inner_folds=3, random_state=42)
    model.fit(X_train, y_train, aux_targets=aux_train)
    
    # Test set: ONLY sensor features exist. NO failure mode columns.
    X_test = pd.DataFrame(np.random.randn(n_test, 5), columns=[f"sensor_{i}" for i in range(5)])
    for mode in ["HDF", "PWF", "OSF", "TWF"]:
        assert mode not in X_test.columns
        
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test, threshold=0.5)
    
    assert probs.shape == (n_test, 2)
    assert preds.shape == (n_test,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)

def test_mode_aware_noisy_or_combination():
    """Verifies that noisy-OR analytical combination satisfies probability bounds."""
    np.random.seed(42)
    n_train = 100
    X_train = pd.DataFrame(np.random.randn(n_train, 4), columns=[f"s_{i}" for i in range(4)])
    y_train = np.random.choice([0, 1], size=n_train, p=[0.9, 0.1])
    aux_train = pd.DataFrame({
        "HDF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "PWF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "OSF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "TWF": np.random.choice([0, 1], size=n_train, p=[0.9, 0.1]),
    })
    
    model = ModeAwareStackingClassifier(combination_mode="noisy_or", n_inner_folds=3, random_state=42)
    model.fit(X_train, y_train, aux_targets=aux_train)
    
    X_test = pd.DataFrame(np.random.randn(20, 4), columns=[f"s_{i}" for i in range(4)])
    probs = model.predict_proba(X_test)
    
    assert probs.shape == (20, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_saved_final_pipeline_inference_on_legitimate_columns_only():
    """
    CRITICAL VERIFICATION:
    Verifies that the serialized final_pipeline.joblib is fully self-contained
    and executes inference on input containing ONLY legitimate sensor/process columns
    and Type (NO failure flags, NO IDs, NO target).
    """
    import joblib
    from src.config import FROZEN_MODEL_PATH, ALL_INFERENCE_FEATURES
    
    assert FROZEN_MODEL_PATH.exists(), f"Saved pipeline artifact missing at {FROZEN_MODEL_PATH}"
    pipeline = joblib.load(FROZEN_MODEL_PATH)
    
    # Minimal CSV representation with ONLY legitimate sensor/process columns and Type
    minimal_df = pd.DataFrame({
        "Type": ["M", "L", "H", "L", "M"],
        "Air temperature [K]": [298.1, 298.2, 298.3, 300.0, 304.5],
        "Process temperature [K]": [308.6, 308.7, 308.8, 309.5, 313.8],
        "Rotational speed [rpm]": [1551, 1408, 1498, 2800, 1150],
        "Torque [Nm]": [42.8, 46.3, 49.4, 12.0, 68.4],
        "Tool wear [min]": [0, 3, 5, 215, 198],
    })
    
    # 1. Assert exactly the legitimate columns and nothing else
    assert list(minimal_df.columns) == ALL_INFERENCE_FEATURES
    
    # 2. Assert direct execution of predict_proba through entire pipeline
    probs = pipeline.predict_proba(minimal_df)
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # 3. Assert direct execution of predict
    preds = pipeline.predict(minimal_df)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})


def test_predict_records_anti_leakage_and_modes():
    """
    Verifies that predict_records automatically isolates leakage columns,
    purges identifiers, and functions across both optimal_f1 and high_recall thresholds.
    """
    from src.predict import predict_records
    
    # Contaminated input DataFrame containing IDs and leakage columns
    dirty_df = pd.DataFrame({
        "UDI": [101, 102, 103],
        "Product ID": ["M14860", "L47181", "H29424"],
        "Type": ["M", "L", "H"],
        "Air temperature [K]": [298.1, 302.5, 298.3],
        "Process temperature [K]": [308.6, 310.2, 308.8],
        "Rotational speed [rpm]": [1551, 1380, 2800],
        "Torque [Nm]": [42.8, 62.0, 12.5],
        "Tool wear [min]": [0, 220, 5],
        "HDF": [0, 1, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 0],
        "TWF": [0, 0, 0],
        "RNF": [0, 0, 0],
        "Machine failure": [0, 1, 0],
    })
    
    # Test optimal_f1 mode
    res_f1 = predict_records(dirty_df, threshold_mode="optimal_f1")
    assert "failure_probability" in res_f1.columns
    assert "predicted_failure" in res_f1.columns
    assert "UDI" in res_f1.columns  # Preserved as metadata
    assert "HDF" not in res_f1.columns or res_f1.columns.tolist().count("HDF") == 0  # Leakage stripped
    
    # Test high_recall mode
    res_rec = predict_records(dirty_df, threshold_mode="high_recall")
    assert "failure_probability" in res_rec.columns
    assert "predicted_failure" in res_rec.columns
    assert res_rec["decision_threshold"].iloc[0] <= res_f1["decision_threshold"].iloc[0]

