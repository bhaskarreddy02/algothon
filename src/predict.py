"""
PredictiveGuard Production Batch & Single-Record Inference Module.
ALGOTHON'26 — Track: ALG-DATA-02 "Predict What Happens Next"

Provides end-to-end production inference on unseen machine telemetry datasets.
Strict Anti-Leakage & Operational Guarantees:
1. Rejects or removes identifier columns (UDI, Product ID).
2. Prevents failure mode flags (HDF, PWF, OSF, TWF, RNF) from entering the inference pipeline.
3. Fully resilient to schema perturbations (capitalization, unit brackets, underscores, common aliases).
4. Employs the frozen production pipeline (fitted exclusively on development data).
5. Supports calibrated probabilities and configurable operational decision thresholds:
   - 'optimal_f1': Balanced operating point maximizing classification F1-score.
   - 'high_recall': Conservative safety operating point prioritizing early intervention.
   - Custom threshold: Explicit float override.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FROZEN_MODEL_PATH,
    THRESHOLD_CONFIG_PATH,
    PREDICTIONS_DIR,
    ALL_INFERENCE_FEATURES,
    TARGET_COLUMN,
)
from src.data import (
    normalize_column_names,
    remove_identifiers,
    remove_leakage_columns,
    validate_schema,
)
from src.utils import logger


def load_threshold_configuration(config_path: Union[str, Path] = THRESHOLD_CONFIG_PATH) -> Dict[str, Any]:
    """Loads calibrated operational decision thresholds from JSON config."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Threshold config not found at %s. Falling back to 0.50.", config_path)
        return {"optimal_f1_threshold": 0.50, "high_recall_threshold": 0.35, "default_threshold": 0.50}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_inference_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardizes raw input DataFrame for inference:
    1. Normalizes column names and aliases.
    2. Purges identifier columns (UDI, Product ID) while caching them for output reporting.
    3. Strictly drops failure mode flags (TWF, HDF, PWF, OSF, RNF) to guarantee zero leakage.
    4. Validates presence of mandatory sensor features.
    
    Returns:
        clean_features: Feature DataFrame aligned with pipeline expectations.
        metadata_df: Extracted identifiers/metadata to attach to output if desired.
    """
    df_norm = normalize_column_names(df.copy())
    clean_df, id_df = remove_identifiers(df_norm)
    clean_df, leak_df = remove_leakage_columns(clean_df)
    
    # If target column exists (e.g. during offline scoring), drop it from features
    if TARGET_COLUMN in clean_df.columns:
        clean_df = clean_df.drop(columns=[TARGET_COLUMN])
        
    validate_schema(clean_df, require_target=False)
    
    # Ensure correct column ordering for features
    clean_features = clean_df[ALL_INFERENCE_FEATURES].copy()
    
    metadata = id_df if id_df is not None else pd.DataFrame(index=df.index)
    return clean_features, metadata


def predict_records(
    df: pd.DataFrame,
    model_path: Union[str, Path] = FROZEN_MODEL_PATH,
    threshold_config_path: Union[str, Path] = THRESHOLD_CONFIG_PATH,
    threshold_mode: str = "optimal_f1",
    custom_threshold: Optional[float] = None,
    attach_inputs: bool = False,
) -> pd.DataFrame:
    """
    Generates failure probability and binary classifications for input records.
    
    Args:
        df: Input DataFrame containing raw sensor telemetry.
        model_path: Path to serialized final pipeline artifact.
        threshold_config_path: Path to threshold configuration JSON.
        threshold_mode: 'optimal_f1' or 'high_recall'.
        custom_threshold: Optional explicit threshold override in [0.0, 1.0].
        attach_inputs: If True, prepends original input columns to the output.
        
    Returns:
        results_df: DataFrame with predicted probabilities and binary decisions.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the pipeline first.")
        
    pipeline = joblib.load(model_path)
    clean_features, metadata = prepare_inference_features(df)
    
    # Generate probabilities via frozen pipeline
    probabilities = pipeline.predict_proba(clean_features)[:, 1]
    
    # Determine operating threshold
    thresh_cfg = load_threshold_configuration(threshold_config_path)
    if custom_threshold is not None:
        active_threshold = float(custom_threshold)
        mode_used = f"custom ({active_threshold:.4f})"
    elif threshold_mode == "high_recall":
        active_threshold = float(thresh_cfg.get("high_recall_threshold", 0.3763))
        mode_used = f"high_recall ({active_threshold:.4f})"
    else:
        active_threshold = float(thresh_cfg.get("optimal_f1_threshold", 0.7029))
        mode_used = f"optimal_f1 ({active_threshold:.4f})"
        
    binary_predictions = (probabilities >= active_threshold).astype(int)
    
    output_df = pd.DataFrame({
        "failure_probability": np.round(probabilities, 5),
        "predicted_failure": binary_predictions,
        "decision_threshold": active_threshold,
        "operating_mode": threshold_mode if custom_threshold is None else "custom",
    }, index=df.index)
    
    # Prepend identifier columns if available
    if not metadata.empty:
        output_df = pd.concat([metadata, output_df], axis=1)
        
    if attach_inputs:
        output_df = pd.concat([df, output_df], axis=1)
        
    logger.info("Generated predictions for %d records using %s operating threshold.",
                len(output_df), mode_used)
    return output_df


def run_batch_inference(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = PREDICTIONS_DIR / "predictions.csv",
    model_path: Union[str, Path] = FROZEN_MODEL_PATH,
    threshold_config_path: Union[str, Path] = THRESHOLD_CONFIG_PATH,
    threshold_mode: str = "optimal_f1",
    custom_threshold: Optional[float] = None,
) -> Path:
    """CLI / batch execution entrypoint for file-to-file predictions."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting batch inference on: %s", input_path)
    df_raw = pd.read_csv(input_path)
    
    preds_df = predict_records(
        df=df_raw,
        model_path=model_path,
        threshold_config_path=threshold_config_path,
        threshold_mode=threshold_mode,
        custom_threshold=custom_threshold,
        attach_inputs=True,
    )
    
    preds_df.to_csv(output_path, index=False)
    logger.info("Batch inference complete. Predictions written to %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="PredictiveGuard Production Inference Module (ALGOTHON'26)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV for prediction")
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_DIR / "predictions.csv"), help="Path to output CSV")
    parser.add_argument("--model", type=str, default=str(FROZEN_MODEL_PATH), help="Path to saved model pipeline")
    parser.add_argument("--threshold-config", type=str, default=str(THRESHOLD_CONFIG_PATH), help="Path to threshold JSON")
    parser.add_argument("--threshold-mode", type=str, choices=["optimal_f1", "high_recall"], default="optimal_f1",
                        help="Operating mode: 'optimal_f1' (balanced) or 'high_recall' (safety-first)")
    parser.add_argument("--safety-mode", action="store_true", help="Shortcut for --threshold-mode high_recall")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit custom decision threshold override")
    
    args = parser.parse_args()
    mode = "high_recall" if args.safety_mode else args.threshold_mode
    
    run_batch_inference(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        threshold_config_path=args.threshold_config,
        threshold_mode=mode,
        custom_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
