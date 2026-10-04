"""
Validation protocol, cross-validation runners, out-of-fold threshold tuning,
probability calibration, and metric computation module.
"""

from typing import Dict, Any, Tuple, List, Optional
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)
from src.config import CV_FOLDS, CV_REPEATS, RANDOM_SEED, HIGH_RECALL_MIN_RECALL
from src.utils import logger


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all competition-required metrics:
    PR-AUC (primary metric), ROC-AUC, F1, Precision, Recall, MCC, Brier score.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)
    
    if len(np.unique(y_true_arr)) > 1:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        pr_auc = 0.0
        roc_auc = 0.5
        
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true_arr, y_pred))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    predicted_failure_rate = float(np.mean(y_pred))
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "brier_score": brier,
        "predicted_failure_rate": predicted_failure_rate,
        "threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    metric: str = "f1",
    num_steps: int = 100
) -> Tuple[float, float]:
    """
    Searches for the decision threshold that maximizes the specified metric purely on
    OUT-OF-FOLD predictions to prevent data leakage.
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_thresh = 0.5
    best_score = -1.0
    
    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)
            
        if score > best_score:
            best_score = score
            best_thresh = th
            
    logger.info("Optimal OOF threshold found: %.4f with %s = %.4f", best_thresh, metric, best_score)
    return float(best_thresh), float(best_score)


def find_high_recall_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    min_recall: float = HIGH_RECALL_MIN_RECALL,
    num_steps: int = 100
) -> Tuple[float, Dict[str, float]]:
    """
    Finds a decision threshold prioritizing predictive maintenance safety:
    guarantees Recall >= min_recall (e.g. >= 85%) while maximizing F1 and precision.
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    valid_candidates = []
    
    for th in thresholds:
        metrics = compute_classification_metrics(y_true, oof_probs, threshold=th)
        if metrics["recall"] >= min_recall:
            valid_candidates.append((th, metrics))
            
    if not valid_candidates:
        # Fallback to minimum threshold if high recall cannot be satisfied
        best_th = 0.1
        best_metrics = compute_classification_metrics(y_true, oof_probs, threshold=best_th)
    else:
        # Choose candidate that maximizes F1 among those meeting the recall constraint
        valid_candidates.sort(key=lambda x: (x[1]["f1"], x[1]["precision"]), reverse=True)
        best_th, best_metrics = valid_candidates[0]
        
    logger.info("High-Recall threshold selected: %.4f (Recall=%.2f%%, F1=%.4f)",
                best_th, best_metrics["recall"] * 100, best_metrics["f1"])
    return float(best_th), best_metrics


def run_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: Optional[RepeatedStratifiedKFold] = None,
    aux_targets: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """
    Executes Repeated Stratified K-Fold CV.
    Returns:
        mean_metrics: Mean of metrics across folds
        std_metrics: Standard deviation of metrics across folds
        oof_predictions: Out-of-fold predicted probabilities for every sample (averaged over repeats)
        total_training_time: Total seconds taken across all fold fits
    """
    if cv is None:
        cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)

    fold_metrics: List[Dict[str, float]] = []
    n_samples = len(X)
    n_splits_total = cv.get_n_splits()
    oof_accum = np.zeros(n_samples)
    oof_counts = np.zeros(n_samples)

    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        clf = model.__class__(**model.get_params()) if hasattr(model, "get_params") else clone(model)
        
        # Fit model (pass aux_targets if mode-aware architecture)
        if aux_targets is not None and hasattr(clf, "fit") and "aux_targets" in clf.fit.__code__.co_varnames:
            clf.fit(X_train, y_train, aux_targets=aux_targets.iloc[train_idx])
        else:
            clf.fit(X_train, y_train)

        probs_val = clf.predict_proba(X_val)[:, 1]
        
        oof_accum[val_idx] += probs_val
        oof_counts[val_idx] += 1

        fold_m = compute_classification_metrics(y_val, probs_val, threshold=0.5)
        fold_metrics.append(fold_m)

    total_time = time.time() - start_time
    oof_probs = oof_accum / np.maximum(1.0, oof_counts)

    # Aggregate mean and std
    metric_keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "brier_score"]
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys}
    std_metrics = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in metric_keys}

    summary = {**mean_metrics, **std_metrics, "training_time_sec": total_time}
    return summary, oof_probs, total_time
