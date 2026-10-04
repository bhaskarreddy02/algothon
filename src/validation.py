"""
Validation protocol, cross-validation runners, out-of-fold threshold tuning,
and metric computation module.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)
from src.config import CV_FOLDS, CV_REPEATS, RANDOM_SEED
from src.utils import logger

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all competition-required metrics:
    PR-AUC (headline metric), ROC-AUC, F1, Precision, Recall, MCC.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Check if both classes are present for ROC-AUC / PR-AUC calculation
    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        pr_auc = 0.0
        roc_auc = 0.5
        
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
    }

def find_optimal_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    metric: str = "f1",
    num_steps: int = 100
) -> Tuple[float, float]:
    """
    Searches for the threshold that maximizes the specified metric purely on
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
