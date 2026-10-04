"""
End-to-end model training, hyperparameter optimization, mode-aware experimentation,
calibration, threshold tuning, and model artifact serialization.

Strict Anti-Leakage:
Operates EXCLUSIVELY on train_development.csv (80% development split).
The 20% holdout test set (holdout_test.csv) remains locked until Phase 5.
"""

import sys
from pathlib import Path
import time
import json
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, matthews_corrcoef, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    TRAIN_DATA_PATH,
    FROZEN_MODEL_PATH,
    METRICS_REPORT_PATH,
    THRESHOLD_CONFIG_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    CV_FOLDS,
    CV_REPEATS,
    HIGH_RECALL_MIN_RECALL,
)
from src.data import load_data
from src.features import build_preprocessing_pipeline
from src.models import get_baseline_models, ModeAwareStackingClassifier
from src.validation import compute_classification_metrics, find_optimal_threshold, find_high_recall_threshold
from src.utils import logger, seed_everything, save_artifact, save_json

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_model_comparison(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Evaluates baseline and candidate models using Repeated Stratified K-Fold CV (5x3).
    Saves comprehensive performance metrics to reports/model_comparison.csv.
    """
    logger.info("Starting model comparison with Repeated Stratified K-Fold CV (5 folds x 3 repeats)...")
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    models = get_baseline_models(scale_pos_weight=28.5)
    
    records = []
    
    for name, model in models.items():
        logger.info("Evaluating model: %s", name)
        scale = (name == "logistic_regression")
        pipe_prep = build_preprocessing_pipeline(include_physics=True, scale_numeric=scale)
        
        pr_scores, roc_scores, f1_scores = [], [], []
        prec_scores, rec_scores, mcc_scores = [], [], []
        fold_times = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            t0 = time.time()
            X_tr_proc = pipe_prep.fit_transform(X_train)
            X_va_proc = pipe_prep.transform(X_val)
            
            clf = clone(model)
            clf.fit(X_tr_proc, y_train)
            probs = clf.predict_proba(X_va_proc)[:, 1]
            fold_times.append(time.time() - t0)
            
            m = compute_classification_metrics(y_val, probs, threshold=0.5)
            pr_scores.append(m["pr_auc"])
            roc_scores.append(m["roc_auc"])
            f1_scores.append(m["f1"])
            prec_scores.append(m["precision"])
            rec_scores.append(m["recall"])
            mcc_scores.append(m["mcc"])
            
        records.append({
            "model": name,
            "features": "physics_augmented",
            "PR-AUC mean": float(np.mean(pr_scores)),
            "PR-AUC std": float(np.std(pr_scores)),
            "ROC-AUC": float(np.mean(roc_scores)),
            "F1": float(np.mean(f1_scores)),
            "precision": float(np.mean(prec_scores)),
            "recall": float(np.mean(rec_scores)),
            "MCC": float(np.mean(mcc_scores)),
            "training time (s)": float(np.sum(fold_times)),
        })
        logger.info("%s -> PR-AUC: %.4f +/- %.4f | F1: %.4f | Recall: %.4f",
                    name, np.mean(pr_scores), np.std(pr_scores), np.mean(f1_scores), np.mean(rec_scores))
        
    df_comparison = pd.DataFrame(records).sort_values(by="PR-AUC mean", ascending=False)
    df_comparison.to_csv(METRICS_REPORT_PATH, index=False)
    logger.info("Saved model comparison to %s", METRICS_REPORT_PATH)
    return df_comparison


def run_optuna_tuning(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimizes hyperparameters for top candidate models (LightGBM, Random Forest, XGBoost)
    using Optuna targeting cross-validated PR-AUC on development training data only.
    """
    logger.info("Starting Bayesian Hyperparameter Optimization with Optuna (%d trials)...", n_trials)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    prep = build_preprocessing_pipeline(include_physics=True, scale_numeric=False)
    X_proc = prep.fit_transform(X)
    
    # 1. Optimize LightGBM
    def lgbm_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 30.0),
            "random_state": RANDOM_SEED,
            "verbose": -1,
            "n_jobs": -1,
        }
        scores = []
        for tr_idx, va_idx in cv.split(X_proc, y):
            clf = LGBMClassifier(**params)
            clf.fit(X_proc[tr_idx], y.iloc[tr_idx])
            probs = clf.predict_proba(X_proc[va_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[va_idx], probs))
        return float(np.mean(scores))

    study_lgbm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study_lgbm.optimize(lgbm_objective, n_trials=n_trials)
    logger.info("Best LightGBM PR-AUC: %.4f with params: %s", study_lgbm.best_value, study_lgbm.best_params)
    
    # 2. Optimize Random Forest
    def rf_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 150),
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        scores = []
        for tr_idx, va_idx in cv.split(X_proc, y):
            clf = RandomForestClassifier(**params)
            clf.fit(X_proc[tr_idx], y.iloc[tr_idx])
            probs = clf.predict_proba(X_proc[va_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[va_idx], probs))
        return float(np.mean(scores))

    study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study_rf.optimize(rf_objective, n_trials=n_trials)
    logger.info("Best Random Forest PR-AUC: %.4f with params: %s", study_rf.best_value, study_rf.best_params)

    best_tuning_results = {
        "lgbm": {
            "best_pr_auc": study_lgbm.best_value,
            "best_params": study_lgbm.best_params,
        },
        "random_forest": {
            "best_pr_auc": study_rf.best_value,
            "best_params": study_rf.best_params,
        },
    }
    optuna_path = MODELS_DIR / "optuna_best_params.json"
    save_json(best_tuning_results, optuna_path)
    return best_tuning_results


def run_mode_aware_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    leakage_df: pd.DataFrame,
    best_lgbm_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates the Mode-Aware Prediction Innovation.
    Compares:
      1. Standard Best Single Model (Tuned LightGBM)
      2. Mode-Aware Stacking (Strict Out-Of-Fold Auxiliary Probabilities)
      3. Mode-Aware Noisy-OR Combination
    """
    logger.info("Executing Mode-Aware Prediction Experiment with Strict OOF Auxiliary Stacking...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    prep = build_preprocessing_pipeline(include_physics=True, scale_numeric=False)
    X_proc = prep.fit_transform(X)
    
    # Base single model
    base_scores = []
    stacking_scores = []
    noisy_or_scores = []
    
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_proc, y)):
        X_tr, y_tr = X_proc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_proc[va_idx], y.iloc[va_idx]
        aux_tr = leakage_df.iloc[tr_idx]
        
        # 1. Standard single model
        lgbm_single = LGBMClassifier(**best_lgbm_params, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        lgbm_single.fit(X_tr, y_tr)
        pr_single = average_precision_score(y_va, lgbm_single.predict_proba(X_va)[:, 1])
        base_scores.append(pr_single)
        
        # 2. Mode-Aware Stacking
        aux_clf = LGBMClassifier(n_estimators=60, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        main_clf = LGBMClassifier(**best_lgbm_params, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        stacking_model = ModeAwareStackingClassifier(
            base_estimator=main_clf,
            auxiliary_estimator=aux_clf,
            combination_mode="stacking",
            n_inner_folds=4,
            random_state=RANDOM_SEED + fold_idx
        )
        stacking_model.fit(X_tr, y_tr, aux_targets=aux_tr)
        pr_stack = average_precision_score(y_va, stacking_model.predict_proba(X_va)[:, 1])
        stacking_scores.append(pr_stack)
        
        # 3. Mode-Aware Noisy-OR
        noisy_or_model = ModeAwareStackingClassifier(
            auxiliary_estimator=aux_clf,
            combination_mode="noisy_or",
            n_inner_folds=4,
            random_state=RANDOM_SEED + fold_idx
        )
        noisy_or_model.fit(X_tr, y_tr, aux_targets=aux_tr)
        pr_noisy = average_precision_score(y_va, noisy_or_model.predict_proba(X_va)[:, 1])
        noisy_or_scores.append(pr_noisy)
        
    results = {
        "standard_single_lgbm_prauc": float(np.mean(base_scores)),
        "mode_aware_stacking_prauc": float(np.mean(stacking_scores)),
        "mode_aware_noisy_or_prauc": float(np.mean(noisy_or_scores)),
        "stacking_prauc_delta": float(np.mean(stacking_scores) - np.mean(base_scores)),
        "noisy_or_prauc_delta": float(np.mean(noisy_or_scores) - np.mean(base_scores)),
    }
    
    logger.info("Mode-Aware Comparison: Single LightGBM = %.4f | Stacking = %.4f | Noisy-OR = %.4f",
                results["standard_single_lgbm_prauc"],
                results["mode_aware_stacking_prauc"],
                results["mode_aware_noisy_or_prauc"])
    
    save_json(results, REPORTS_DIR / "mode_aware_comparison.json")
    return results


def run_calibration_and_thresholding(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Computes probability calibration (Uncalibrated vs Platt Sigmoid vs Isotonic) on OOF predictions,
    tunes decision thresholds strictly out-of-fold, and saves diagnostic figures.
    """
    logger.info("Evaluating probability calibration and threshold optimization on OOF predictions...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    prep = build_preprocessing_pipeline(include_physics=True, scale_numeric=False)
    X_proc = prep.fit_transform(X)
    
    oof_uncal = np.zeros(len(X))
    oof_platt = np.zeros(len(X))
    oof_isotonic = np.zeros(len(X))
    
    from sklearn.isotonic import IsotonicRegression
    
    # 1. Generate OOF uncalibrated probabilities
    for tr_idx, va_idx in cv.split(X_proc, y):
        X_tr, y_tr = X_proc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_proc[va_idx], y.iloc[va_idx]
        
        base_clf = LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        base_clf.fit(X_tr, y_tr)
        oof_uncal[va_idx] = base_clf.predict_proba(X_va)[:, 1]
        
    # 2. Cross-validated Platt (Sigmoid) and Isotonic Calibration on OOF Probabilities
    cal_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for c_tr, c_va in cal_cv.split(oof_uncal.reshape(-1, 1), y):
        # Platt Sigmoid calibration
        lr = LogisticRegression(random_state=RANDOM_SEED)
        lr.fit(oof_uncal[c_tr].reshape(-1, 1), y.iloc[c_tr])
        oof_platt[c_va] = lr.predict_proba(oof_uncal[c_va].reshape(-1, 1))[:, 1]
        
        # Isotonic calibration
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_uncal[c_tr], y.iloc[c_tr])
        oof_isotonic[c_va] = iso.predict(oof_uncal[c_va])
        
    brier_uncal = brier_score_loss(y, oof_uncal)
    brier_platt = brier_score_loss(y, oof_platt)
    brier_iso = brier_score_loss(y, oof_isotonic)
    
    logger.info("Brier Scores -> Uncalibrated: %.5f | Platt: %.5f | Isotonic: %.5f",
                brier_uncal, brier_platt, brier_iso)
    
    # Plot Calibration Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    prob_true_u, prob_pred_u = calibration_curve(y, oof_uncal, n_bins=10)
    prob_true_p, prob_pred_p = calibration_curve(y, oof_platt, n_bins=10)
    prob_true_i, prob_pred_i = calibration_curve(y, oof_isotonic, n_bins=10)
    
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(prob_pred_u, prob_true_u, "s-", color="#e74c3c", label=f"Uncalibrated (Brier={brier_uncal:.4f})")
    axes[0].plot(prob_pred_p, prob_true_p, "o-", color="#3498db", label=f"Platt / Sigmoid (Brier={brier_platt:.4f})")
    axes[0].plot(prob_pred_i, prob_true_i, "^-", color="#2ecc71", label=f"Isotonic (Brier={brier_iso:.4f})")
    axes[0].set_xlabel("Mean Predicted Failure Probability", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Fraction of True Machine Failures", fontsize=11, fontweight="bold")
    axes[0].set_title("Probability Calibration Reliability Diagram (OOF CV)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper left")
    
    # Threshold Tuning Trade-Off Curve
    thresholds = np.linspace(0.01, 0.99, 100)
    prec_list, rec_list, f1_list, mcc_list = [], [], [], []
    
    for th in thresholds:
        m = compute_classification_metrics(y, oof_uncal, threshold=th)
        prec_list.append(m["precision"])
        rec_list.append(m["recall"])
        f1_list.append(m["f1"])
        mcc_list.append(m["mcc"])
        
    best_f1_th, best_f1_score = find_optimal_threshold(y.values, oof_uncal, metric="f1")
    high_rec_th, high_rec_metrics = find_high_recall_threshold(y.values, oof_uncal, min_recall=HIGH_RECALL_MIN_RECALL)
    
    axes[1].plot(thresholds, prec_list, label="Precision", color="#2980b9", linewidth=2)
    axes[1].plot(thresholds, rec_list, label="Recall", color="#e67e22", linewidth=2)
    axes[1].plot(thresholds, f1_list, label="F1-Score", color="#27ae60", linewidth=2.5)
    axes[1].plot(thresholds, mcc_list, label="MCC", color="#8e44ad", linestyle=":", linewidth=2)
    axes[1].axvline(best_f1_th, color="#27ae60", linestyle="--", linewidth=1.5,
                    label=f"Max F1 Threshold ({best_f1_th:.2f}, F1={best_f1_score:.3f})")
    axes[1].axvline(high_rec_th, color="#e67e22", linestyle="--", linewidth=1.5,
                    label=f"High-Recall Threshold ({high_rec_th:.2f}, Rec={high_rec_metrics['recall']*100:.1f}%)")
    axes[1].set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    axes[1].set_title("Operational Threshold Trade-Off Analysis (OOF Predictions)", fontsize=13, fontweight="bold")
    axes[1].legend(loc="center left")
    
    plt.tight_layout()
    cal_fig_path = FIGURES_DIR / "10_calibration_and_threshold_tradeoff.png"
    plt.savefig(cal_fig_path)
    plt.close()
    logger.info("Saved calibration and threshold trade-off curves to %s", cal_fig_path)
    
    threshold_config = {
        "optimal_f1_threshold": float(best_f1_th),
        "optimal_f1_score": float(best_f1_score),
        "high_recall_threshold": float(high_rec_th),
        "high_recall_metrics": high_rec_metrics,
        "default_threshold": float(best_f1_th),
        "brier_scores": {
            "uncalibrated": float(brier_uncal),
            "platt": float(brier_platt),
            "isotonic": float(brier_iso),
        },
    }
    save_json(threshold_config, THRESHOLD_CONFIG_PATH)
    
    # Train final full pipeline on complete 80% development set
    final_pipeline = Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline(include_physics=True, scale_numeric=False)),
        ("classifier", LGBMClassifier(**best_params, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1))
    ])
    final_pipeline.fit(X, y)
    save_artifact(final_pipeline, FROZEN_MODEL_PATH)
    logger.info("Successfully trained and serialized final frozen pipeline to %s", FROZEN_MODEL_PATH)
    
    return final_pipeline, threshold_config


def main():
    seed_everything(RANDOM_SEED)
    logger.info("Starting Phase 4 Training & Cross-Validation Workflow...")
    
    # 1. Ingest Development Training Data (80% split)
    X, y, leakage_df = load_data(TRAIN_DATA_PATH, require_target=True, drop_leakage=True)
    
    # 2. Run Comprehensive Model Suite Comparison
    if METRICS_REPORT_PATH.exists():
        logger.info("Loading existing model comparison from %s", METRICS_REPORT_PATH)
        df_comparison = pd.read_csv(METRICS_REPORT_PATH)
    else:
        df_comparison = run_model_comparison(X, y)
    print("\n--- Model Comparison Summary ---")
    print(df_comparison[["model", "PR-AUC mean", "PR-AUC std", "ROC-AUC", "F1", "recall", "MCC"]].to_string(index=False))
    
    # 3. Hyperparameter Optimization via Optuna
    optuna_path = MODELS_DIR / "optuna_best_params.json"
    if optuna_path.exists() and False:  # Force re-tuning with 50+ trials
        logger.info("Loading existing Optuna tuning results from %s", optuna_path)
        with open(optuna_path, "r", encoding="utf-8") as f:
            optuna_results = json.load(f)
    else:
        optuna_results = run_optuna_tuning(X, y, n_trials=50)
    best_lgbm_params = optuna_results["lgbm"]["best_params"]
    
    # 4. Mode-Aware Experimentation (Strictly Out-of-Fold Auxiliary Stacking)
    mode_aware_path = REPORTS_DIR / "mode_aware_comparison.json"
    if mode_aware_path.exists():
        logger.info("Loading existing Mode-Aware experiment results from %s", mode_aware_path)
        with open(mode_aware_path, "r", encoding="utf-8") as f:
            mode_aware_results = json.load(f)
    else:
        mode_aware_results = run_mode_aware_experiment(X, y, leakage_df, best_lgbm_params)
        
    # 5. Calibration & Out-Of-Fold Threshold Tuning
    final_pipeline, thresh_cfg = run_calibration_and_thresholding(X, y, best_lgbm_params)
    
    logger.info("Phase 4 Complete: Final model pipeline, metrics, and threshold config successfully produced.")

if __name__ == "__main__":
    main()
