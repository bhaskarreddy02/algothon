"""
Feature Ablation Study: Model A (Raw Features Only) vs Model B (Raw + Physics-Informed Features)
Using Repeated Stratified K-Fold CV (5 folds x 3 repeats) on the 80% training development set.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import TRAIN_DATA_PATH, RANDOM_SEED, FIGURES_DIR, REPORTS_DIR
from src.data import load_data
from src.features import build_preprocessing_pipeline
from src.utils import logger, seed_everything

def run_ablation():
    seed_everything(RANDOM_SEED)
    logger.info("Loading development training data from %s...", TRAIN_DATA_PATH)
    X, y, _ = load_data(TRAIN_DATA_PATH, require_target=True, drop_leakage=True)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    
    models = {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "LightGBM": lambda: LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, verbose=-1, n_jobs=-1),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "Logistic Regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
    }
    
    records = []
    fold_details = []

    logger.info("Executing 15-split Repeated Stratified CV for Model A (Raw) vs Model B (Raw + Physics)...")
    
    for model_name, model_fn in models.items():
        logger.info("Evaluating model: %s", model_name)
        scale = (model_name == "Logistic Regression")
        
        pipe_raw = build_preprocessing_pipeline(include_physics=False, scale_numeric=scale)
        pipe_physics = build_preprocessing_pipeline(include_physics=True, scale_numeric=scale)
        
        prauc_raw, roauc_raw, f1_raw = [], [], []
        prauc_phys, roauc_phys, f1_phys = [], [], []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Model A: Raw features
            X_tr_raw = pipe_raw.fit_transform(X_train)
            X_va_raw = pipe_raw.transform(X_val)
            clf_a = model_fn()
            clf_a.fit(X_tr_raw, y_train)
            probs_a = clf_a.predict_proba(X_va_raw)[:, 1]
            
            pr_a = average_precision_score(y_val, probs_a)
            roc_a = roc_auc_score(y_val, probs_a)
            f1_a = f1_score(y_val, (probs_a >= 0.5).astype(int), zero_division=0)
            
            prauc_raw.append(pr_a)
            roauc_raw.append(roc_a)
            f1_raw.append(f1_a)
            
            # Model B: Raw + Physics features
            X_tr_phys = pipe_physics.fit_transform(X_train)
            X_va_phys = pipe_physics.transform(X_val)
            clf_b = model_fn()
            clf_b.fit(X_tr_phys, y_train)
            probs_b = clf_b.predict_proba(X_va_phys)[:, 1]
            
            pr_b = average_precision_score(y_val, probs_b)
            roc_b = roc_auc_score(y_val, probs_b)
            f1_b = f1_score(y_val, (probs_b >= 0.5).astype(int), zero_division=0)
            
            prauc_phys.append(pr_b)
            roauc_phys.append(roc_b)
            f1_phys.append(f1_b)
            
            fold_details.append({
                "Model": model_name,
                "Fold": fold_idx,
                "Raw_PRAUC": pr_a,
                "Physics_PRAUC": pr_b,
                "Delta_PRAUC": pr_b - pr_a,
            })
            
        pr_delta = np.mean(prauc_phys) - np.mean(prauc_raw)
        pr_pct = (pr_delta / np.mean(prauc_raw)) * 100.0
        
        records.append({
            "Model": model_name,
            "Raw PR-AUC (Mean)": np.mean(prauc_raw),
            "Raw PR-AUC (Std)": np.std(prauc_raw),
            "Physics PR-AUC (Mean)": np.mean(prauc_phys),
            "Physics PR-AUC (Std)": np.std(prauc_phys),
            "PR-AUC Absolute Delta": pr_delta,
            "PR-AUC Relative Delta (%)": pr_pct,
            "Raw ROC-AUC (Mean)": np.mean(roauc_raw),
            "Physics ROC-AUC (Mean)": np.mean(roauc_phys),
            "Raw F1 (Mean)": np.mean(f1_raw),
            "Physics F1 (Mean)": np.mean(f1_phys),
        })
        
        logger.info(
            "%s: Raw PR-AUC = %.4f +/- %.4f | Physics PR-AUC = %.4f +/- %.4f (Delta: %+.4f / %+.2f%%)",
            model_name, np.mean(prauc_raw), np.std(prauc_raw), np.mean(prauc_phys), np.std(prauc_phys),
            pr_delta, pr_pct
        )

    df_ablation = pd.DataFrame(records)
    ablation_csv_path = REPORTS_DIR / "feature_ablation.csv"
    df_ablation.to_csv(ablation_csv_path, index=False)
    logger.info("Saved ablation comparison table to %s", ablation_csv_path)
    
    # Plot Ablation Comparison
    df_folds = pd.DataFrame(fold_details)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Barplot comparing PR-AUC
    models_list = list(models.keys())
    x = np.arange(len(models_list))
    width = 0.35
    
    raw_means = df_ablation["Raw PR-AUC (Mean)"]
    raw_stds = df_ablation["Raw PR-AUC (Std)"]
    phys_means = df_ablation["Physics PR-AUC (Mean)"]
    phys_stds = df_ablation["Physics PR-AUC (Std)"]
    
    axes[0].bar(x - width/2, raw_means, width, yerr=raw_stds, label='Model A: Raw Features Only', color='#7f8c8d', capsize=5, alpha=0.85)
    axes[0].bar(x + width/2, phys_means, width, yerr=phys_stds, label='Model B: Raw + Physics-Informed', color='#27ae60', capsize=5, alpha=0.9)
    axes[0].set_ylabel('PR-AUC (Precision-Recall Area)', fontsize=12, fontweight='bold')
    axes[0].set_title('Ablation Study: PR-AUC Across Model Architectures\n(5-Fold x 3-Repeat Stratified CV)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_list, fontsize=11)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc='lower right', frameon=True)
    
    for i in range(len(models_list)):
        delta_str = f"+{df_ablation['PR-AUC Relative Delta (%)'].iloc[i]:.1f}%"
        axes[0].annotate(delta_str, (x[i] + width/2, phys_means.iloc[i] + phys_stds.iloc[i] + 0.03),
                         ha='center', va='bottom', fontweight='bold', color='#1e8449', fontsize=10)

    # Boxplot of PR-AUC improvement deltas
    sns.boxplot(data=df_folds, x='Model', y='Delta_PRAUC', ax=axes[1], palette='crest', width=0.4)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Improvement Line')
    axes[1].set_ylabel('PR-AUC Delta (Physics - Raw)', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Per-Fold PR-AUC Gains from Physics Engineering', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "09_feature_ablation_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    logger.info("Saved ablation visualization to %s", fig_path)
    
    print("\n================ FEATURE ABLATION SUMMARY ================")
    print(df_ablation[["Model", "Raw PR-AUC (Mean)", "Physics PR-AUC (Mean)", "PR-AUC Relative Delta (%)"]].to_string(index=False))
    print("==========================================================\n")

if __name__ == "__main__":
    run_ablation()
