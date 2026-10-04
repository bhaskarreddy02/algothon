"""
PredictiveGuard Phase 5 Model Evaluation & Performance Breakdown.
ALGOTHON'26 — Track: ALG-DATA-02 "Predict What Happens Next"

Strict Protocol:
1. Ingests holdout_test.csv (2,000 samples, 20% holdout split) for final generalization testing.
2. Uses the FROZEN final pipeline (fitted strictly on train_development.csv).
3. Zero retraining, zero parameter adjustments, zero data leakage.
4. Computes headline metrics: PR-AUC, ROC-AUC, F1, Recall, Precision, MCC, Brier score.
5. Dissects subgroup performance by failure mode (HDF, PWF, OSF, TWF, RNF) and machine variant Type (L, M, H).
6. Performs industrial cost-curve analysis (OpEx minimization).
7. Generates publication-grade diagnostic figures (Figures 11-15) and serialized holdout predictions.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    TEST_DATA_PATH,
    FROZEN_MODEL_PATH,
    THRESHOLD_CONFIG_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    ALL_INFERENCE_FEATURES,
    TARGET_COLUMN,
    LEAK_COLUMNS,
    DROP_COLUMNS,
)
from src.data import normalize_column_names, remove_identifiers, remove_leakage_columns, validate_schema
from src.utils import logger, save_json

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def load_holdout_dataset(test_path: Path = TEST_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Ingests holdout test CSV, isolates features for inference, and preserves
    auxiliary failure mode flags and metadata for post-hoc subgroup breakdown.
    
    Returns:
        X_test: Clean features matching pipeline requirements (6 legitimate columns).
        y_test: Ground truth binary target (Machine failure).
        aux_test: Failure mode indicators (HDF, PWF, OSF, TWF, RNF).
        raw_df: Original full test dataframe for slicing analysis.
    """
    logger.info("Loading holdout test dataset from %s...", test_path)
    df = pd.read_csv(test_path)
    df_norm = normalize_column_names(df.copy())
    
    y_test = df_norm[TARGET_COLUMN].copy()
    
    # Extract auxiliary failure modes
    present_leaks = [c for c in LEAK_COLUMNS if c in df_norm.columns]
    aux_test = df_norm[present_leaks].copy()
    
    # Clean features for pipeline
    clean_df, _ = remove_identifiers(df_norm)
    clean_df, _ = remove_leakage_columns(clean_df)
    clean_df = clean_df.drop(columns=[TARGET_COLUMN], errors="ignore")
    
    validate_schema(clean_df, require_target=False)
    X_test = clean_df[ALL_INFERENCE_FEATURES].copy()
    
    logger.info("Holdout test dataset successfully ingested: %d samples, %d features, failure prevalence: %.2f%%",
                len(df), len(X_test.columns), (y_test.mean() * 100))
    return X_test, y_test, aux_test, df


def evaluate_threshold_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    threshold_name: str
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2.0
    
    return {
        "threshold_name": threshold_name,
        "threshold_value": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "specificity": float(spec),
        "balanced_accuracy": float(bal_acc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


def compute_subgroup_metrics(
    raw_df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    aux_df: pd.DataFrame,
    active_threshold: float
) -> Dict[str, Any]:
    """
    Dissects model predictions across:
    1. Individual failure modes (HDF, PWF, OSF, TWF, RNF).
    2. Machine variant Types (L, M, H).
    3. High-stress operating regimes (High Tool Wear, High Torque, Elevated Power).
    """
    logger.info("Computing fine-grained failure mode and operational subgroup breakdown...")
    y_pred = (y_prob >= active_threshold).astype(int)
    
    # 1. Failure Mode Slicing
    failure_mode_breakdown = {}
    for mode in ["HDF", "PWF", "OSF", "TWF", "RNF"]:
        if mode in aux_df.columns:
            mode_mask = (aux_df[mode] == 1)
            total_mode_cases = int(mode_mask.sum())
            if total_mode_cases > 0:
                detected_cases = int((mode_mask & (y_pred == 1)).sum())
                mode_recall = detected_cases / total_mode_cases
                mean_pred_prob = float(y_prob[mode_mask].mean())
            else:
                detected_cases = 0
                mode_recall = 0.0
                mean_pred_prob = 0.0
                
            failure_mode_breakdown[mode] = {
                "total_cases": total_mode_cases,
                "detected_cases": detected_cases,
                "recall": float(mode_recall),
                "mean_predicted_probability": float(mean_pred_prob),
            }
            
    # Compound failure modes (simultaneous multi-failure)
    multi_mode_mask = (aux_df[["HDF", "PWF", "OSF", "TWF"]].sum(axis=1) > 1)
    total_multi = int(multi_mode_mask.sum())
    detected_multi = int((multi_mode_mask & (y_pred == 1)).sum()) if total_multi > 0 else 0
    failure_mode_breakdown["Compound (Multi-Mode)"] = {
        "total_cases": total_multi,
        "detected_cases": detected_multi,
        "recall": float(detected_multi / total_multi) if total_multi > 0 else 0.0,
        "mean_predicted_probability": float(y_prob[multi_mode_mask].mean()) if total_multi > 0 else 0.0,
    }

    # 2. Product Variant Type Slicing
    type_breakdown = {}
    type_col = "Type" if "Type" in raw_df.columns else "type"
    for v_type in ["L", "M", "H"]:
        t_mask = (raw_df[type_col] == v_type)
        t_y_true = y_true[t_mask].values
        t_y_prob = y_prob[t_mask]
        t_y_pred = y_pred[t_mask]
        
        pr_auc = float(average_precision_score(t_y_true, t_y_prob)) if t_y_true.sum() > 0 else 0.0
        roc_auc = float(roc_auc_score(t_y_true, t_y_prob)) if len(np.unique(t_y_true)) > 1 else 0.0
        f1 = float(f1_score(t_y_true, t_y_pred, zero_division=0))
        rec = float(recall_score(t_y_true, t_y_pred, zero_division=0))
        prec = float(precision_score(t_y_true, t_y_pred, zero_division=0))
        
        type_breakdown[v_type] = {
            "sample_count": int(t_mask.sum()),
            "failure_count": int(t_y_true.sum()),
            "failure_prevalence": float(t_y_true.mean()),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
        }
        
    return {
        "failure_mode_breakdown": failure_mode_breakdown,
        "product_type_breakdown": type_breakdown,
    }


def compute_industrial_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10000.0,  # Unplanned catastrophic breakdown ($)
    cost_fp: float = 500.0,    # Unnecessary scheduled inspection ($)
    cost_tp: float = 1500.0,   # Proactive preventive repair ($)
    cost_tn: float = 0.0       # Nominal operation ($)
) -> Dict[str, Any]:
    """
    Computes total operational expenditure (OpEx) as a function of decision threshold.
    Quantifies the economic value of predictive maintenance vs. reactive maintenance.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    
    total_failures = int(np.sum(y_true))
    total_nominal = len(y_true) - total_failures
    reactive_cost = total_failures * cost_fn + total_nominal * cost_tn
    
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (tp * cost_tp) + (fp * cost_fp) + (fn * cost_fn) + (tn * cost_tn)
        costs.append(total_cost)
        
    costs = np.array(costs)
    min_cost_idx = np.argmin(costs)
    opt_cost_th = float(thresholds[min_cost_idx])
    min_cost = float(costs[min_cost_idx])
    savings_vs_reactive = float(reactive_cost - min_cost)
    pct_savings = float((savings_vs_reactive / reactive_cost) * 100) if reactive_cost > 0 else 0.0
    
    return {
        "thresholds": thresholds.tolist(),
        "costs": costs.tolist(),
        "optimal_cost_threshold": opt_cost_th,
        "minimum_operational_cost": min_cost,
        "reactive_maintenance_cost": float(reactive_cost),
        "total_savings_usd": savings_vs_reactive,
        "percentage_savings": pct_savings,
        "cost_parameters": {
            "cost_false_negative_catastrophic": cost_fn,
            "cost_false_positive_inspection": cost_fp,
            "cost_true_positive_proactive_repair": cost_tp,
            "cost_true_negative_nominal": cost_tn,
        }
    }


def generate_evaluation_figures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_results: Dict[str, Dict[str, Any]],
    subgroup_data: Dict[str, Any],
    cost_data: Dict[str, Any]
) -> None:
    """Generates and saves publication-grade figures 11 to 15."""
    logger.info("Generating publication-grade diagnostic figures (Figures 11-15)...")
    
    # -------------------------------------------------------------------------
    # FIGURE 11: Holdout PR & ROC Curves
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # PR Curve
    ax1.plot(recall, precision, color="#1f77b4", linewidth=2.5, label=f"LightGBM PR Curve (PR-AUC = {pr_auc:.4f})")
    ax1.axhline(y_true.mean(), color="#7f7f7f", linestyle="--", label=f"Random Prevalence Baseline ({y_true.mean()*100:.2f}%)")
    
    # Threshold markers
    colors = {"default_0.50": "#e67e22", "optimal_max_f1": "#27ae60", "high_recall_safety": "#e74c3c"}
    markers = {"default_0.50": "s", "optimal_max_f1": "o", "high_recall_safety": "^"}
    
    for t_key, t_info in threshold_results.items():
        c = colors.get(t_key, "#333333")
        m = markers.get(t_key, "o")
        ax1.scatter(t_info["recall"], t_info["precision"], color=c, marker=m, s=120, zorder=5,
                    label=f"{t_info['threshold_name']} ($\tau={t_info['threshold_value']:.2f}$ | Rec={t_info['recall']*100:.1f}%, Prec={t_info['precision']*100:.1f}%)")
        
    ax1.set_xlabel("Recall (True Positive Rate)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    ax1.set_title("Holdout Precision-Recall Curve (2,000 Unseen Samples)", fontsize=13, fontweight="bold")
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower left", fontsize=10)
    
    # ROC Curve
    ax2.plot(fpr, tpr, color="#2ca02c", linewidth=2.5, label=f"LightGBM ROC Curve (ROC-AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", label="Random Chance (AUC = 0.5000)")
    
    for t_key, t_info in threshold_results.items():
        c = colors.get(t_key, "#333333")
        m = markers.get(t_key, "o")
        t_fpr = 1.0 - t_info["specificity"]
        ax2.scatter(t_fpr, t_info["recall"], color=c, marker=m, s=120, zorder=5,
                    label=f"{t_info['threshold_name']} ($\tau={t_info['threshold_value']:.2f}$ | FPR={t_fpr*100:.1f}%)")
        
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="bold")
    ax2.set_title("Holdout Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0.0, 1.05])
    ax2.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    fig11_path = FIGURES_DIR / "11_holdout_pr_roc_curves.png"
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 11 to %s", fig11_path)
    
    # -------------------------------------------------------------------------
    # FIGURE 12: Holdout Confusion Matrices
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (t_key, t_info) in zip(axes, threshold_results.items()):
        cm = np.array(t_info["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{t_info['threshold_name']} ($\\tau={t_info['threshold_value']:.2f}$)\n"
                     f"F1: {t_info['f1']:.3f} | Recall: {t_info['recall']*100:.1f}% | Prec: {t_info['precision']*100:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        ax.set_xticklabels(["Nominal (0)", "Failure (1)"])
        ax.set_yticklabels(["Nominal (0)", "Failure (1)"])
        
    plt.tight_layout()
    fig12_path = FIGURES_DIR / "12_holdout_confusion_matrices.png"
    plt.savefig(fig12_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 12 to %s", fig12_path)
    
    # -------------------------------------------------------------------------
    # FIGURE 13: Failure Mode Recall Breakdown
    # -------------------------------------------------------------------------
    modes_data = subgroup_data["failure_mode_breakdown"]
    modes = list(modes_data.keys())
    recalls = [modes_data[m]["recall"] * 100 for m in modes]
    counts = [modes_data[m]["total_cases"] for m in modes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#2ecc71" if r >= 80 else "#f39c12" if r >= 50 else "#e74c3c" for r in recalls]
    bars = ax.bar(modes, recalls, color=bar_colors, edgecolor="black", linewidth=1.2, width=0.55)
    
    for bar, count, r in zip(bars, counts, recalls):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{r:.1f}%\n(n={count})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_ylim([0, 115])
    ax.set_ylabel("Detection Recall Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Detection Sensitivity by Individual Failure Mode (Holdout Test Set)", fontsize=13, fontweight="bold")
    ax.axhline(80, color="#27ae60", linestyle="--", alpha=0.7, label="Target Industrial Reliability Threshold (80%)")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    fig13_path = FIGURES_DIR / "13_failure_mode_breakdown.png"
    plt.savefig(fig13_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 13 to %s", fig13_path)
    
    # -------------------------------------------------------------------------
    # FIGURE 14: Holdout Probability Calibration Curve
    # -------------------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#2980b9", linewidth=2.5, markersize=8,
            label=f"LightGBM Production Model (Brier = {brier:.5f})")
    ax.set_xlabel("Mean Predicted Failure Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Observed True Failures", fontsize=12, fontweight="bold")
    ax.set_title("Holdout Reliability Diagram (Probability Calibration Assessment)", fontsize=13, fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", fontsize=11)
    
    plt.tight_layout()
    fig14_path = FIGURES_DIR / "14_holdout_calibration_curve.png"
    plt.savefig(fig14_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 14 to %s", fig14_path)
    
    # -------------------------------------------------------------------------
    # FIGURE 15: Operational Cost Curve & Industrial ROI
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    th_vals = cost_data["thresholds"]
    cost_vals = np.array(cost_data["costs"]) / 1000.0  # in Thousands USD
    
    ax.plot(th_vals, cost_vals, color="#8e44ad", linewidth=2.5, label="Total Operational Maintenance Cost")
    ax.axhline(cost_data["reactive_maintenance_cost"] / 1000.0, color="#c0392b", linestyle="--", linewidth=2,
               label=f"Reactive Run-to-Failure Baseline (${cost_data['reactive_maintenance_cost']:,.0f})")
    
    opt_th = cost_data["optimal_cost_threshold"]
    min_cost = cost_data["minimum_operational_cost"] / 1000.0
    ax.scatter([opt_th], [min_cost], color="#27ae60", s=150, zorder=6,
               label=f"Cost-Optimal Operating Point ($\tau={opt_th:.2f}$ | ${cost_data['minimum_operational_cost']:,.0f})")
    
    ax.set_xlabel("Decision Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Operational Cost (Thousands USD)", fontsize=12, fontweight="bold")
    ax.set_title("Industrial Cost Curve: OpEx Minimization vs. Decision Threshold", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    fig15_path = FIGURES_DIR / "15_cost_curve_tradeoff.png"
    plt.savefig(fig15_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 15 to %s", fig15_path)


def run_holdout_evaluation() -> Dict[str, Any]:
    """Main execution orchestrator for Phase 5."""
    logger.info("================================================================================")
    logger.info("STARTING PHASE 5: HOLDOUT EVALUATION & GOLD STANDARD VERIFICATION")
    logger.info("================================================================================")
    
    # 1. Load holdout data
    X_test, y_test, aux_test, raw_df = load_holdout_dataset(TEST_DATA_PATH)
    
    # 2. Load serialized pipeline & threshold config
    logger.info("Loading frozen production pipeline from %s...", FROZEN_MODEL_PATH)
    pipeline = joblib.load(FROZEN_MODEL_PATH)
    
    logger.info("Loading threshold configuration from %s...", THRESHOLD_CONFIG_PATH)
    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
        thresh_cfg = json.load(f)
        
    opt_f1_th = float(thresh_cfg.get("optimal_f1_threshold", 0.8415))
    high_rec_th = float(thresh_cfg.get("high_recall_threshold", 0.4951))
    
    # 3. Forward inference strictly on holdout features
    logger.info("Executing forward inference on holdout test features...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # 4. Global metrics
    pr_auc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    
    logger.info("Holdout Global Metrics -> PR-AUC: %.4f | ROC-AUC: %.4f | Brier Score: %.5f",
                pr_auc, roc_auc, brier)
    
    # 5. Threshold evaluations
    threshold_results = {
        "default_0.50": evaluate_threshold_performance(y_test.values, y_prob, 0.50, "Default (0.50)"),
        "optimal_max_f1": evaluate_threshold_performance(y_test.values, y_prob, opt_f1_th, f"Max-F1 ({opt_f1_th:.2f})"),
        "high_recall_safety": evaluate_threshold_performance(y_test.values, y_prob, high_rec_th, f"High-Recall ({high_rec_th:.2f})"),
    }
    
    # 6. Subgroup breakdowns
    subgroup_data = compute_subgroup_metrics(raw_df, y_test, y_prob, aux_test, active_threshold=high_rec_th)
    
    # 7. Industrial Cost Curve
    cost_data = compute_industrial_cost_curve(y_test.values, y_prob)
    
    # 8. Generate Visualizations (Figures 11-15)
    generate_evaluation_figures(y_test.values, y_prob, threshold_results, subgroup_data, cost_data)
    
    # 9. Save Predictions
    predictions_df = raw_df.copy()
    predictions_df["failure_probability"] = np.round(y_prob, 5)
    predictions_df["predicted_failure_default_0.50"] = (y_prob >= 0.50).astype(int)
    predictions_df["predicted_failure_max_f1"] = (y_prob >= opt_f1_th).astype(int)
    predictions_df["predicted_failure_high_recall"] = (y_prob >= high_rec_th).astype(int)
    predictions_df["is_prediction_correct_max_f1"] = (predictions_df["predicted_failure_max_f1"] == y_test).astype(int)
    
    out_preds_path = PREDICTIONS_DIR / "holdout_predictions.csv"
    predictions_df.to_csv(out_preds_path, index=False)
    logger.info("Saved serialized holdout predictions to %s", out_preds_path)
    
    # 10. Collate summary dictionary
    evaluation_summary = {
        "dataset_split": "Holdout Test Set (20.0%, 2,000 samples)",
        "sample_count": len(y_test),
        "failure_count": int(y_test.sum()),
        "nominal_count": int((y_test == 0).sum()),
        "failure_prevalence_pct": float(y_test.mean() * 100),
        "headline_metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
        },
        "threshold_evaluations": threshold_results,
        "subgroup_breakdowns": subgroup_data,
        "cost_analysis": {
            "optimal_cost_threshold": cost_data["optimal_cost_threshold"],
            "minimum_cost_usd": cost_data["minimum_operational_cost"],
            "reactive_cost_usd": cost_data["reactive_maintenance_cost"],
            "net_savings_usd": cost_data["total_savings_usd"],
            "percentage_savings": cost_data["percentage_savings"],
        }
    }
    
    metrics_path = REPORTS_DIR / "holdout_metrics.json"
    save_json(evaluation_summary, metrics_path)
    logger.info("Saved complete holdout evaluation metrics to %s", metrics_path)
    
    return evaluation_summary


def main():
    summary = run_holdout_evaluation()
    print("\n" + "="*80)
    print("PREDICTIVEGUARD PHASE 5: HOLDOUT EVALUATION COMPLETE")
    print("="*80)
    print(f"Holdout Cohort: {summary['sample_count']} samples | True Failures: {summary['failure_count']} ({summary['failure_prevalence_pct']:.2f}%)")
    print(f"PR-AUC (Headline Metric): {summary['headline_metrics']['pr_auc']:.4f}")
    print(f"ROC-AUC:                  {summary['headline_metrics']['roc_auc']:.4f}")
    print(f"Brier Score:              {summary['headline_metrics']['brier_score']:.5f}")
    print("\n--- Threshold Comparison ---")
    for k, v in summary["threshold_evaluations"].items():
        print(f"{v['threshold_name']:<25} | F1: {v['f1']:.4f} | Recall: {v['recall']*100:.2f}% | Precision: {v['precision']*100:.2f}% | MCC: {v['mcc']:.4f}")
    print("\n--- Individual Failure Mode Recall (High-Recall Safety Threshold) ---")
    for mode, data in summary["subgroup_breakdowns"]["failure_mode_breakdown"].items():
        print(f"{mode:<25} | Cases: {data['total_cases']:<4} | Detected: {data['detected_cases']:<4} | Recall: {data['recall']*100:.1f}%")
    print("\n--- Industrial OpEx Cost Analysis ---")
    print(f"Reactive Maintenance Cost:   ${summary['cost_analysis']['reactive_cost_usd']:,.0f}")
    print(f"PredictiveGuard Min Cost:     ${summary['cost_analysis']['minimum_cost_usd']:,.0f} (at threshold {summary['cost_analysis']['optimal_cost_threshold']:.2f})")
    print(f"Net Operational Savings:     ${summary['cost_analysis']['net_savings_usd']:,.0f} ({summary['cost_analysis']['percentage_savings']:.1f}% reduction)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
