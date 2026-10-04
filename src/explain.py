"""
PredictiveGuard Model Interpretability & Explainability Module.
ALGOTHON'26 — Track: ALG-DATA-02 "Predict What Happens Next"

Provides transparent, actionable, and physically grounded explanations:
1. Global Feature Importance via TreeSHAP (consistent, additive feature attribution).
2. Permutation Feature Importance (evaluating drop in PR-AUC / ROC-AUC on development data).
3. Tree-based Native Feature Gain & Split Importance comparison.
4. Publication-grade diagnostic figures:
   - Figure 16: SHAP Summary (Beeswarm) Plot
   - Figure 17: SHAP Feature Dependence & Physical Interaction Plots
   - Figure 18: Local Waterfall Diagnostic Plots for 4 Industrial Case Studies
5. Operator-facing Local Inference Explanations:
   - Translates raw sensor SHAP attributions into domain-specific maintenance guidance.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    TRAIN_DATA_PATH,
    FROZEN_MODEL_PATH,
    THRESHOLD_CONFIG_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    ALL_INFERENCE_FEATURES,
    CANONICAL_NUMERIC_FEATURES,
)
from src.data import load_data, normalize_column_names, remove_identifiers, remove_leakage_columns, validate_schema
from src.utils import logger, save_json

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def get_pipeline_feature_names(pipeline: Any) -> List[str]:
    """Extracts ordered transformed feature names from the fitted preprocessing pipeline."""
    prep = pipeline.named_steps["preprocessing"]
    ct = prep.named_steps["preprocessor"]
    
    # Numeric features
    num_cols = ct.transformers_[0][2]
    
    # One-hot encoded categorical features
    ohe = ct.named_transformers_["cat"].named_steps["encoder"]
    cat_categories = ohe.categories_[0]
    cat_names = [f"Type_{c}" for c in cat_categories]
    
    return list(num_cols) + cat_names


def explain_sample_human_readable(
    feature_names: List[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    base_value: float,
    pred_prob: float,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Translates mathematical SHAP values into an intuitive, operator-ready
    maintenance diagnostic report.
    """
    contributions = []
    for name, val, sv in zip(feature_names, feature_values, shap_values):
        contributions.append({
            "feature": name,
            "value": float(val),
            "shap_value": float(sv),
            "abs_shap": float(abs(sv)),
        })
        
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)
    
    risk_drivers = [c for c in contributions if c["shap_value"] > 0][:top_k]
    mitigating_factors = [c for c in contributions if c["shap_value"] < 0][:top_k]
    
    # Determine actionable industrial recommendation (Decision Support)
    top_driver_name = risk_drivers[0]["feature"] if risk_drivers else ""
    if "overstrain" in top_driver_name or "wear_torque" in top_driver_name:
        action = "CRITICAL (Decision Support): Operating telemetry is consistent with tool overstrain. Inspect cutting insert and tool holder; replacement may be required."
    elif "low_speed_low_tempdiff" in top_driver_name or "temp" in top_driver_name:
        action = "WARNING (Decision Support): Operating telemetry indicates potential convective cooling deficit. Inspect coolant flow, radiator fan, and clean thermal heat sink."
    elif "power" in top_driver_name:
        action = "WARNING (Decision Support): Mechanical power draw is outside the safe operating band (3.5kW - 9.0kW). Check spindle motor drive, inverter, and mechanical transmission."
    elif "wear" in top_driver_name:
        action = "ADVISORY (Decision Support): Tool wear duration is approaching the critical replacement boundary (200 min). Plan scheduled tool inspection and possible swap."
    else:
        action = "NOMINAL (Decision Support): Machine operating within verified safe physical envelopes. Continue standard monitoring."

        
    return {
        "failure_probability": float(pred_prob),
        "base_value": float(base_value),
        "dominant_risk_driver": top_driver_name,
        "recommended_maintenance_action": action,
        "top_risk_drivers": risk_drivers,
        "top_mitigating_factors": mitigating_factors,
    }


def compute_global_interpretability(
    sample_size: int = 2000
) -> Dict[str, Any]:
    """
    Executes full global interpretability workflow:
    1. TreeSHAP values across development data.
    2. Permutation feature importance.
    3. LightGBM native split and gain importance.
    4. Generates Figures 16, 17, 18.
    """
    logger.info("Starting Phase 6 Interpretability & Explainability analysis...")
    
    # 1. Load Model & Data
    pipeline = joblib.load(FROZEN_MODEL_PATH)
    prep = pipeline.named_steps["preprocessing"]
    clf = pipeline.named_steps["classifier"]
    
    X_dev, y_dev, _ = load_data(TRAIN_DATA_PATH, require_target=True, drop_leakage=True)
    
    # Sample development data for SHAP computation
    if len(X_dev) > sample_size:
        np.random.seed(RANDOM_SEED)
        sample_indices = np.random.choice(len(X_dev), size=sample_size, replace=False)
        X_sample = X_dev.iloc[sample_indices].copy()
        y_sample = y_dev.iloc[sample_indices].copy()
    else:
        X_sample = X_dev.copy()
        y_sample = y_dev.copy()
        
    logger.info("Transforming %d samples through preprocessing pipeline...", len(X_sample))
    X_proc = prep.transform(X_sample)
    feature_names = get_pipeline_feature_names(pipeline)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=X_sample.index)
    
    # 2. Compute TreeSHAP Values
    logger.info("Computing TreeSHAP values using shap.TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    shap_explanation = explainer(X_proc_df)
    shap_values = shap_explanation.values
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
    else:
        base_value = float(base_value)
        
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    logger.info("Top 5 features by Mean |SHAP|:\n%s", shap_importance_df.head(5).to_string(index=False))
    
    # 3. Compute Permutation Feature Importance
    logger.info("Computing Permutation Feature Importance (PR-AUC metric drop)...")
    perm_result = permutation_importance(
        clf, X_proc, y_sample,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    perm_importance_df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std": perm_result.importances_std,
    }).sort_values(by="perm_importance_mean", ascending=False)
    
    # 4. Extract Native LightGBM Importance
    native_gain = clf.booster_.feature_importance(importance_type="gain")
    native_split = clf.booster_.feature_importance(importance_type="split")
    native_df = pd.DataFrame({
        "feature": feature_names,
        "native_gain": native_gain,
        "native_split": native_split,
    })
    
    # Merge Importance Metrics into Unified Table
    unified_importance = shap_importance_df.merge(perm_importance_df, on="feature").merge(native_df, on="feature")
    unified_importance = unified_importance.sort_values(by="mean_abs_shap", ascending=False)
    
    importance_csv_path = REPORTS_DIR / "feature_importance_comparison.csv"
    unified_importance.to_csv(importance_csv_path, index=False)
    logger.info("Saved unified feature importance table to %s", importance_csv_path)

    # -------------------------------------------------------------------------
    # FIGURE 16: SHAP Summary (Beeswarm) Plot
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 16: SHAP Summary Beeswarm Plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_proc_df,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_type="dot"
    )
    plt.title("TreeSHAP Global Feature Attribution & Directionality (Top 15 Features)",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("SHAP Value (Impact on Log-Odds of Machine Failure)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig16_path = FIGURES_DIR / "16_shap_summary_beeswarm.png"
    plt.savefig(fig16_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 16 to %s", fig16_path)

    # -------------------------------------------------------------------------
    # FIGURE 17: SHAP Feature Dependence & Physical Interactions
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 17: SHAP Dependence & Physical Interactions...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Overstrain Ratio (showing sharp jump at capacity = 1.0)
    idx_os = feature_names.index("overstrain_ratio")
    scatter_os = axes[0, 0].scatter(
        X_proc_df["overstrain_ratio"], shap_values[:, idx_os],
        c=X_proc_df["Torque [Nm]"], cmap="coolwarm", alpha=0.75, s=25
    )
    axes[0, 0].axvline(1.0, color="#e74c3c", linestyle="--", linewidth=1.8, label="Structural Capacity Limit (1.0)")
    axes[0, 0].set_xlabel("Overstrain Ratio [Tool Wear * Torque / Capacity]", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("SHAP Value for Overstrain Ratio", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("SHAP Dependence: Overstrain Ratio (Color: Torque [Nm])", fontsize=12, fontweight="bold")
    cbar1 = plt.colorbar(scatter_os, ax=axes[0, 0])
    cbar1.set_label("Torque [Nm]", fontsize=10)
    axes[0, 0].legend(loc="upper left")

    # Subplot 2: Low Speed Low Temp Diff Interaction (HDF Region)
    idx_hdf = feature_names.index("low_speed_low_tempdiff")
    scatter_hdf = axes[0, 1].scatter(
        X_proc_df["low_speed_low_tempdiff"], shap_values[:, idx_hdf],
        c=X_proc_df["temp_diff"], cmap="plasma_r", alpha=0.75, s=25
    )
    axes[0, 1].set_xlabel("Low Speed Low TempDiff Severity [Deficit Index]", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("SHAP Value for HDF Interaction", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("SHAP Dependence: Thermal Entrapment Index (Color: Temp Diff [K])", fontsize=12, fontweight="bold")
    cbar2 = plt.colorbar(scatter_hdf, ax=axes[0, 1])
    cbar2.set_label("Temp Diff [K]", fontsize=10)

    # Subplot 3: Mechanical Power (PWF Safety Envelope)
    idx_pwr = feature_names.index("power_w")
    scatter_pwr = axes[1, 0].scatter(
        X_proc_df["power_w"], shap_values[:, idx_pwr],
        c=X_proc_df["Rotational speed [rpm]"], cmap="viridis", alpha=0.75, s=25
    )
    axes[1, 0].axvspan(3500, 9000, color="#2ecc71", alpha=0.15, label="Verified Safe Power Band (3.5 - 9.0 kW)")
    axes[1, 0].set_xlabel("Calculated Mechanical Power [W]", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("SHAP Value for Power [W]", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("SHAP Dependence: Mechanical Power (Color: Speed [rpm])", fontsize=12, fontweight="bold")
    cbar3 = plt.colorbar(scatter_pwr, ax=axes[1, 0])
    cbar3.set_label("Rotational Speed [rpm]", fontsize=10)
    axes[1, 0].legend(loc="upper right")

    # Subplot 4: Tool Wear (Linear Accumulation & Critical Transition)
    idx_wear = feature_names.index("Tool wear [min]")
    scatter_wear = axes[1, 1].scatter(
        X_proc_df["Tool wear [min]"], shap_values[:, idx_wear],
        c=X_proc_df["overstrain_ratio"], cmap="magma", alpha=0.75, s=25
    )
    axes[1, 1].axvline(200, color="#f39c12", linestyle=":", linewidth=1.8, label="TWF Early Warning (200 min)")
    axes[1, 1].axvline(240, color="#c0392b", linestyle="--", linewidth=1.8, label="TWF Maximum Endurance (240 min)")
    axes[1, 1].set_xlabel("Tool Wear [min]", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("SHAP Value for Tool Wear", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("SHAP Dependence: Tool Wear Duration (Color: Overstrain Ratio)", fontsize=12, fontweight="bold")
    cbar4 = plt.colorbar(scatter_wear, ax=axes[1, 1])
    cbar4.set_label("Overstrain Ratio", fontsize=10)
    axes[1, 1].legend(loc="upper left")

    plt.tight_layout()
    fig17_path = FIGURES_DIR / "17_shap_dependence_interactions.png"
    plt.savefig(fig17_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 17 to %s", fig17_path)

    # -------------------------------------------------------------------------
    # FIGURE 18: Local Waterfall Case Studies (4 Operational Scenarios)
    # -------------------------------------------------------------------------
    logger.info("Generating Figure 18: Local Waterfall Case Studies for 4 Operational Scenarios...")
    raw_full = pd.read_csv(TRAIN_DATA_PATH)
    
    # Representative case study row indices:
    # Row 25: HDF Failure (Heat Dissipation)
    # Row 34: PWF Failure (Power Breakdown)
    # Row 35: OSF Failure (Overstrain Breakdown)
    # Row 0:  Nominal Machine Operation
    case_indices = [25, 34, 35, 0]
    case_titles = [
        "Case Study 1: Heat Dissipation Failure (HDF)\n[Low Speed & Impaired Thermal Convection]",
        "Case Study 2: Mechanical Power Failure (PWF)\n[Excessive Power Draw > 9,000 W]",
        "Case Study 3: Overstrain Failure (OSF)\n[Combined High Wear & Cutting Torque]",
        "Case Study 4: Nominal Machine Operation\n[Continuous Normal Production within Safe Bands]"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    case_study_reports = {}
    
    for ax, row_idx, title in zip(axes_flat, case_indices, case_titles):
        row_raw = X_dev.iloc[row_idx:row_idx+1]
        row_proc = prep.transform(row_raw)[0]
        
        # Calculate local SHAP values for this specific row
        row_shap = explainer(pd.DataFrame([row_proc], columns=feature_names)).values[0]
        row_prob = pipeline.predict_proba(row_raw)[0, 1]
        
        # Sort features by absolute contribution
        sorted_indices = np.argsort(np.abs(row_shap))[::-1][:8]
        top_names = [feature_names[i] for i in sorted_indices]
        top_shaps = [row_shap[i] for i in sorted_indices]
        
        # Invert order for horizontal bar chart (top at the top)
        top_names = top_names[::-1]
        top_shaps = top_shaps[::-1]
        
        bar_colors = ["#e74c3c" if s > 0 else "#27ae60" for s in top_shaps]
        bars = ax.barh(top_names, top_shaps, color=bar_colors, edgecolor="black", linewidth=1.0)
        
        # Label values on bars
        for bar, val in zip(bars, top_shaps):
            xpos = val + (0.05 if val >= 0 else -0.05)
            ha = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height()/2.0, f"{val:+.3f}",
                    ha=ha, va="center", fontsize=9, fontweight="bold")
            
        ax.axvline(0, color="black", linestyle="-", linewidth=1.2)
        ax.set_title(f"{title}\nPredicted Failure Probability: {row_prob*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Impact on Failure Log-Odds", fontsize=10, fontweight="bold")
        
        # Generate diagnostic text
        explanation_diag = explain_sample_human_readable(
            feature_names, row_proc, row_shap, base_value, row_prob, top_k=3
        )
        case_study_reports[f"row_{row_idx}"] = {
            "title": title.split("\n")[0],
            "raw_inputs": X_dev.iloc[row_idx].to_dict(),
            "explanation": explanation_diag,
        }

    plt.tight_layout()
    fig18_path = FIGURES_DIR / "18_shap_waterfall_case_studies.png"
    plt.savefig(fig18_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 18 to %s", fig18_path)

    # 5. Collate Summary Dictionary
    interpretability_summary = {
        "global_importance_ranking": unified_importance.to_dict(orient="records"),
        "top_features_summary": {
            "top_1": unified_importance.iloc[0]["feature"],
            "top_2": unified_importance.iloc[1]["feature"],
            "top_3": unified_importance.iloc[2]["feature"],
            "top_4": unified_importance.iloc[3]["feature"],
            "top_5": unified_importance.iloc[4]["feature"],
        },
        "case_studies": case_study_reports,
    }
    
    summary_path = REPORTS_DIR / "interpretability_summary.json"
    save_json(interpretability_summary, summary_path)
    logger.info("Saved complete interpretability summary to %s", summary_path)
    
    return interpretability_summary


def explain_single_record(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Production callable utility for single-record explainability."""
    pipeline = joblib.load(FROZEN_MODEL_PATH)
    prep = pipeline.named_steps["preprocessing"]
    clf = pipeline.named_steps["classifier"]
    
    df_raw = pd.DataFrame([input_dict])
    df_norm = normalize_column_names(df_raw)
    clean_df, _ = remove_identifiers(df_norm)
    clean_df, _ = remove_leakage_columns(clean_df)
    clean_features = clean_df[ALL_INFERENCE_FEATURES].copy()
    
    proc = prep.transform(clean_features)[0]
    feature_names = get_pipeline_feature_names(pipeline)
    
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer(pd.DataFrame([proc], columns=feature_names)).values[0]
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)
        
    prob = pipeline.predict_proba(clean_features)[0, 1]
    
    return explain_sample_human_readable(
        feature_names, proc, shap_vals, base_val, prob, top_k=5
    )


def main():
    parser = argparse.ArgumentParser(description="PredictiveGuard Model Interpretability Module")
    parser.add_argument("--global-explain", action="store_true", default=True,
                        help="Run full global interpretability analysis and produce Figures 16-18")
    parser.add_argument("--sample-size", type=int, default=2000, help="Sample size for SHAP estimation")
    args = parser.parse_args()

    summary = compute_global_interpretability(sample_size=args.sample_size)
    print("\n" + "="*80)
    print("PREDICTIVEGUARD PHASE 6: INTERPRETABILITY & SHAP ANALYSIS COMPLETE")
    print("="*80)
    print("Top 5 Features by Mean Absolute TreeSHAP Attribution:")
    for rank, (k, feat) in enumerate(summary["top_features_summary"].items(), 1):
        print(f"  {rank}. {feat}")
    print("\nCase Studies Generated in reports/figures/18_shap_waterfall_case_studies.png:")
    for k, v in summary["case_studies"].items():
        print(f"  - {v['title']}: Predicted Probability = {v['explanation']['failure_probability']*100:.1f}%")
        print(f"    Action: {v['explanation']['recommended_maintenance_action']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
