"""
Comprehensive EDA & Physics Verification Script.
Generates all high-resolution figures for reports/figures and quantifies all 17 requirements.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "ai4i2020.csv"
FIG_DIR = ROOT_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def run():
    print("Loading AI4I 2020 dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Dataset Shape
    print(f"Shape: {df.shape}")
    
    # 2. Data Types
    print("Data types:\n", df.dtypes)
    
    # 3. Missing values
    missing = df.isnull().sum()
    print("Missing values:\n", missing[missing > 0])
    
    # 4. Duplicate rows
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # 5. Class Distribution
    fail_counts = df['Machine failure'].value_counts()
    print(f"Class distribution: 0={fail_counts[0]} ({fail_counts[0]/len(df):.2%}), 1={fail_counts[1]} ({fail_counts[1]/len(df):.2%})")
    
    # 6. Failure rate by Type
    type_fail = df.groupby('Type')['Machine failure'].agg(['count', 'sum', 'mean'])
    print("Failure rate by Type:\n", type_fail)
    
    # Physics derived variables
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power_w'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2.0 * np.pi / 60.0)
    df['wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
    
    # Overstrain thresholds
    osf_limits = {'L': 11000.0, 'M': 12000.0, 'H': 13000.0}
    df['osf_threshold'] = df['Type'].map(osf_limits)
    df['osf_ratio'] = df['wear_torque'] / df['osf_threshold']
    
    # Figure 1: Class and Failure Mode Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1A: Machine failure by Type
    sns.countplot(data=df, x='Type', hue='Machine failure', palette=['#2b5c8f', '#d9534f'], ax=axes[0])
    axes[0].set_title("Machine Failure Count by Machine Type (L, M, H)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Machine Type (L=Low, M=Medium, H=High variant)")
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    
    # 1B: Failure mode breakdown
    modes = ['HDF', 'OSF', 'PWF', 'TWF', 'RNF']
    mode_counts = df[modes].sum().sort_values(ascending=False)
    sns.barplot(x=mode_counts.index, y=mode_counts.values, palette='viridis', ax=axes[1])
    axes[1].set_title("Breakdown of Underlying Failure Modes (Data Leakage Flags)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Failure Mode Flag")
    axes[1].set_ylabel("Count of Occurrences")
    for i, v in enumerate(mode_counts.values):
        axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_and_mode_distribution.png")
    plt.close()
    print("Saved Figure 1.")

    # Figure 2: Sensor Distributions split by Machine failure
    features_to_plot = [
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'temp_diff'
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, feat in enumerate(features_to_plot):
        sns.kdeplot(data=df[df['Machine failure'] == 0], x=feat, ax=axes[idx], label='No Failure (0)', color='#1f77b4', fill=True, alpha=0.3)
        sns.kdeplot(data=df[df['Machine failure'] == 1], x=feat, ax=axes[idx], label='Failure (1)', color='#d62728', fill=True, alpha=0.5)
        axes[idx].set_title(f"Distribution: {feat}", fontsize=11, fontweight='bold')
        axes[idx].legend(loc='best')
    plt.suptitle("Raw & Differential Sensor Distributions by Machine Failure Status", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_sensor_distributions_by_failure.png")
    plt.close()
    print("Saved Figure 2.")

    # Figure 3: Correlation Matrix
    corr_cols = [
        'Air temperature [K]', 'Process temperature [K]', 'temp_diff',
        'Rotational speed [rpm]', 'Torque [Nm]', 'power_w',
        'Tool wear [min]', 'wear_torque', 'Machine failure',
        'HDF', 'PWF', 'OSF', 'TWF', 'RNF'
    ]
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True, square=True)
    plt.title("Correlation Matrix: Sensor Signals, Physics Features & Failure Modes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_correlation_heatmap.png")
    plt.close()
    print("Saved Figure 3.")

    # Figure 4: HDF Region (Heat Dissipation Failure)
    # HDF occurs when: temp_diff < 8.6 K AND rotational_speed < 1380 rpm
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['Rotational speed [rpm]'], df['temp_diff'],
        c=df['HDF'].map({0: '#b0bec5', 1: '#d32f2f'}),
        alpha=df['HDF'].map({0: 0.25, 1: 0.9}),
        s=df['HDF'].map({0: 15, 1: 50}),
        edgecolors='none'
    )
    plt.axhline(8.6, color='black', linestyle='--', linewidth=1.5, label='HDF Temp Diff Boundary (8.6 K)')
    plt.axvline(1380, color='darkblue', linestyle='--', linewidth=1.5, label='HDF Speed Boundary (1380 rpm)')
    plt.axvspan(1100, 1380, ymin=0, ymax=(8.6 - df['temp_diff'].min()) / (df['temp_diff'].max() - df['temp_diff'].min()), 
                color='red', alpha=0.15, label='Theoretical HDF Region')
    plt.title("HDF Region Verification: Temperature Difference vs. Rotational Speed", fontsize=13, fontweight='bold')
    plt.xlabel("Rotational speed [rpm]")
    plt.ylabel("Process temp - Air temp [K]")
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_hdf_region_tempdiff_vs_rpm.png")
    plt.close()
    print("Saved Figure 4.")

    # Figure 5: PWF Region (Power Failure)
    # Power < 3500 W OR Power > 9000 W
    plt.figure(figsize=(11, 5))
    sns.histplot(df[df['PWF'] == 0]['power_w'], bins=60, color='#1f77b4', alpha=0.4, label='Normal Power (PWF=0)', stat='density')
    sns.histplot(df[df['PWF'] == 1]['power_w'], bins=40, color='#d62728', alpha=0.8, label='Power Failure (PWF=1)', stat='density')
    plt.axvline(3500, color='red', linestyle='--', linewidth=2, label='Lower Limit: 3500 W')
    plt.axvline(9000, color='red', linestyle='--', linewidth=2, label='Upper Limit: 9000 W')
    plt.axvspan(0, 3500, color='red', alpha=0.1, label='Low Power Cutoff')
    plt.axvspan(9000, 14000, color='red', alpha=0.1, label='High Power Cutoff')
    plt.title("PWF Region Verification: Mechanical Power Distribution and Physical Bounds", fontsize=13, fontweight='bold')
    plt.xlabel("Calculated Mechanical Power [W] = Torque × RPM × 2π / 60")
    plt.ylabel("Density")
    plt.xlim(0, 13000)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_pwf_region_power_distribution.png")
    plt.close()
    print("Saved Figure 5.")

    # Figure 6: OSF Region (Overstrain Failure) by Type
    # Wear * Torque > Threshold (L: 11000, M: 12000, H: 13000)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    types = ['L', 'M', 'H']
    colors = {'L': '#3498db', 'M': '#2ecc71', 'H': '#9b59b6'}
    for idx, t in enumerate(types):
        sub = df[df['Type'] == t]
        thresh = osf_limits[t]
        axes[idx].scatter(sub['Tool wear [min]'], sub['wear_torque'], 
                          c=sub['OSF'].map({0: '#b0bec5', 1: '#e74c3c'}),
                          alpha=sub['OSF'].map({0: 0.25, 1: 0.9}),
                          s=sub['OSF'].map({0: 15, 1: 45}))
        axes[idx].axhline(thresh, color='black', linestyle='--', linewidth=2, label=f'Threshold ({thresh:.0f} min·Nm)')
        axes[idx].set_title(f"Type {t}: Wear × Torque (Threshold = {thresh:.0f})", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Tool wear [min]")
        if idx == 0:
            axes[idx].set_ylabel("Wear × Torque [min·Nm]")
        axes[idx].legend(loc='upper left')
    plt.suptitle("OSF Region Verification: Overstrain Thresholds Across Product Variants", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_osf_region_wear_torque_by_type.png")
    plt.close()
    print("Saved Figure 6.")

    # Figure 7: Temperature Random Walk & Sequence Order
    plt.figure(figsize=(14, 5))
    plt.plot(df['UDI'][:1000], df['Air temperature [K]'][:1000], label='Air Temperature [K]', color='#3498db', linewidth=1.2)
    plt.plot(df['UDI'][:1000], df['Process temperature [K]'][:1000], label='Process Temperature [K]', color='#e67e22', linewidth=1.2)
    plt.title("Sensor Sequence Analysis: Temporal Random-Walk Behavior in Temperatures (First 1,000 Rows)", fontsize=13, fontweight='bold')
    plt.xlabel("Row Sequence (UDI)")
    plt.ylabel("Temperature [K]")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_temperature_random_walk_row_order.png")
    plt.close()
    print("Saved Figure 7.")

    # Figure 8: Validation Strategy Comparison: Stratified vs Time-Series
    # Run repeated cross-validation to get robust distributions
    X = df[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].copy()
    X['temp_diff'] = df['temp_diff']
    X['power_w'] = df['power_w']
    X['wear_torque'] = df['wear_torque']
    X['Type_L'] = (df['Type'] == 'L').astype(int)
    X['Type_M'] = (df['Type'] == 'M').astype(int)
    y = df['Machine failure']

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    strat_prauc = []
    strat_f1 = []
    for tr, te in rskf.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        probs = clf.predict_proba(X.iloc[te])[:, 1]
        strat_prauc.append(average_precision_score(y.iloc[te], probs))
        strat_f1.append(f1_score(y.iloc[te], (probs >= 0.5).astype(int), zero_division=0))

    tscv = TimeSeriesSplit(n_splits=5)
    time_prauc = []
    time_f1 = []
    for tr, te in tscv.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        probs = clf.predict_proba(X.iloc[te])[:, 1]
        time_prauc.append(average_precision_score(y.iloc[te], probs))
        time_f1.append(f1_score(y.iloc[te], (probs >= 0.5).astype(int), zero_division=0))

    val_df = pd.DataFrame({
        'PR-AUC': strat_prauc + time_prauc,
        'Strategy': ['Stratified K-Fold (5x3)'] * len(strat_prauc) + ['TimeSeriesSplit (5-split)'] * len(time_prauc)
    })
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=val_df, x='Strategy', y='PR-AUC', palette=['#2ecc71', '#e74c3c'], width=0.4)
    sns.stripplot(data=val_df, x='Strategy', y='PR-AUC', color='black', alpha=0.6, jitter=0.2)
    plt.title(f"Validation Strategy Comparison: PR-AUC\nStratified: {np.mean(strat_prauc):.3f}±{np.std(strat_prauc):.3f} vs Time-Series: {np.mean(time_prauc):.3f}±{np.std(time_prauc):.3f}", 
              fontsize=12, fontweight='bold')
    plt.ylabel("PR-AUC (Precision-Recall Area)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_cv_strategy_comparison.png")
    plt.close()
    print("Saved Figure 8.")
    
    # Save numerical summary
    stats_path = ROOT_DIR / "reports" / "eda_summary.json"
    import json
    summary = {
        "n_samples": len(df),
        "n_features": df.shape[1],
        "failure_rate": float(df['Machine failure'].mean()),
        "mode_counts": {m: int(df[m].sum()) for m in modes},
        "type_failure_rates": {t: float(df[df['Type'] == t]['Machine failure'].mean()) for t in ['L', 'M', 'H']},
        "temp_autocorr_lag1": float(df['Air temperature [K]'].autocorr(1)),
        "rpm_autocorr_lag1": float(df['Rotational speed [rpm]'].autocorr(1)),
        "torque_autocorr_lag1": float(df['Torque [Nm]'].autocorr(1)),
        "stratified_prauc_mean": float(np.mean(strat_prauc)),
        "stratified_prauc_std": float(np.std(strat_prauc)),
        "timeseries_prauc_mean": float(np.mean(time_prauc)),
        "timeseries_prauc_std": float(np.std(time_prauc))
    }
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print("EDA script finished successfully.")

if __name__ == '__main__':
    run()
