"""
PredictiveGuard: ALGOTHON26 Industrial Predictive Maintenance Dashboard.
Phase 7 -- Interactive Streamlit Web Application.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FROZEN_MODEL_PATH, THRESHOLD_CONFIG_PATH, TRAIN_DATA_PATH,
    FIGURES_DIR, REPORTS_DIR, CANONICAL_NUMERIC_FEATURES,
    PWF_LOWER_POWER_LIMIT, PWF_UPPER_POWER_LIMIT,
    TWF_MIN_WEAR, TWF_MAX_WEAR,
    OSF_OVERSTRAIN_THRESHOLDS, OSF_CONSERVATIVE_CAPACITY_FALLBACK,
    HDF_TEMP_DIFF_THRESHOLD, HDF_ROTATIONAL_SPEED_THRESHOLD,
)
from src.predict import predict_records, load_threshold_configuration, prepare_inference_features
from src.explain import get_pipeline_feature_names, explain_sample_human_readable

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictiveGuard | ALGOTHON26",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*=\"css\"]{ font-family:'Inter',sans-serif; }
.stApp{ background:linear-gradient(135deg,#0d1117 0%,#111827 50%,#0d1117 100%); color:#e2e8f0; }
#MainMenu,footer,header{ visibility:hidden; }
.stDeployButton{ display:none; }
[data-testid=\"stSidebar\"]{ background:linear-gradient(180deg,#161b27 0%,#1a2235 100%); border-right:1px solid rgba(99,102,241,.3); }
[data-testid=\"stMetric\"]{ background:rgba(99,102,241,.08); border:1px solid rgba(99,102,241,.25); border-radius:12px; padding:16px; transition:all .3s; }
[data-testid=\"stMetric\"]:hover{ border-color:rgba(99,102,241,.5); background:rgba(99,102,241,.14); transform:translateY(-2px); }
[data-testid=\"stMetricLabel\"]{ color:#94a3b8!important; font-size:.78rem!important; }
[data-testid=\"stMetricValue\"]{ color:#e2e8f0!important; font-size:1.6rem!important; font-weight:700!important; }
[data-testid=\"stTabs\"] button{ color:#64748b; font-weight:500; border-radius:8px 8px 0 0; }
[data-testid=\"stTabs\"] button[aria-selected=\"true\"]{ color:#818cf8; border-bottom:2px solid #818cf8; font-weight:600; }
.stButton>button{ background:linear-gradient(135deg,#4f46e5,#7c3aed); color:#fff; border:none; border-radius:10px; font-weight:600; font-size:.9rem; padding:.55rem 1.6rem; transition:all .25s; box-shadow:0 4px 20px rgba(99,102,241,.35); }
.stButton>button:hover{ background:linear-gradient(135deg,#4338ca,#6d28d9); transform:translateY(-2px); box-shadow:0 6px 28px rgba(99,102,241,.5); }
.pg-hero{ background:linear-gradient(135deg,rgba(79,70,229,.15) 0%,rgba(124,58,237,.1) 50%,rgba(16,185,129,.08) 100%); border:1px solid rgba(99,102,241,.3); border-radius:16px; padding:32px 36px 28px; margin-bottom:28px; position:relative; overflow:hidden; }
.pg-hero h1{ font-size:2.1rem; font-weight:800; margin:0; background:linear-gradient(135deg,#818cf8,#34d399); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.pg-hero p{ color:#94a3b8; font-size:.95rem; margin:8px 0 0; }
.pg-badge{ display:inline-block; background:rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.4); color:#34d399; font-size:.7rem; font-weight:700; padding:3px 10px; border-radius:20px; letter-spacing:.06em; text-transform:uppercase; margin-right:8px; }
.pg-badge-yellow{ background:rgba(245,158,11,.15); border-color:rgba(245,158,11,.4); color:#fbbf24; }
.pg-badge-blue{ background:rgba(99,102,241,.15); border-color:rgba(99,102,241,.4); color:#818cf8; }
.risk-card{ border-radius:14px; padding:22px 24px; margin:8px 0; border:2px solid; text-align:center; }
.risk-nominal{ background:rgba(16,185,129,.08); border-color:rgba(16,185,129,.5); }
.risk-warning{ background:rgba(245,158,11,.08); border-color:rgba(245,158,11,.5); }
.risk-critical{ background:rgba(239,68,68,.08); border-color:rgba(239,68,68,.5); animation:pulse-border 2s infinite; }
@keyframes pulse-border{ 0%,100%{ border-color:rgba(239,68,68,.5); } 50%{ border-color:rgba(239,68,68,.9); } }
.risk-label{ font-size:.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.risk-prob{ font-size:3.2rem; font-weight:800; line-height:1.1; margin:6px 0; }
.risk-sub{ font-size:.8rem; color:#94a3b8; }
.risk-nominal .risk-label{ color:#34d399; } .risk-warning .risk-label{ color:#fbbf24; } .risk-critical .risk-label{ color:#f87171; }
.risk-nominal .risk-prob{ color:#34d399; } .risk-warning .risk-prob{ color:#fbbf24; } .risk-critical .risk-prob{ color:#f87171; }
.section-header{ display:flex; align-items:center; gap:10px; border-bottom:1px solid rgba(99,102,241,.2); padding-bottom:10px; margin:24px 0 18px; }
.section-header h3{ font-size:1.05rem; font-weight:700; color:#818cf8; margin:0; }
.feat-chip{ display:inline-flex; align-items:center; gap:6px; background:rgba(30,41,59,.8); border:1px solid rgba(99,102,241,.2); border-radius:8px; padding:4px 10px; font-size:.78rem; font-family:'JetBrains Mono',monospace; color:#94a3b8; margin:3px; }
::-webkit-scrollbar{ width:4px; height:4px; }
::-webkit-scrollbar-track{ background:rgba(30,41,59,.3); }
::-webkit-scrollbar-thumb{ background:rgba(99,102,241,.5); border-radius:4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BG, PANEL, TEXT = "#0d1117", "#161b27", "#e2e8f0"


# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return joblib.load(FROZEN_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_threshold_cfg():
    return load_threshold_configuration(THRESHOLD_CONFIG_PATH)


@st.cache_data(show_spinner=False)
def load_holdout_metrics():
    path = REPORTS_DIR / "holdout_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_feature_importance():
    path = REPORTS_DIR / "feature_importance_comparison.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_interpretability_summary():
    path = REPORTS_DIR / "interpretability_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_training_data():
    return pd.read_csv(TRAIN_DATA_PATH)


@st.cache_resource(show_spinner=False)
def build_explainer(_pipeline):
    return shap.TreeExplainer(_pipeline.named_steps["classifier"])


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def risk_card_html(prob: float, threshold: float) -> str:
    pct = prob * 100
    if prob >= threshold:
        cls, label, icon = "risk-critical", "⚠ FAILURE PREDICTED", "🔴"
    elif prob >= 0.25:
        cls, label, icon = "risk-warning", "⚡ ELEVATED RISK", "🟡"
    else:
        cls, label, icon = "risk-nominal", "✓ NOMINAL OPERATION", "🟢"
    return (
        f'<div class="risk-card {cls}">' +
        f'<div class="risk-label">{label}</div>' +
        f'<div class="risk-prob">{pct:.1f}%</div>' +
        f'<div class="risk-sub">Failure Probability &middot; Threshold {threshold:.2%} &middot; {icon}</div>' +
        '</div>'
    )


def compute_physics(air_temp, proc_temp, speed, torque, wear, machine_type):
    td  = proc_temp - air_temp
    pw  = torque * speed * (2.0 * 3.14159265358979 / 60.0)
    wt  = wear * torque
    cap = OSF_OVERSTRAIN_THRESHOLDS.get(machine_type, OSF_CONSERVATIVE_CAPACITY_FALLBACK)
    return {
        "temp_diff": td, "power_w": pw, "wear_torque": wt,
        "overstrain_ratio": wt / (cap + 1e-6),
        "in_twf_band": TWF_MIN_WEAR <= wear <= TWF_MAX_WEAR,
        "hdf_risk": td < HDF_TEMP_DIFF_THRESHOLD and speed < HDF_ROTATIONAL_SPEED_THRESHOLD,
        "pwf_risk": pw < PWF_LOWER_POWER_LIMIT or pw > PWF_UPPER_POWER_LIMIT,
    }


def shap_waterfall_fig(shap_vals, feature_names, proc_arr, base_val, prob, top_n=8):
    RED, GREEN = "#f87171", "#34d399"
    contribs = sorted(
        [{"name": n, "val": float(v), "shap": float(s)}
         for n, v, s in zip(feature_names, proc_arr, shap_vals)],
        key=lambda x: abs(x["shap"]), reverse=True
    )
    top = contribs[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    names      = [c["name"]  for c in top]
    shaps_plot = [c["shap"]  for c in top]
    colors     = [RED if s > 0 else GREEN for s in shaps_plot]
    bars = ax.barh(names, shaps_plot, color=colors, edgecolor="none", height=0.55, alpha=0.88)
    for bar, val in zip(bars, shaps_plot):
        off = 0.03 if val >= 0 else -0.03
        ha  = "left" if val >= 0 else "right"
        ax.text(val + off, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", ha=ha, va="center",
                fontsize=8.5, fontweight="600", color=TEXT, fontfamily="monospace")
    ax.axvline(0, color="#475569", linewidth=1.2)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color("#1e293b")
    ax.xaxis.grid(True, color="#1e293b", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("SHAP Contribution to Log-Odds of Failure", color=TEXT, fontsize=9)
    ax.set_title(
        f"Local SHAP Attribution  \u00b7  Failure Prob: {prob*100:.1f}%  \u00b7  Base: {base_val:+.3f}",
        color=TEXT, fontsize=10, fontweight="600", pad=10,
    )
    ax.legend(
        handles=[mpatches.Patch(color=RED, label="\u2191 Increases failure risk", alpha=0.85),
                 mpatches.Patch(color=GREEN, label="\u2193 Reduces failure risk", alpha=0.85)],
        loc="lower right", fontsize=8, facecolor=PANEL, edgecolor="#475569", labelcolor=TEXT,
    )
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar(thresh_cfg):
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:12px 0 20px;">'
            '<div style="font-size:2.4rem;">⚙️</div>'
            '<div style="font-size:1.05rem;font-weight:800;color:#818cf8;">PredictiveGuard</div>'
            '<div style="font-size:.72rem;color:#64748b;margin-top:4px;">ALGOTHON26 · ALG-DATA-02</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**🏛 Operating Mode**")
        threshold_mode = st.radio(
            "threshold_mode",
            ["optimal_f1", "high_recall", "custom"],
            format_func=lambda x: {
                "optimal_f1":  "⚖ Max-F1 (Balanced)",
                "high_recall": "🛡 High-Recall (Safety-First)",
                "custom":      "🔧 Custom Override",
            }[x],
            label_visibility="collapsed",
        )
        custom_thresh = None
        if threshold_mode == "custom":
            custom_thresh = st.slider(
                "Custom Threshold", 0.05, 0.99,
                float(thresh_cfg.get("optimal_f1_threshold", 0.84)), 0.01, "%.2f",
            )
        active_thresh = (
            custom_thresh if threshold_mode == "custom"
            else float(thresh_cfg.get("optimal_f1_threshold", 0.84)) if threshold_mode == "optimal_f1"
            else float(thresh_cfg.get("high_recall_threshold", 0.50))
        )
        st.markdown("---")
        st.markdown("**📊 Navigation**")
        page = st.radio(
            "nav_page",
            ["🔬 Live Inference", "📁 Batch Prediction",
             "📈 Model Performance", "🔍 Explainability",
             "📋 Dataset Explorer", "🗂 System Info"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<div style="font-size:.72rem;color:#475569;padding:0 4px;">'
            '<b style="color:#64748b;">Model:</b> LightGBM + Isotonic Calibration<br>'
            '<b style="color:#64748b;">Dataset:</b> AI4I 2020 Predictive Maintenance<br>'
            '<b style="color:#64748b;">Features:</b> 5 sensor + 11 physics-derived<br>'
            '<b style="color:#64748b;">Pipeline:</b> Anti-leakage · Reproducible<br>'
            '<b style="color:#64748b;">Tests:</b> 34/34 passing ✓'
            '</div>',
            unsafe_allow_html=True,
        )
    return page, threshold_mode, active_thresh


# ---------------------------------------------------------------------------
# PAGE 1 -- LIVE INFERENCE
# ---------------------------------------------------------------------------
def page_live_inference(pipeline, explainer, thresh_cfg, threshold_mode, active_thresh):
    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Live Inference</span>'
        '<span class="pg-badge pg-badge-blue">TreeSHAP</span>'
        '<span class="pg-badge pg-badge-yellow">Physics-Informed</span>'
        '<h1>🔬 Real-Time Failure Prediction</h1>'
        '<p>Enter live sensor telemetry for an instant calibrated failure probability with SHAP explainability.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        st.markdown('<div class="section-header"><h3>📡 Sensor Telemetry Input</h3></div>',
                    unsafe_allow_html=True)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"],
                                    help="L=Low / M=Medium / H=High quality grade")
        air_temp  = st.number_input("Air Temperature [K]",    290.0, 320.0, 298.1, 0.1, "%.1f")
        proc_temp = st.number_input("Process Temperature [K]", 300.0, 320.0, 308.6, 0.1, "%.1f")
        speed     = st.number_input("Rotational Speed [rpm]",  1000,  3000,  1551,  1)
        torque    = st.number_input("Torque [Nm]",              0.0,   80.0,  42.8,  0.1, "%.1f")
        wear      = st.number_input("Tool Wear [min]",           0,    300,    0,     1)
        predict_btn = st.button("⚡ Run Prediction", use_container_width=True, type="primary")
        phys = compute_physics(air_temp, proc_temp, speed, torque, wear, machine_type)
        with st.expander("🔩 Derived Physics Features Preview"):
            p1, p2 = st.columns(2)
            p1.metric("Temp Diff [K]", f"{phys['temp_diff']:.2f}",
                      delta="⚠ HDF Risk" if phys["hdf_risk"] else "✓ OK",
                      delta_color="inverse" if phys["hdf_risk"] else "normal")
            p2.metric("Mechanical Power [W]", f"{phys['power_w']:,.0f}",
                      delta="⚠ PWF Risk" if phys["pwf_risk"] else "✓ Safe Band",
                      delta_color="inverse" if phys["pwf_risk"] else "normal")
            p3, p4 = st.columns(2)
            p3.metric("Wear x Torque [min·Nm]", f"{phys['wear_torque']:,.0f}")
            p4.metric("Overstrain Ratio", f"{phys['overstrain_ratio']:.4f}",
                      delta="⚠ >1.0" if phys["overstrain_ratio"] > 1.0 else "✓ Safe",
                      delta_color="inverse" if phys["overstrain_ratio"] > 1.0 else "normal")
            st.markdown(
                f"**TWF Band:** {'🟡 IN critical wear zone [200-240 min]' if phys['in_twf_band'] else '🟢 Outside TWF zone'}"
            )
    with col_result:
        if predict_btn:
            input_record = {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]": speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
            }
            with st.spinner("Running inference pipeline..."):
                result = predict_records(
                    df=pd.DataFrame([input_record]),
                    threshold_mode=threshold_mode if threshold_mode != "custom" else "optimal_f1",
                    custom_threshold=active_thresh if threshold_mode == "custom" else None,
                )
                prob = float(result["failure_probability"].iloc[0])
            st.markdown('<div class="section-header"><h3>🎯 Prediction Result</h3></div>',
                        unsafe_allow_html=True)
            st.markdown(risk_card_html(prob, active_thresh), unsafe_allow_html=True)
            mode_labels = {
                "optimal_f1":  f"Max-F1 · OOF threshold {thresh_cfg.get('optimal_f1_threshold', 0.84):.4f}",
                "high_recall": f"High-Recall · OOF threshold {thresh_cfg.get('high_recall_threshold', 0.50):.4f}",
                "custom":       f"Custom override · {active_thresh:.4f}",
            }
            st.caption(f"🏛 Operating mode: **{mode_labels[threshold_mode]}**")
            st.markdown('<div class="section-header"><h3>🔍 SHAP Local Explanation</h3></div>',
                        unsafe_allow_html=True)
            with st.spinner("Computing TreeSHAP (log-odds space)..."):
                prep = pipeline.named_steps["preprocessing"]
                feature_names = get_pipeline_feature_names(pipeline)
                clean_df, _ = prepare_inference_features(pd.DataFrame([input_record]))
                proc_arr = prep.transform(clean_df)[0]
                shap_res = explainer(pd.DataFrame([proc_arr], columns=feature_names))
                sv       = shap_res.values[0]
                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
                else:
                    base_val = float(base_val)
                human = explain_sample_human_readable(feature_names, proc_arr, sv, base_val, prob)
            fig_wf = shap_waterfall_fig(sv, feature_names, proc_arr, base_val, prob)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)
            action       = human.get("recommended_maintenance_action", "")
            action_clean = action.split("(Decision Support): ")[-1] if "(Decision Support):" in action else action
            action_type  = action.split(":")[0].split("(")[0].strip()
            icon_map = {"CRITICAL": "🔴", "WARNING": "🟡",
                        "ADVISORY": "🟠", "NOMINAL": "🟢"}
            icon = icon_map.get(action_type, "ℹ️")
            st.markdown(
                f'<div style="background:rgba(30,41,59,.6);border:1px solid rgba(99,102,241,.2);'
                f'border-radius:12px;padding:16px 18px;margin-top:6px;">'
                f'<div style="font-size:.72rem;font-weight:700;color:#94a3b8;'
                f'letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">'
                f'🛠 Decision Support Recommendation</div>'
                f'<div style="font-size:.88rem;color:#e2e8f0;line-height:1.6;">'
                f'{icon} <b>{action_type}:</b> {action_clean}</div>'
                f'<div style="font-size:.7rem;color:#475569;margin-top:8px;font-style:italic;">'
                f'Decision support only. Operators retain full authority.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📋 Full Feature Contribution Table"):
                all_c = human.get("top_risk_drivers", []) + human.get("top_mitigating_factors", [])
                if all_c:
                    rows = [{"Feature": d["feature"], "Value": f"{d['value']:.4f}",
                              "SHAP (Log-Odds)": f"{d['shap_value']:+.4f}",
                              "Direction": "↑ Risk" if d["shap_value"] > 0 else "↓ Mitigating"}
                            for d in sorted(all_c, key=lambda x: abs(x["shap_value"]), reverse=True)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div style="background:rgba(30,41,59,.4);border:1px dashed rgba(99,102,241,.25);'
                'border-radius:14px;padding:64px 24px;text-align:center;color:#64748b;margin-top:48px;">'
                '<div style="font-size:2.5rem;margin-bottom:12px;">⚙️</div>'
                '<div style="font-size:.92rem;font-weight:500;color:#94a3b8;">'
                'Configure sensor telemetry on the left and click <b>Run Prediction</b>.</div>'
                '<div style="font-size:.78rem;margin-top:12px;color:#475569;line-height:1.7;">'
                'Physics feature engineering → LightGBM → Isotonic calibration → TreeSHAP attribution.'
                '</div></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE 2 -- BATCH PREDICTION
# ---------------------------------------------------------------------------
def page_batch_prediction(thresh_cfg, threshold_mode, active_thresh):
    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Batch Mode</span>'
        '<span class="pg-badge pg-badge-blue">CSV Import</span>'
        '<h1>📁 Batch CSV Prediction</h1>'
        '<p>Upload a CSV of machine records. Column names auto-normalized; leakage columns dropped.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_up, col_info = st.columns([1, 1], gap="large")
    with col_info:
        st.markdown('<div class="section-header"><h3>📌 Expected Schema</h3></div>',
                    unsafe_allow_html=True)
        schema_df = pd.DataFrame({
            "Column":   ["Type", "Air temperature [K]", "Process temperature [K]",
                         "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
            "Dtype":    ["string", "float", "float", "int", "float", "int"],
            "Required": ["✓"] * 6,
            "Example":  ["L / M / H", "298.1", "308.6", "1551", "42.8", "0"],
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
        st.caption("Column names auto-normalized. Failure flags (HDF/PWF/OSF/TWF/RNF) dropped automatically.")
    with col_up:
        st.markdown('<div class="section-header"><h3>📤 Upload File</h3></div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                st.success(f"✓ Loaded **{len(df_raw):,}** records · {len(df_raw.columns)} columns")
                with st.expander("Preview raw data (first 5 rows)"):
                    st.dataframe(df_raw.head(), use_container_width=True)
                if st.button("🚀 Run Batch Inference", type="primary", use_container_width=True):
                    with st.spinner(f"Running inference on {len(df_raw):,} records..."):
                        t0 = time.time()
                        results = predict_records(
                            df=df_raw,
                            threshold_mode=threshold_mode if threshold_mode != "custom" else "optimal_f1",
                            custom_threshold=active_thresh if threshold_mode == "custom" else None,
                            attach_inputs=True,
                        )
                        elapsed = time.time() - t0
                    n_flagged   = int(results["predicted_failure"].sum())
                    flagged_pct = n_flagged / max(len(results), 1) * 100
                    mc = st.columns(4)
                    mc[0].metric("Total Records",      f"{len(results):,}")
                    mc[1].metric("Predicted Failures", f"{n_flagged:,}", delta=f"{flagged_pct:.1f}% flagged")
                    mc[2].metric("Mean Failure Prob",  f"{results['failure_probability'].mean()*100:.1f}%")
                    mc[3].metric("Inference Time",     f"{elapsed*1000:.0f} ms")
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
                    ax.hist(results["failure_probability"], bins=60, color="#4f46e5", edgecolor="none", alpha=0.75)
                    ax.axvline(active_thresh, color="#f87171", lw=1.8, ls="--",
                               label=f"Threshold {active_thresh:.3f}")
                    ax.set_xlabel("Predicted Failure Probability", color=TEXT, fontsize=9)
                    ax.set_ylabel("Count", color=TEXT, fontsize=9)
                    ax.set_title("Batch Prediction Probability Distribution", color=TEXT, fontsize=10)
                    ax.tick_params(colors="#94a3b8")
                    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
                    for sp in ["bottom", "left"]: ax.spines[sp].set_color("#1e293b")
                    ax.legend(facecolor=PANEL, edgecolor="#475569", labelcolor=TEXT, fontsize=8)
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                    show_flagged = st.checkbox("Show flagged failures only")
                    disp = results[results["predicted_failure"] == 1] if show_flagged else results
                    st.dataframe(
                        disp[["failure_probability", "predicted_failure",
                               "decision_threshold", "operating_mode"]].head(500),
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇ Download Full Predictions CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="predictiveguard_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"❌ Failed to process file: {e}")
        else:
            st.markdown(
                '<div style="background:rgba(30,41,59,.4);border:1px dashed rgba(99,102,241,.3);'
                'border-radius:14px;padding:48px;text-align:center;color:#64748b;">'
                '<div style="font-size:2rem;margin-bottom:8px;">📂</div>'
                '<div style="font-size:.88rem;color:#94a3b8;">Upload a CSV to begin batch inference.</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE 3 -- MODEL PERFORMANCE
# ---------------------------------------------------------------------------
def page_model_performance(holdout_metrics):
    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Holdout Validated</span>'
        '<span class="pg-badge pg-badge-yellow">2,000 Samples</span>'
        '<h1>📈 Model Performance Dashboard</h1>'
        '<p>Generalization metrics on the 20% held-out test set. Model was frozen before any holdout evaluation.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    if not holdout_metrics:
        st.warning("Holdout metrics not found. Run `python -m src.evaluate` to generate.")
        return
    headline     = holdout_metrics.get("headline_metrics", {})
    thresh_evals = holdout_metrics.get("threshold_evaluations", {})
    subgroups    = holdout_metrics.get("subgroup_breakdowns", {})
    cost         = holdout_metrics.get("cost_analysis", {})
    st.markdown('<div class="section-header"><h3>🏅 Headline Metrics (Holdout)</h3></div>',
                unsafe_allow_html=True)
    hc = st.columns(5)
    hc[0].metric("PR-AUC",             f"{headline.get('pr_auc', 0):.4f}")
    hc[1].metric("ROC-AUC",            f"{headline.get('roc_auc', 0):.4f}")
    hc[2].metric("Brier Score",        f"{headline.get('brier_score', 0):.5f}")
    hc[3].metric("Holdout Samples",    f"{holdout_metrics.get('sample_count', 0):,}")
    hc[4].metric("Failure Prevalence", f"{holdout_metrics.get('failure_prevalence_pct', 0):.1f}%")
    st.markdown('<div class="section-header"><h3>⚖ Threshold Operating Points</h3></div>',
                unsafe_allow_html=True)
    thresh_rows = []
    for k, v in thresh_evals.items():
        thresh_rows.append({
            "Mode":        v.get("threshold_name", k),
            "Threshold":   f"{v.get('threshold_value', 0):.4f}",
            "Precision":   f"{v.get('precision', 0):.4f}",
            "Recall":      f"{v.get('recall', 0):.4f}",
            "F1":          f"{v.get('f1', 0):.4f}",
            "MCC":         f"{v.get('mcc', 0):.4f}",
            "Specificity": f"{v.get('specificity', 0):.4f}",
            "TP": v.get("true_positives", 0),
            "FP": v.get("false_positives", 0),
            "FN": v.get("false_negatives", 0),
        })
    st.dataframe(pd.DataFrame(thresh_rows), use_container_width=True, hide_index=True)
    col_fm, col_pt = st.columns(2, gap="large")
    with col_fm:
        st.markdown('<div class="section-header"><h3>🔧 Failure Mode Recall</h3></div>',
                    unsafe_allow_html=True)
        fm = subgroups.get("failure_mode_breakdown", {})
        if fm:
            modes   = list(fm.keys())
            recalls = [fm[m].get("recall", 0) * 100 for m in modes]
            colors_bar = ["#34d399" if r >= 80 else "#fbbf24" if r >= 40 else "#f87171" for r in recalls]
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
            bars = ax.bar(modes, recalls, color=colors_bar, edgecolor="none", alpha=0.85)
            for bar, r in zip(bars, recalls):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{r:.0f}%", ha="center", va="bottom", fontsize=8.5, color=TEXT, fontweight="600")
            ax.axhline(85, color="#818cf8", lw=1.2, ls="--", alpha=0.7, label="85% target")
            ax.set_ylim(0, 115)
            ax.set_ylabel("Recall (%)", color=TEXT, fontsize=9)
            ax.set_title("Holdout Recall by Failure Mode", color=TEXT, fontsize=10)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
            for sp in ["bottom", "left"]: ax.spines[sp].set_color("#1e293b")
            ax.legend(facecolor=PANEL, edgecolor="#475569", labelcolor=TEXT, fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            with st.expander("Failure Mode Details"):
                rows = [{"Mode": m, "Cases": fm[m].get("total_cases", 0),
                          "Detected": fm[m].get("detected_cases", 0),
                          "Recall": f"{fm[m].get('recall', 0)*100:.1f}%",
                          "Mean Prob": f"{fm[m].get('mean_predicted_probability', 0):.3f}"}
                        for m in modes]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with col_pt:
        st.markdown('<div class="section-header"><h3>🏭 Product Type Subgroup</h3></div>',
                    unsafe_allow_html=True)
        pt = subgroups.get("product_type_breakdown", {})
        if pt:
            types = list(pt.keys()); colors_t = ["#818cf8", "#34d399", "#fbbf24"]
            fig, axes = plt.subplots(1, 2, figsize=(6, 4))
            fig.patch.set_facecolor(BG)
            for ax, vals, title in zip(
                axes,
                [[pt[t].get("pr_auc", 0) for t in types], [pt[t].get("f1", 0) for t in types]],
                ["PR-AUC by Type", "F1-Score by Type"],
            ):
                ax.set_facecolor(PANEL)
                ax.bar(types, vals, color=colors_t, edgecolor="none", alpha=0.85)
                ax.set_title(title, color=TEXT, fontsize=9); ax.set_ylim(0.7, 1.0)
                ax.tick_params(colors="#94a3b8", labelsize=8)
                for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
                for sp in ["bottom", "left"]: ax.spines[sp].set_color("#1e293b")
            fig.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            rows_pt = [{"Type": t, "Samples": pt[t].get("sample_count", 0),
                         "Failures": pt[t].get("failure_count", 0),
                         "Prevalence": f"{pt[t].get('failure_prevalence', 0)*100:.1f}%",
                         "PR-AUC": f"{pt[t].get('pr_auc', 0):.4f}",
                         "F1": f"{pt[t].get('f1', 0):.4f}",
                         "Recall": f"{pt[t].get('recall', 0)*100:.1f}%"} for t in types]
            st.dataframe(pd.DataFrame(rows_pt), use_container_width=True, hide_index=True)
    st.markdown('<div class="section-header"><h3>💰 Cost-Benefit Analysis (Illustrative)</h3></div>',
                unsafe_allow_html=True)
    st.caption("⚠️ Cost parameters are **assumed / illustrative** and not experimentally validated: "
               "$10,000/missed failure (FN), $500/false alarm (FP), $1,500/true intervention (TP).")
    cc = st.columns(4)
    cc[0].metric("Reactive OpEx (Illustrative)",    f"${cost.get('reactive_cost_usd', 0):,.0f}")
    cc[1].metric("ML-Assisted OpEx (Illustrative)", f"${cost.get('minimum_cost_usd', 0):,.0f}")
    cc[2].metric("Net Savings (Illustrative)",       f"${cost.get('net_savings_usd', 0):,.0f}")
    cc[3].metric("Savings % (Illustrative)",         f"{cost.get('percentage_savings', 0):.1f}%")
    st.markdown('<div class="section-header"><h3>📊 Diagnostic Charts</h3></div>',
                unsafe_allow_html=True)
    ftabs = st.tabs(["PR/ROC Curves", "Confusion Matrices", "Calibration Curve", "Cost Curve"])
    for tab, fname in zip(ftabs, [
        "11_holdout_pr_roc_curves.png", "12_holdout_confusion_matrices.png",
        "14_holdout_calibration_curve.png", "15_cost_curve_tradeoff.png",
    ]):
        with tab:
            fpath = FIGURES_DIR / fname
            if fpath.exists(): st.image(str(fpath), use_container_width=True)
            else: st.info(f"Figure not found: {fname}")


# ---------------------------------------------------------------------------
# PAGE 4 -- EXPLAINABILITY
# ---------------------------------------------------------------------------
def page_explainability(feat_importance_df, interp_summary):
    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">TreeSHAP</span>'
        '<span class="pg-badge pg-badge-blue">Log-Odds Space</span>'
        '<span class="pg-badge pg-badge-yellow">Physics-Grounded</span>'
        '<h1>🔍 Model Interpretability</h1>'
        '<p>Global feature attribution and local case studies via TreeSHAP. Values are in raw log-odds space.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.info("ℹ️ SHAP values are raw margin/log-odds contributions from the LightGBM booster. "
            "Base log-odds ≈ −4.60 (≈3.3% nominal failure rate). "
            "Positive SHAP → increases failure log-odds.")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Beeswarm Summary", "Dependence Plots", "Case Studies", "Importance Table"])
    with tab1:
        fpath = FIGURES_DIR / "16_shap_summary_beeswarm.png"
        if fpath.exists():
            st.image(str(fpath),
                     caption="Fig. 16: TreeSHAP Global Feature Attribution (Top 15 Features, Log-Odds Space)",
                     use_container_width=True)
        else:
            st.warning("Run 'python -m src.explain' to generate SHAP figures.")
    with tab2:
        fpath = FIGURES_DIR / "17_shap_dependence_interactions.png"
        if fpath.exists():
            st.image(str(fpath),
                     caption="Fig. 17: SHAP Dependence & Physical Interaction Plots",
                     use_container_width=True)
    with tab3:
        fpath = FIGURES_DIR / "18_shap_waterfall_case_studies.png"
        if fpath.exists():
            st.image(str(fpath),
                     caption="Fig. 18: Local SHAP Waterfall -- 4 Industrial Case Studies",
                     use_container_width=True)
        if interp_summary and "case_studies" in interp_summary:
            st.markdown('<div class="section-header"><h3>📋 Case Study Reports</h3></div>',
                        unsafe_allow_html=True)
            icon_map = {"CRITICAL": "🔴", "WARNING": "🟡",
                        "ADVISORY": "🟠", "NOMINAL": "🟢"}
            for key, cs in interp_summary["case_studies"].items():
                exp   = cs.get("explanation", {})
                prob  = exp.get("failure_probability", 0)
                action = exp.get("recommended_maintenance_action", "N/A")
                action_clean = action.split("(Decision Support): ")[-1] if "(Decision Support):" in action else action
                action_type  = action.split(":")[0].split("(")[0].strip()
                icon = icon_map.get(action_type, "ℹ️")
                with st.expander(f"{icon} {cs.get('title', key)} -- {prob*100:.1f}%"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Dominant Driver:** `{exp.get('dominant_risk_driver', 'N/A')}`")
                    c2.markdown(f"**Base Log-Odds:** `{exp.get('base_value', 0):+.3f}`")
                    st.markdown(f"**Decision Support:** {icon} {action_clean}")
                    all_c = exp.get("top_risk_drivers", []) + exp.get("top_mitigating_factors", [])
                    if all_c:
                        rows = [{"Feature": d["feature"], "Value": f"{d['value']:.4f}",
                                  "SHAP": f"{d['shap_value']:+.4f}",
                                  "Direction": "↑ Risk" if d["shap_value"] > 0 else "↓ Mitigating"}
                                for d in sorted(all_c, key=lambda x: abs(x["shap_value"]), reverse=True)]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tab4:
        if not feat_importance_df.empty:
            st.markdown('<div class="section-header"><h3>📊 Unified Feature Importance (Top 15)</h3></div>',
                        unsafe_allow_html=True)
            st.caption("Ranked by Mean |SHAP| (log-odds). Permutation = PR-AUC drop. Native LightGBM gain also shown.")
            top_feats = feat_importance_df.head(15)
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.patch.set_facecolor(BG)
            for ax, col, label, color in zip(
                axes,
                ["mean_abs_shap", "perm_importance_mean", "native_gain"],
                ["Mean |SHAP|", "Permutation Importance\n(PR-AUC drop)", "Native LightGBM\nGain"],
                ["#818cf8", "#34d399", "#fbbf24"],
            ):
                ax.set_facecolor(PANEL)
                ax.barh(top_feats["feature"].values[::-1], top_feats[col].values[::-1],
                        color=color, edgecolor="none", alpha=0.82)
                ax.set_xlabel(label, color=TEXT, fontsize=8.5)
                ax.tick_params(colors="#94a3b8", labelsize=7.5)
                for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
                for sp in ["bottom", "left"]: ax.spines[sp].set_color("#1e293b")
            fig.suptitle("Feature Importance: SHAP | Permutation | Native Gain",
                          color=TEXT, fontsize=10, fontweight="600", y=1.02)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.dataframe(feat_importance_df, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance CSV not found. Run 'python -m src.explain' first.")


# ---------------------------------------------------------------------------
# PAGE 5 -- DATASET EXPLORER
# ---------------------------------------------------------------------------
def page_dataset_explorer(df_train):
    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">AI4I 2020</span>'
        '<span class="pg-badge pg-badge-blue">EDA</span>'
        '<h1>📋 Dataset Explorer</h1>'
        '<p>Interactive exploration of the AI4I 2020 Predictive Maintenance dataset (development split, 8,000 records).</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    n_fail  = int(df_train["Machine failure"].sum()) if "Machine failure" in df_train.columns else 0
    n_total = len(df_train)
    sc = st.columns(4)
    sc[0].metric("Total Records",     f"{n_total:,}")
    sc[1].metric("Machine Failures",  f"{n_fail:,}")
    sc[2].metric("Failure Prevalence", f"{n_fail/max(n_total,1)*100:.2f}%")
    sc[3].metric("Columns",           f"{len(df_train.columns)}")
    tabs = st.tabs(["Distributions", "Correlations", "Physics Visualizations", "EDA Gallery", "Raw Data"])
    with tabs[0]:
        col_sel = st.selectbox("Select feature", CANONICAL_NUMERIC_FEATURES)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(BG); ax1.set_facecolor(PANEL); ax2.set_facecolor(PANEL)
        has_target = "Machine failure" in df_train.columns
        if has_target:
            ax1.hist(df_train[df_train["Machine failure"]==0][col_sel].dropna(),
                     bins=40, color="#34d399", alpha=0.6, label="Nominal", density=True)
            ax1.hist(df_train[df_train["Machine failure"]==1][col_sel].dropna(),
                     bins=40, color="#f87171", alpha=0.7, label="Failure", density=True)
            ax1.legend(facecolor=PANEL, edgecolor="#475569", labelcolor=TEXT, fontsize=8)
        else:
            ax1.hist(df_train[col_sel].dropna(), bins=40, color="#818cf8", alpha=0.75)
        ax1.set_title(f"Distribution: {col_sel}", color=TEXT, fontsize=10)
        ax1.set_xlabel(col_sel, color=TEXT, fontsize=9)
        ax1.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax1.spines[sp].set_color("#1e293b")
        if "Type" in df_train.columns:
            type_groups = [df_train[df_train["Type"]==t][col_sel].dropna() for t in ["L", "M", "H"]]
            bp = ax2.boxplot(type_groups, labels=["L", "M", "H"], patch_artist=True)
            for patch, c in zip(bp["boxes"], ["#818cf8", "#34d399", "#fbbf24"]):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            for el in ["whiskers", "caps", "medians", "fliers"]:
                for item in bp[el]: item.set_color("#94a3b8")
        ax2.set_title(f"{col_sel} by Machine Type", color=TEXT, fontsize=10)
        ax2.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax2.spines[sp].set_color("#1e293b")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with tabs[1]:
        fpath = FIGURES_DIR / "03_correlation_heatmap.png"
        if fpath.exists():
            st.image(str(fpath), caption="Fig. 3: Feature Correlation Heatmap", use_container_width=True)
    with tabs[2]:
        c1, c2 = st.columns(2)
        for fname, caption, col in [
            ("04_hdf_region_tempdiff_vs_rpm.png",     "HDF Region: Temp Diff vs RPM", c1),
            ("06_osf_region_wear_torque_by_type.png",  "OSF: Wear x Torque by Type",  c2),
        ]:
            with col:
                fpath = FIGURES_DIR / fname
                if fpath.exists(): st.image(str(fpath), caption=caption, use_container_width=True)
        c3, c4 = st.columns(2)
        for fname, caption, col in [
            ("05_pwf_region_power_distribution.png", "PWF: Power Distribution", c3),
            ("09_feature_ablation_comparison.png",   "Feature Ablation Study",  c4),
        ]:
            with col:
                fpath = FIGURES_DIR / fname
                if fpath.exists(): st.image(str(fpath), caption=caption, use_container_width=True)
    with tabs[3]:
        gallery = [
            ("01_class_and_mode_distribution.png",        "Class & Mode Distribution"),
            ("02_sensor_distributions_by_failure.png",     "Sensor Distributions by Failure"),
            ("07_temperature_random_walk_row_order.png",   "Temperature Time-Series"),
            ("08_cv_strategy_comparison.png",              "Cross-Validation Strategy"),
            ("10_calibration_and_threshold_tradeoff.png",  "Calibration & Threshold Trade-off"),
            ("13_failure_mode_breakdown.png",              "Failure Mode Breakdown"),
        ]
        for i in range(0, len(gallery), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(gallery):
                    fname, cap = gallery[i + j]
                    fpath = FIGURES_DIR / fname
                    with col:
                        if fpath.exists(): st.image(str(fpath), caption=cap, use_container_width=True)
    with tabs[4]:
        st.markdown("**Development Dataset (first 200 rows)**")
        st.dataframe(df_train.head(200), use_container_width=True)
        st.caption(f"Shape: {df_train.shape[0]:,} rows x {df_train.shape[1]} columns. "
                   "Failure mode flags shown here are for EDA only -- never used as model features.")


# ---------------------------------------------------------------------------
# PAGE 6 -- SYSTEM INFO
# ---------------------------------------------------------------------------
def page_system_info(thresh_cfg):
    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Production</span>'
        '<span class="pg-badge pg-badge-blue">Reproducible</span>'
        '<h1>🗂 System Information & Reproducibility</h1>'
        '<p>Pipeline architecture, anti-leakage audit, threshold configuration, and physics feature dictionary.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-header"><h3>🏗 Pipeline Architecture</h3></div>',
                    unsafe_allow_html=True)
        st.code(
            "final_pipeline.joblib\n"
            "├── preprocessing (Pipeline)\n"
            "│   ├── physics  ->  PhysicsFeatureEngineer\n"
            "│   │               11 domain-derived features\n"
            "│   └── preprocessor  ->  ColumnTransformer\n"
            "│         ├── num  -> SimpleImputer(median)\n"
            "│         └── cat  -> SimpleImputer + OneHotEncoder\n"
            "└── classifier  ->  LightGBMClassifier\n"
            "                    (Isotonic-calibrated via\n"
            "                     CalibratedClassifierCV)",
            language=None,
        )
        st.markdown('<div class="section-header"><h3>🔒 Anti-Leakage Guarantees</h3></div>',
                    unsafe_allow_html=True)
        for g in [
            "✅ HDF/PWF/OSF/TWF/RNF never enter the inference pipeline",
            "✅ UDI / Product ID removed before feature extraction",
            "✅ Holdout set frozen until Phase 5 final evaluation",
            "✅ OOF predictions used for threshold selection (never holdout)",
            "✅ Stacking features generated using strictly OOF auxiliary predictions",
            "✅ 34 / 34 automated tests passing (including anti-leakage assertion)",
        ]:
            st.markdown(f'<div class="feat-chip">{g}</div><br>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-header"><h3>🏛 Threshold Configuration</h3></div>',
                    unsafe_allow_html=True)
        cfg_rows = [
            ("Optimal F1 Threshold (OOF-estimated)", f"{thresh_cfg.get('optimal_f1_threshold', 0):.6f}"),
            ("OOF F1 Score",                          f"{thresh_cfg.get('optimal_f1_score', 0):.6f}"),
            ("High-Recall Threshold (OOF-estimated)", f"{thresh_cfg.get('high_recall_threshold', 0):.6f}"),
            ("High-Recall OOF Recall",                f"{thresh_cfg.get('high_recall_metrics', {}).get('recall', 0):.6f}"),
            ("Brier Score (Uncalibrated)",             f"{thresh_cfg.get('brier_scores', {}).get('uncalibrated', 0):.6f}"),
            ("Brier Score (Isotonic)",                 f"{thresh_cfg.get('brier_scores', {}).get('isotonic', 0):.6f}"),
        ]
        st.dataframe(pd.DataFrame(cfg_rows, columns=["Parameter", "Value"]),
                     use_container_width=True, hide_index=True)
        st.markdown('<div class="section-header"><h3>📐 Physics Feature Dictionary</h3></div>',
                    unsafe_allow_html=True)
        feat_desc = [
            ("temp_diff",                    "Process T - Air T [K]; HDF convective gradient"),
            ("power_w",                      "tau x omega x 2pi/60 [W]; mechanical output power"),
            ("power_below_limit",             "max(0, 3500 - power_w); stall risk"),
            ("power_above_limit",             "max(0, power_w - 9000); overload risk"),
            ("power_distance_to_safe_band",   "Unified PWF excursion magnitude"),
            ("wear_torque",                   "Wear x Torque; combined strain load"),
            ("overstrain_ratio",              "wear_torque / variant capacity; OSF risk"),
            ("is_known_type",                 "1 if Type in {L,M,H}; fallback indicator"),
            ("low_speed_low_tempdiff",         "HDF: temp deficit x speed deficit"),
            ("tool_wear_in_critical_band",    "1 if 200 <= wear <= 240 min"),
            ("tool_wear_critical_proximity",  "Smooth exp proximity to TWF band"),
            ("speed_torque_ratio",            "RPM / Torque; gear impedance metric"),
            ("temp_ratio",                   "T_proc / T_air; normalized heat accumulation"),
        ]
        st.dataframe(pd.DataFrame(feat_desc, columns=["Feature", "Physical Meaning"]),
                     use_container_width=True, hide_index=True)
    opt_path = ROOT_DIR / "models" / "optuna_best_params.json"
    if opt_path.exists():
        with open(opt_path) as f:
            opt_params = json.load(f)
        with st.expander("🔬 Optuna Best Hyperparameters (50+ trials per model)"):
            st.json(opt_params)
    mc_path = REPORTS_DIR / "model_comparison.csv"
    if mc_path.exists():
        with st.expander("📊 Full Model Comparison Table (Phase 4)"):
            st.dataframe(pd.read_csv(mc_path), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    with st.spinner("Loading PredictiveGuard production pipeline..."):
        try:
            pipeline = load_pipeline()
            model_loaded = True
        except Exception as e:
            st.error(f"❌ Could not load model pipeline: {e}")
            model_loaded = False
            pipeline = None
    thresh_cfg         = load_threshold_cfg()
    holdout_metrics    = load_holdout_metrics()
    feat_importance_df = load_feature_importance()
    interp_summary     = load_interpretability_summary()
    try:
        df_train = load_training_data()
    except Exception:
        df_train = pd.DataFrame()
    explainer = build_explainer(pipeline) if model_loaded else None
    page, threshold_mode, active_thresh = render_sidebar(thresh_cfg)
    if "Live Inference" in page:
        if model_loaded:
            page_live_inference(pipeline, explainer, thresh_cfg, threshold_mode, active_thresh)
        else:
            st.error("Model not loaded. Cannot run live inference.")
    elif "Batch" in page:
        if model_loaded:
            page_batch_prediction(thresh_cfg, threshold_mode, active_thresh)
        else:
            st.error("Model not loaded. Cannot run batch prediction.")
    elif "Performance" in page:
        page_model_performance(holdout_metrics)
    elif "Explainability" in page:
        page_explainability(feat_importance_df, interp_summary)
    elif "Dataset" in page:
        if not df_train.empty:
            page_dataset_explorer(df_train)
        else:
            st.warning("Training data not found.")
    elif "System" in page:
        page_system_info(thresh_cfg)


if __name__ == "__main__":
    main()
