"""
Transform app/streamlit_app.py to replace all emojis with Material Symbols Outlined icons.
"""

import re

with open("app/streamlit_app.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update imports
if "icon_html" not in code:
    code = code.replace(
        "from app.components import page_header, chip_html, compact_physics_card, result_verdict_card",
        "from app.components import page_header, chip_html, compact_physics_card, result_verdict_card, icon_html"
    )

# 2. Risk card HTML
old_risk_card = """def risk_card_html(prob: float, threshold: float) -> str:
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
    )"""

new_risk_card = """def risk_card_html(prob: float, threshold: float) -> str:
    pct = prob * 100
    if prob >= threshold:
        cls = "risk-critical"
        label = "FAILURE PREDICTED"
        icon = icon_html("error_outline", "icon-danger")
    elif prob >= 0.25:
        cls = "risk-warning"
        label = "ELEVATED RISK"
        icon = icon_html("warning_amber", "icon-warning")
    else:
        cls = "risk-nominal"
        label = "NOMINAL OPERATION"
        icon = icon_html("check_circle", "icon-success")
    return (
        f'<div class="risk-card {cls}">'
        f'<div class="risk-label" style="display: flex; align-items: center; justify-content: center; gap: 6px;">{icon} <span>{label}</span></div>'
        f'<div class="risk-prob">{pct:.1f}%</div>'
        f'<div class="risk-sub">Failure Probability &middot; Operating Threshold {threshold:.2%}</div>'
        '</div>'
    )"""

code = code.replace(old_risk_card, new_risk_card)

# 3. Page 1 Live Inference hero & controls
code = code.replace(
    """    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Live Inference</span>'
        '<span class="pg-badge pg-badge-blue">TreeSHAP</span>'
        '<span class="pg-badge pg-badge-yellow">Physics-Informed</span>'
        '<h1>🔬 Real-Time Failure Prediction</h1>'
        '<p>Enter live sensor telemetry for an instant calibrated failure probability with SHAP explainability.</p>'
        '</div>',
        unsafe_allow_html=True,
    )""",
    """    page_header(
        title="Real-Time Failure Prediction",
        description="Enter live sensor telemetry for an instant calibrated failure probability with SHAP explainability.",
        badges=["Live Inference", "TreeSHAP", "Physics-Informed"],
        icon="precision_manufacturing"
    )"""
)

code = code.replace(
    '<div class="section-header"><h3>📡 Sensor Telemetry Input</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">tune</span><span class="section-title">Sensor Telemetry Input</span></div>'
)

code = code.replace(
    'predict_btn = st.button("⚡ Run Prediction", use_container_width=True, type="primary")',
    'predict_btn = st.button("Run Prediction", use_container_width=True, type="primary", icon=":material/bolt:")'
)

code = code.replace(
    'with st.expander("🔩 Derived Physics Features Preview"):',
    'with st.expander("Derived Physics Features Preview", icon=":material/science:"):'
)

code = code.replace(
    'delta="⚠ HDF Risk" if phys["hdf_risk"] else "✓ OK",',
    'delta="HDF Risk" if phys["hdf_risk"] else "Normal",'
)

code = code.replace(
    'delta="⚠ PWF Risk" if phys["pwf_risk"] else "✓ Safe Band",',
    'delta="PWF Risk" if phys["pwf_risk"] else "Safe Band",'
)

code = code.replace(
    'delta="⚠ >1.0" if phys["overstrain_ratio"] > 1.0 else "✓ Safe",',
    'delta="Over Limit (>1.0)" if phys["overstrain_ratio"] > 1.0 else "Safe",'
)

code = code.replace(
    'f"**TWF Band:** {\'🟡 IN critical wear zone [200-240 min]\' if phys[\'in_twf_band\'] else \'🟢 Outside TWF zone\'}"',
    "f\"**TWF Status:** {chip_html('Critical wear zone [200-240 min]', 'warning') if phys['in_twf_band'] else chip_html('Outside TWF zone', 'success')}\""
)

code = code.replace(
    '<div class="section-header"><h3>🎯 Prediction Result</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">analytics</span><span class="section-title">Prediction Result</span></div>'
)

code = code.replace(
    'st.caption(f"🏛 Operating mode: **{mode_labels[threshold_mode]}**")',
    'st.caption(f"Operating mode: **{mode_labels[threshold_mode]}**")'
)

code = code.replace(
    '<div class="section-header"><h3>🔍 SHAP Local Explanation</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">troubleshoot</span><span class="section-title">SHAP Local Explanation</span></div>'
)

# Decision support box in Page 1
old_ds = """            icon_map = {"CRITICAL": "🔴", "WARNING": "🟡",
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
            )"""

new_ds = """            icon_map = {
                "CRITICAL": icon_html("error_outline", "icon-danger"),
                "WARNING": icon_html("warning_amber", "icon-warning"),
                "ADVISORY": icon_html("info", "icon-accent"),
                "NOMINAL": icon_html("check_circle", "icon-success"),
            }
            icon = icon_map.get(action_type, icon_html("info", "icon-muted"))
            st.markdown(
                f'<div class="pg-card" style="margin-top: 10px;">'
                f'<div style="font-size:.75rem;font-weight:700;color:var(--pg-muted);letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:6px;">'
                f'<span class="material-symbols-outlined icon-inline">build</span> <span>Decision Support Recommendation</span></div>'
                f'<div style="font-size:.88rem;color:var(--pg-text);line-height:1.6;display:flex;align-items:flex-start;gap:8px;">'
                f'<span>{icon}</span> <span><strong style="color:var(--pg-text);">{action_type}:</strong> {action_clean}</span></div>'
                f'<div style="font-size:.72rem;color:var(--pg-muted);margin-top:8px;font-style:italic;">'
                f'Decision support only. Operators retain full authority.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )"""

code = code.replace(old_ds, new_ds)

code = code.replace(
    'with st.expander("📋 Full Feature Contribution Table"):',
    'with st.expander("Full Feature Contribution Table", icon=":material/table_rows:"):'
)

code = code.replace(
    '"Direction": "↑ Risk" if d["shap_value"] > 0 else "↓ Mitigating"',
    '"Direction": "+ Risk" if d["shap_value"] > 0 else "- Mitigating"'
)

# Placeholder in Page 1
old_ph = """            st.markdown(
                '<div style="background:rgba(30,41,59,.4);border:1px dashed rgba(99,102,241,.25);'
                'border-radius:14px;padding:64px 24px;text-align:center;color:#64748b;margin-top:48px;">'
                '<div style="font-size:2.5rem;margin-bottom:12px;">⚙️</div>'
                '<div style="font-size:.92rem;font-weight:500;color:#94a3b8;">'
                'Configure sensor telemetry on the left and click <b>Run Prediction</b>.</div>'
                '<div style="font-size:.78rem;margin-top:12px;color:#475569;line-height:1.7;">'
                'Physics feature engineering → LightGBM → Isotonic calibration → TreeSHAP attribution.'
                '</div></div>',
                unsafe_allow_html=True,
            )"""

new_ph = """            st.markdown(
                '<div style="background-color:var(--pg-surface);border:1px dashed var(--pg-border-strong);'
                'border-radius:8px;padding:64px 24px;text-align:center;color:var(--pg-muted);margin-top:24px;">'
                '<div style="margin-bottom:12px;"><span class="material-symbols-outlined icon-lg icon-muted">precision_manufacturing</span></div>'
                '<div style="font-size:.92rem;font-weight:600;color:var(--pg-text);">'
                'Configure sensor telemetry on the left and click <b>Run Prediction</b>.</div>'
                '<div style="font-size:.80rem;margin-top:8px;color:var(--pg-muted);line-height:1.6;">'
                'Physics feature engineering &rarr; LightGBM &rarr; Isotonic calibration &rarr; TreeSHAP attribution.'
                '</div></div>',
                unsafe_allow_html=True,
            )"""

code = code.replace(old_ph, new_ph)

# 4. Page 2 Batch Prediction
code = code.replace(
    """    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Batch Mode</span>'
        '<span class="pg-badge pg-badge-blue">CSV Import</span>'
        '<h1>📁 Batch CSV Prediction</h1>'
        '<p>Upload a CSV of machine records. Column names auto-normalized; leakage columns dropped.</p>'
        '</div>',
        unsafe_allow_html=True,
    )""",
    """    page_header(
        title="Batch CSV Prediction",
        description="Upload a CSV of machine records. Column names auto-normalized; leakage columns dropped.",
        badges=["Batch Mode", "CSV Import"],
        icon="upload_file"
    )"""
)

code = code.replace(
    '<div class="section-header"><h3>📌 Expected Schema</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">rule</span><span class="section-title">Expected Schema</span></div>'
)

code = code.replace(
    '"Required": ["✓"] * 6,',
    '"Required": ["Yes"] * 6,'
)

code = code.replace(
    '<div class="section-header"><h3>📤 Upload File</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">cloud_upload</span><span class="section-title">Upload File</span></div>'
)

code = code.replace(
    'st.success(f"✓ Loaded **{len(df_raw):,}** records · {len(df_raw.columns)} columns")',
    'st.success(f"Loaded {len(df_raw):,} records · {len(df_raw.columns)} columns", icon=":material/check_circle:")'
)

code = code.replace(
    'if st.button("🚀 Run Batch Inference", type="primary", use_container_width=True):',
    'if st.button("Run Batch Inference", type="primary", use_container_width=True, icon=":material/play_arrow:"):'
)

code = code.replace(
    '"⬇ Download Full Predictions CSV"',
    '"Download Full Predictions CSV", icon=":material/download:"'
)

code = code.replace(
    'st.error(f"❌ Failed to process file: {e}")',
    'st.error(f"Failed to process file: {e}", icon=":material/error:")'
)

code = code.replace(
    """            st.markdown(
                '<div style="background:rgba(30,41,59,.4);border:1px dashed rgba(99,102,241,.3);'
                'border-radius:14px;padding:48px;text-align:center;color:#64748b;">'
                '<div style="font-size:2rem;margin-bottom:8px;">📂</div>'
                '<div style="font-size:.88rem;color:#94a3b8;">Upload a CSV to begin batch inference.</div>'
                '</div>',
                unsafe_allow_html=True,
            )""",
    """            st.markdown(
                '<div style="background-color:var(--pg-surface);border:1px dashed var(--pg-border-strong);'
                'border-radius:8px;padding:48px;text-align:center;color:var(--pg-muted);">'
                '<div style="margin-bottom:8px;"><span class="material-symbols-outlined icon-lg icon-muted">folder_open</span></div>'
                '<div style="font-size:.88rem;color:var(--pg-text);">Upload a CSV file to begin batch inference.</div>'
                '</div>',
                unsafe_allow_html=True,
            )"""
)

# 5. Page 3 Model Performance
code = code.replace(
    """    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Holdout Validated</span>'
        '<span class="pg-badge pg-badge-yellow">2,000 Samples</span>'
        '<h1>📈 Model Performance Dashboard</h1>'
        '<p>Generalization metrics on the 20% held-out test set. Model was frozen before any holdout evaluation.</p>'
        '</div>',
        unsafe_allow_html=True,
    )""",
    """    page_header(
        title="Model Performance Dashboard",
        description="Generalization metrics on the 20% held-out test set. Model was frozen before any holdout evaluation.",
        badges=["Holdout Validated", "2,000 Samples"],
        icon="monitoring"
    )"""
)

code = code.replace(
    '<div class="section-header"><h3>🏅 Headline Metrics (Holdout)</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">speed</span><span class="section-title">Headline Metrics (Holdout)</span></div>'
)

code = code.replace(
    '<div class="section-header"><h3>⚖ Threshold Operating Points</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">tune</span><span class="section-title">Threshold Operating Points</span></div>'
)

code = code.replace(
    '<div class="section-header"><h3>🔧 Failure Mode Recall</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">build</span><span class="section-title">Failure Mode Recall</span></div>'
)

code = code.replace(
    '<div class="section-header"><h3>🏭 Product Type Subgroup</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">factory</span><span class="section-title">Product Type Subgroup</span></div>'
)

code = code.replace(
    '<div class="section-header"><h3>💰 Cost-Benefit Analysis (Illustrative)</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">payments</span><span class="section-title">Cost-Benefit Analysis (Illustrative)</span></div>'
)

code = code.replace(
    'st.caption("⚠️ Cost parameters are **assumed / illustrative** and not experimentally validated: "',
    'st.caption("Cost parameters are **assumed / illustrative** and not experimentally validated: "'
)

code = code.replace(
    '<div class="section-header"><h3>📊 Diagnostic Charts</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">query_stats</span><span class="section-title">Diagnostic Charts</span></div>'
)

# 6. Page 4 Explainability
code = code.replace(
    """    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">TreeSHAP</span>'
        '<span class="pg-badge pg-badge-blue">Log-Odds Space</span>'
        '<span class="pg-badge pg-badge-yellow">Physics-Grounded</span>'
        '<h1>🔍 Model Interpretability</h1>'
        '<p>Global feature attribution and local case studies via TreeSHAP. Values are in raw log-odds space.</p>'
        '</div>',
        unsafe_allow_html=True,
    )""",
    """    page_header(
        title="Model Interpretability",
        description="Global feature attribution and local case studies via TreeSHAP. Values are in raw log-odds space.",
        badges=["TreeSHAP", "Log-Odds Space", "Physics-Grounded"],
        icon="troubleshoot"
    )"""
)

code = code.replace(
    'st.info("ℹ️ SHAP values are raw margin/log-odds contributions',
    'st.info("SHAP values are raw margin/log-odds contributions'
)

code = code.replace(
    '<div class="section-header"><h3>📋 Case Study Reports</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">assignment</span><span class="section-title">Case Study Reports</span></div>'
)

old_cs_loop = """            icon_map = {"CRITICAL": "🔴", "WARNING": "🟡",
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
                    st.markdown(f"**Decision Support:** {icon} {action_clean}")"""

new_cs_loop = """            icon_map = {
                "CRITICAL": icon_html("error_outline", "icon-danger"),
                "WARNING": icon_html("warning_amber", "icon-warning"),
                "ADVISORY": icon_html("info", "icon-accent"),
                "NOMINAL": icon_html("check_circle", "icon-success"),
            }
            for key, cs in interp_summary["case_studies"].items():
                exp   = cs.get("explanation", {})
                prob  = exp.get("failure_probability", 0)
                action = exp.get("recommended_maintenance_action", "N/A")
                action_clean = action.split("(Decision Support): ")[-1] if "(Decision Support):" in action else action
                action_type  = action.split(":")[0].split("(")[0].strip()
                icon = icon_map.get(action_type, icon_html("info", "icon-muted"))
                with st.expander(f"{cs.get('title', key)} · Breakdown Probability {prob*100:.1f}%"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Dominant Driver:** `{exp.get('dominant_risk_driver', 'N/A')}`")
                    c2.markdown(f"**Base Log-Odds:** `{exp.get('base_value', 0):+.3f}`")
                    st.markdown(f"**Decision Support:** {icon} {action_clean}", unsafe_allow_html=True)"""

code = code.replace(old_cs_loop, new_cs_loop)

code = code.replace(
    '<div class="section-header"><h3>📊 Unified Feature Importance (Top 15)</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">bar_chart</span><span class="section-title">Unified Feature Importance (Top 15)</span></div>'
)

# 7. Page 5 Dataset Explorer
code = code.replace(
    """    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">AI4I 2020</span>'
        '<span class="pg-badge pg-badge-blue">EDA</span>'
        '<h1>📋 Dataset Explorer</h1>'
        '<p>Interactive exploration of the AI4I 2020 Predictive Maintenance dataset (development split, 8,000 records).</p>'
        '</div>',
        unsafe_allow_html=True,
    )""",
    """    page_header(
        title="Dataset Explorer",
        description="Interactive exploration of the AI4I 2020 Predictive Maintenance dataset (development split, 8,000 records).",
        badges=["AI4I 2020", "EDA"],
        icon="dataset"
    )"""
)

# 8. Page 6 System Info
code = code.replace(
    """    st.markdown(
        '<div class="pg-hero">'
        '<span class="pg-badge">Production</span>'
        '<span class="pg-badge pg-badge-blue">Reproducible</span>'
        '<h1>🗂 System Information & Reproducibility</h1>'
        '<p>Pipeline architecture, anti-leakage audit, threshold configuration, and physics feature dictionary.</p>'
        '</div>',
        unsafe_allow_html=True,
    )""",
    """    page_header(
        title="System Information & Reproducibility",
        description="Pipeline architecture, anti-leakage audit, threshold configuration, and physics feature dictionary.",
        badges=["Production", "Reproducible"],
        icon="terminal"
    )"""
)

code = code.replace(
    '<div class="section-header"><h3>🏗 Pipeline Architecture</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">account_tree</span><span class="section-title">Pipeline Architecture</span></div>'
)

code = code.replace(
    '<div class="section-header"><h3>🔒 Anti-Leakage Guarantees</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">lock</span><span class="section-title">Anti-Leakage Guarantees</span></div>'
)

old_guarantees = """        for g in [
            "✅ HDF/PWF/OSF/TWF/RNF never enter the inference pipeline",
            "✅ UDI / Product ID removed before feature extraction",
            "✅ Holdout set frozen until Phase 5 final evaluation",
            "✅ OOF predictions used for threshold selection (never holdout)",
            "✅ Stacking features generated using strictly OOF auxiliary predictions",
            "✅ 34 / 34 automated tests passing (including anti-leakage assertion)",
        ]:
            st.markdown(f'<div class="feat-chip">{g}</div><br>', unsafe_allow_html=True)"""

new_guarantees = """        for g in [
            "HDF/PWF/OSF/TWF/RNF never enter the inference pipeline",
            "UDI / Product ID removed before feature extraction",
            "Holdout set frozen until Phase 5 final evaluation",
            "OOF predictions used for threshold selection (never holdout)",
            "Stacking features generated using strictly OOF auxiliary predictions",
            "34 / 34 automated tests passing (including anti-leakage assertion)",
        ]:
            st.markdown(f'<div class="feat-chip">{icon_html("verified", "icon-success")} <span>{g}</span></div><br>', unsafe_allow_html=True)"""

code = code.replace(old_guarantees, new_guarantees)

code = code.replace(
    '<div class="section-header"><h3>🏛 Threshold Configuration</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">settings_suggest</span><span class="section-title">Threshold Configuration</span></div>'
)

code = code.replace(
    '<div class="section-header"><h3>📐 Physics Feature Dictionary</h3></div>',
    '<div class="section-header"><span class="material-symbols-outlined icon-inline">menu_book</span><span class="section-title">Physics Feature Dictionary</span></div>'
)

code = code.replace(
    'with st.expander("🔬 Optuna Best Hyperparameters (50+ trials per model)"):',
    'with st.expander("Optuna Best Hyperparameters (50+ trials per model)", icon=":material/tune:"):'
)

code = code.replace(
    'with st.expander("📊 Full Model Comparison Table (Phase 4)"):',
    'with st.expander("Full Model Comparison Table (Phase 4)", icon=":material/table_chart:"):'
)

code = code.replace(
    'st.error(f"❌ Could not load model pipeline: {e}")',
    'st.error(f"Could not load model pipeline: {e}", icon=":material/error:")'
)

with open("app/streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Replacement complete. Check remaining emojis...")
