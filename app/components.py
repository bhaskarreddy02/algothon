"""
PredictiveGuard UI Components (app/components.py)
Reusable, accessible UI components for Streamlit without emojis or gradients.
"""

import textwrap
from typing import List, Optional, Dict, Any, Callable
import streamlit as st


def clean_html(s: str) -> str:
    """Dedent and strip multiline HTML to prevent markdown code block formatting."""
    return textwrap.dedent(s).strip()


def icon_html(name: str, color_class: str = "", style: str = "") -> str:
    """Render a Material Symbols Outlined icon with customizable color and style."""
    cls = f"material-symbols-outlined {color_class}".strip()
    inline_style = f"vertical-align: middle; {style}".strip()
    return f'<span class="{cls}" style="{inline_style}">{name}</span>'


def page_header(title: str, description: str, badges: Optional[List[str]] = None, icon: Optional[str] = None) -> None:
    """Render a clean, flat editorial page header with optional outlined icon."""
    badge_html = ""
    if badges:
        items = "".join([f'<span class="pg-chip pg-chip-neutral" style="margin-right: 6px;">{b}</span>' for b in badges])
        badge_html = f'<div style="margin-bottom: 8px;">{items}</div>'

    icon_prefix = f'{icon_html(icon, style="font-size: 1.6rem; margin-right: 8px;")} ' if icon else ""
    html = f'<div class="pg-page-header">{badge_html}<h1 class="pg-page-title" style="display: flex; align-items: center;">{icon_prefix}<span>{title}</span></h1><p class="pg-page-desc">{description}</p></div>'
    st.markdown(html, unsafe_allow_html=True)


def chip_html(text: str, status: str = "neutral") -> str:
    """Generate HTML for an inline status chip."""
    valid_status = status if status in ["neutral", "success", "warning", "danger", "accent"] else "neutral"
    return f'<span class="pg-chip pg-chip-{valid_status}">{text}</span>'


def progress_bar_html(pct: float, status: str = "accent") -> str:
    """Generate HTML for a thin progress bar."""
    pct_clamped = max(0.0, min(100.0, pct))
    color_map = {
        "accent": "var(--pg-accent)",
        "success": "var(--pg-success)",
        "warning": "var(--pg-warning)",
        "danger": "var(--pg-danger)",
        "neutral": "var(--pg-muted)",
    }
    fill_color = color_map.get(status, "var(--pg-accent)")
    return f"""
    <div class="pg-prog-track">
        <div class="pg-prog-fill" style="width: {pct_clamped:.1f}%; background-color: {fill_color};"></div>
    </div>
    """


def compact_physics_card(label: str, value: str, status_text: str, status: str, pct_progress: float = 0.0) -> None:
    """Render a compact derived physics metric card with live progress bar."""
    chip = chip_html(status_text, status)
    prog = progress_bar_html(pct_progress, status)
    html = clean_html(f"""
    <div style="background-color: var(--pg-surface); border: 1px solid var(--pg-border); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; box-shadow: var(--pg-shadow-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: var(--pg-muted); letter-spacing: 0.03em;">{label}</span>
            {chip}
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: var(--pg-text); font-variant-numeric: tabular-nums;">{value}</div>
        {prog}
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def result_verdict_card(probability: float, threshold: float, failure_mode_hint: str = "") -> None:
    """Render high-contrast prediction result card with active threshold mark."""
    pct = probability * 100.0
    thresh_pct = threshold * 100.0

    if probability >= threshold:
        verdict_cls = "verdict-failure"
        chip = f'{icon_html("error_outline", "icon-danger")} ' + chip_html("Failure Likely", "danger")
        prob_color = "var(--pg-danger)"
    elif probability >= 0.25:
        verdict_cls = "verdict-elevated"
        chip = f'{icon_html("warning_amber", "icon-warning")} ' + chip_html("Elevated Risk", "warning")
        prob_color = "var(--pg-warning)"
    else:
        verdict_cls = "verdict-nominal"
        chip = f'{icon_html("check_circle", "icon-success")} ' + chip_html("Nominal Operation", "success")
        prob_color = "var(--pg-success)"


    reason_block = ""
    if failure_mode_hint:
        reason_block = f"""
        <div style="margin-top: 12px; padding: 10px 12px; background-color: var(--pg-surface-subtle); border: 1px solid var(--pg-border-subtle); border-radius: 6px; font-size: 0.83rem; color: var(--pg-text);">
            <strong style="color: var(--pg-text);">Physical Diagnosis:</strong> {failure_mode_hint}
        </div>
        """

    # Marker position for decision threshold on bar
    html = clean_html(f"""
    <div class="pg-result-card {verdict_cls}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.76rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--pg-muted);">Calibrated Breakdown Risk</span>
            {chip}
        </div>
        <div style="font-size: 2.8rem; font-weight: 800; color: {prob_color}; line-height: 1.1; margin: 8px 0 4px; font-variant-numeric: tabular-nums;">
            {pct:.1f}%
        </div>
        <div style="font-size: 0.80rem; color: var(--pg-muted); margin-bottom: 12px;">
            Active operating threshold: <strong style="color: var(--pg-text);">{thresh_pct:.1f}%</strong>
        </div>
        <!-- Progress bar with threshold marker -->
        <div style="position: relative; height: 8px; width: 100%; background-color: var(--pg-border-subtle); border-radius: 4px; overflow: visible; margin-bottom: 6px;">
            <div style="height: 100%; width: {min(100.0, pct):.1f}%; background-color: {prob_color}; border-radius: 4px;"></div>
            <div style="position: absolute; left: {min(98.0, thresh_pct):.1f}%; top: -3px; bottom: -3px; width: 2px; background-color: var(--pg-text); z-index: 2;" title="Decision threshold ({thresh_pct:.1f}%)"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.70rem; color: var(--pg-muted);">
            <span>0% Safe</span>
            <span>Threshold: {thresh_pct:.1f}%</span>
            <span>100% Critical</span>
        </div>
        {reason_block}
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_sensor_control(
    k: str,
    cfg: Dict[str, Any],
    on_slider_change: Callable,
    on_number_change: Callable,
) -> None:
    """Render a compact sensor input widget with label, click-to-type number box, slider, and bounds."""
    curr_val = st.session_state.get(f"val_{k}", cfg["default"])
    curr_min = st.session_state.get(f"min_{k}", cfg["ds_min"])
    curr_max = st.session_state.get(f"max_{k}", cfg["ds_max"])
    step = cfg["step"]
    fmt = cfg["fmt"]
    ds_min = cfg["ds_min"]
    ds_max = cfg["ds_max"]
    typ_min = cfg["typ_min"]
    typ_max = cfg["typ_max"]
    err = st.session_state.get(f"err_{k}")

    is_int = (fmt == "%d")
    if is_int:
        val_casted = int(round(float(curr_val)))
        min_casted = int(round(float(curr_min)))
        max_casted = int(round(float(curr_max)))
        step_casted = int(round(float(step)))
        floor_min = 0
    else:
        val_casted = float(curr_val)
        min_casted = float(curr_min)
        max_casted = float(curr_max)
        step_casted = float(step)
        floor_min = 0.0

    # Top row: Label on left, number box + unit on right
    c_l, c_n, c_u = st.columns([1.5, 1.2, 0.4])
    with c_l:
        st.markdown(f'<div class="sensor-label" style="padding-top:7px;">{cfg["label"]}</div>', unsafe_allow_html=True)
    with c_n:
        st.number_input(
            label=cfg["label"],
            value=val_casted,
            min_value=floor_min,
            step=step_casted,
            format=fmt,
            label_visibility="collapsed",
            key=f"num_{k}",
            on_change=on_number_change,
            args=(k,),
        )
    with c_u:
        st.markdown(f'<div class="sensor-unit" style="padding-top:8px;">{cfg["unit"]}</div>', unsafe_allow_html=True)

    # Slider
    st.slider(
        label=cfg["label"],
        min_value=min_casted,
        max_value=max_casted,
        value=val_casted,
        step=step_casted,
        label_visibility="collapsed",
        key=f"slider_{k}",
        on_change=on_slider_change,
        args=(k,),
    )

    # Sub-slider metadata row
    min_cls = "sub-slider-amber" if curr_min < ds_min else ""
    max_cls = "sub-slider-amber" if curr_max > ds_max else ""
    min_str = f"{curr_min:.1f}" if step < 1 else f"{int(curr_min)}"
    max_str = f"{curr_max:.1f}" if step < 1 else f"{int(curr_max)}"
    typ_str = f"typical {typ_min:g}–{typ_max:g}"
    
    sub_html = clean_html(f"""
    <div class="sub-slider-row">
        <span class="sub-slider-end {min_cls}">{min_str}</span>
        <span class="sub-slider-typ">{typ_str}</span>
        <span class="sub-slider-end {max_cls}">{max_str}</span>
    </div>
    """)
    st.markdown(sub_html, unsafe_allow_html=True)

    # Inline Validation & Warnings
    if err:
        st.markdown(
            f'<div class="sensor-error">{icon_html("error", "icon-danger", style="font-size:13px;")} <span>{err}</span></div>',
            unsafe_allow_html=True,
        )
    elif curr_val < ds_min or curr_val > ds_max:
        st.markdown(
            f'<div class="sensor-warning">{icon_html("warning", "icon-warning", style="font-size:13px;")} <span>Outside training range {ds_min:g}–{ds_max:g}. Prediction extrapolates.</span></div>',
            unsafe_allow_html=True,
        )
    elif curr_val < typ_min or curr_val > typ_max:
        st.markdown(
            f'<div class="sensor-notice">{icon_html("info", "icon-muted", style="font-size:13px;")} <span>Outside typical range</span></div>',
            unsafe_allow_html=True,
        )


def render_stat_cards(phys: Dict[str, Any]) -> None:
    """Render the 3 prominent compact physics stat cards side-by-side."""
    td = phys.get("temp_diff", 0.0)
    pw = phys.get("power_w", 0.0)
    wt = phys.get("wear_torque", 0.0)
    html = clean_html(f"""
    <div class="stat-cards-grid">
        <div class="stat-card">
            <div class="stat-card-title">Temp delta (K)</div>
            <div class="stat-card-val">{td:.1f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Power (W)</div>
            <div class="stat-card-val">{pw:,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Wear &times; torque</div>
            <div class="stat-card-val">{wt:,.0f}</div>
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_risk_gauge_card(
    prob: float,
    threshold: float,
    threshold_mode: str,
    thresh_cfg: Dict[str, Any],
    phys: Dict[str, Any],
) -> None:
    """Render high-contrast risk gauge card with threshold marker and physics diagnosis."""
    pct = prob * 100.0
    thresh_pct = threshold * 100.0
    
    if prob >= threshold:
        badge_cls = "danger"
        badge_text = "Failure Predicted"
        badge_icon = icon_html("error_outline", "icon-danger")
        prob_color = "var(--pg-danger)"
    elif prob >= 0.25:
        badge_cls = "warning"
        badge_text = "Elevated Risk"
        badge_icon = icon_html("warning_amber", "icon-warning")
        prob_color = "var(--pg-warning)"
    else:
        badge_cls = "success"
        badge_text = "Nominal Operation"
        badge_icon = icon_html("check_circle", "icon-success")
        prob_color = "var(--pg-success)"
        
    mode_titles = {
        "optimal_f1": "Max-F1 (Balanced)",
        "high_recall": "High-Recall (Safety-First)",
        "custom": "Custom Override",
    }
    mode_name = mode_titles.get(threshold_mode, threshold_mode)
    
    diag_hint = ""
    if prob >= threshold or prob >= 0.25:
        reasons = []
        if phys.get("hdf_risk"):
            reasons.append("Heat dissipation deficit (convective cooling stall)")
        if phys.get("pwf_risk"):
            reasons.append("Power excursion outside safe [3.5kW, 9.0kW] envelope")
        if phys.get("overstrain_ratio", 0) > 1.0:
            reasons.append("Mechanical torque-wear overstrain capacity breach")
        if phys.get("in_twf_band"):
            reasons.append("Tool wear reached critical stochastic band [200-240 min]")
        if reasons:
            diag_hint = f"""
            <div style="margin-top: 8px; padding: 6px 10px; background-color: var(--pg-surface-subtle); border: 1px solid var(--pg-border-subtle); border-radius: 6px; font-size: 0.74rem; color: var(--pg-text);">
                <strong style="color: var(--pg-text);">Physics Diagnosis:</strong> {'; '.join(reasons)}
            </div>
            """

    html = clean_html(f"""
    <div class="risk-gauge-container">
        <div class="risk-header-row">
            <span class="risk-header-title">Calibrated Failure Risk</span>
            <div style="display:flex; align-items:center; gap:4px;">
                {badge_icon} <span class="pg-chip pg-chip-{badge_cls}">{badge_text}</span>
            </div>
        </div>
        <div class="risk-big-val" style="color: {prob_color};">
            {pct:.1f}%
        </div>
        <div class="risk-thresh-caption">
            Operating Mode: <strong style="color:var(--pg-text);">{mode_name}</strong> &middot; Threshold: <strong style="color:var(--pg-text);">{thresh_pct:.1f}%</strong>
        </div>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width: {min(100.0, pct):.1f}%; background-color: {prob_color};"></div>
            <div class="risk-bar-marker" style="left: {min(98.5, thresh_pct):.1f}%;" title="Decision threshold: {thresh_pct:.1f}%"></div>
        </div>
        <div class="risk-bar-scale">
            <span>0% Safe</span>
            <span>Threshold: {thresh_pct:.1f}%</span>
            <span>100% Critical</span>
        </div>
        {diag_hint}
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_decision_support(human: Dict[str, Any]) -> None:
    """Render concise decision support recommendation card."""
    action = human.get("recommended_maintenance_action", "Nominal operation.")
    action_clean = action.split("(Decision Support): ")[-1] if "(Decision Support):" in action else action
    action_type = action.split(":")[0].split("(")[0].strip()
    
    icon_map = {
        "CRITICAL": icon_html("error_outline", "icon-danger"),
        "WARNING": icon_html("warning_amber", "icon-warning"),
        "ADVISORY": icon_html("info", "icon-accent"),
        "NOMINAL": icon_html("check_circle", "icon-success"),
    }
    icon = icon_map.get(action_type, icon_html("info", "icon-muted"))
    
    html = clean_html(f"""
    <div style="background-color:var(--pg-surface); border:1px solid var(--pg-border); border-radius:8px; padding:10px 14px; box-shadow:var(--pg-shadow-sm); margin-bottom:8px;">
        <div style="font-size:0.72rem; font-weight:700; color:var(--pg-muted); letter-spacing:0.04em; text-transform:uppercase; margin-bottom:4px; display:flex; align-items:center; gap:5px;">
            <span class="material-symbols-outlined" style="font-size:1rem; color:var(--pg-muted);">build</span>
            <span>Recommended Maintenance Action</span>
        </div>
        <div style="font-size:0.84rem; color:var(--pg-text); line-height:1.45; display:flex; align-items:flex-start; gap:6px;">
            <span style="margin-top:1px;">{icon}</span>
            <span><strong style="color:var(--pg-text);">{action_type}:</strong> {action_clean}</span>
        </div>
        <div style="font-size:0.68rem; color:var(--pg-faint); margin-top:5px; font-style:italic;">
            Decision support only. Operators retain full authority.
        </div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)
