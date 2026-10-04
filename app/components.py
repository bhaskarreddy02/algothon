"""
PredictiveGuard UI Components (app/components.py)
Reusable, accessible UI components for Streamlit without emojis or gradients.
"""

from typing import List, Optional
import streamlit as st


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
    html = f"""
    <div class="pg-page-header">
        {badge_html}
        <h1 class="pg-page-title" style="display: flex; align-items: center;">{icon_prefix}<span>{title}</span></h1>
        <p class="pg-page-desc">{description}</p>
    </div>
    """
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
    html = f"""
    <div style="background-color: var(--pg-surface); border: 1px solid var(--pg-border); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; box-shadow: var(--pg-shadow-sm);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.74rem; font-weight: 600; text-transform: uppercase; color: var(--pg-muted); letter-spacing: 0.03em;">{label}</span>
            {chip}
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: var(--pg-text); font-variant-numeric: tabular-nums;">{value}</div>
        {prog}
    </div>
    """
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
    html = f"""
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
    """
    st.markdown(html, unsafe_allow_html=True)
