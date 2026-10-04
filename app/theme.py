"""
PredictiveGuard Theme Engine (app/theme.py)
Centralized tokens, WCAG-compliant styling, and chart theming for Light & Dark modes.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------------------------------------------------------
# PALETTE TOKENS
# -----------------------------------------------------------------------------

LIGHT: Dict[str, str] = {
    "mode": "light",
    "bg": "#FAF8F5",
    "surface": "#FFFFFF",
    "surface_subtle": "#F4EFEA",
    "surface_card": "#FFFFFF",
    "border": "#E7E2DA",
    "border_subtle": "#ECE8E1",
    "border_strong": "#D5CEBF",
    "text": "#1C1917",
    "muted": "#6B645C",
    "faint": "#9E978E",
    "accent": "#E8590C",         # Burnt orange
    "accent_hover": "#D04804",
    "accent_subtle": "rgba(232, 89, 12, 0.08)",
    "accent_border": "rgba(232, 89, 12, 0.25)",
    "success": "#2F9E44",
    "success_subtle": "rgba(47, 158, 68, 0.08)",
    "success_border": "rgba(47, 158, 68, 0.25)",
    "warning": "#F08C00",
    "warning_subtle": "rgba(240, 140, 0, 0.08)",
    "warning_border": "rgba(240, 140, 0, 0.25)",
    "danger": "#E03131",
    "danger_subtle": "rgba(224, 49, 49, 0.08)",
    "danger_border": "rgba(224, 49, 49, 0.25)",
    "sidebar_bg": "#F5F1EA",
    "sidebar_border": "#E2DCD2",
    "input_bg": "#FFFFFF",
    "input_border": "#D5CEBF",
    "input_focus_border": "#E8590C",
    "metric_bg": "#FFFFFF",
    "code_bg": "#EFEAE1",
    "grid": "#EAE5DC",
    "shadow_sm": "0 1px 2px rgba(28, 25, 23, 0.04)",
    "shadow_md": "0 4px 6px -1px rgba(28, 25, 23, 0.06), 0 2px 4px -2px rgba(28, 25, 23, 0.04)",
    "seg_track_bg": "#EFEBE4",
    "seg_track_border": "#E2DCD2",
    "seg_unselected_text": "#6B655C",
    "seg_hover_bg": "#E7E1D7",
    "seg_hover_text": "#2B2723",
    "seg_selected_bg": "#FFFFFF",
    "seg_selected_text": "#E8590C",
    "seg_selected_border": "#E8754A",
    "seg_selected_shadow": "0 1px 3px rgba(43, 39, 35, 0.08)",
}

DARK: Dict[str, str] = {
    "mode": "dark",
    "bg": "#121212",             # Neutral charcoal (not navy)
    "surface": "#1C1C1C",
    "surface_subtle": "#252525",
    "surface_card": "#1E1E1E",
    "border": "#2E2E2E",
    "border_subtle": "#242424",
    "border_strong": "#3D3D3D",
    "text": "#F2EFEA",
    "muted": "#A39E96",
    "faint": "#66625C",
    "accent": "#FF8A3D",         # Adjusted for dark background contrast
    "accent_hover": "#FFA35E",
    "accent_subtle": "rgba(255, 138, 61, 0.12)",
    "accent_border": "rgba(255, 138, 61, 0.35)",
    "success": "#40C057",
    "success_subtle": "rgba(64, 192, 87, 0.12)",
    "success_border": "rgba(64, 192, 87, 0.35)",
    "warning": "#FAB005",
    "warning_subtle": "rgba(250, 176, 5, 0.12)",
    "warning_border": "rgba(250, 176, 5, 0.35)",
    "danger": "#FA5252",
    "danger_subtle": "rgba(250, 82, 82, 0.12)",
    "danger_border": "rgba(250, 82, 82, 0.35)",
    "sidebar_bg": "#171717",
    "sidebar_border": "#2B2B2B",
    "input_bg": "#222222",
    "input_border": "#383838",
    "input_focus_border": "#FF8A3D",
    "metric_bg": "#1C1C1C",
    "code_bg": "#222222",
    "grid": "#262626",
    "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.4)",
    "shadow_md": "0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -2px rgba(0, 0, 0, 0.4)",
    "seg_track_bg": "#1C1C1C",
    "seg_track_border": "#2E2E2E",
    "seg_unselected_text": "#A39E96",
    "seg_hover_bg": "#252525",
    "seg_hover_text": "#F2EFEA",
    "seg_selected_bg": "#222222",
    "seg_selected_text": "#FF8A3D",
    "seg_selected_border": "#FF8A3D",
    "seg_selected_shadow": "0 1px 3px rgba(0, 0, 0, 0.3)",
}


def get_theme(mode: str = "light") -> Dict[str, str]:
    return DARK if mode.lower() == "dark" else LIGHT


# -----------------------------------------------------------------------------
# GLOBAL CSS INJECTION
# -----------------------------------------------------------------------------

def get_css(t: Dict[str, str]) -> str:
    """Generate comprehensive CSS variables and Streamlit overrides without blank lines."""
    raw = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

.material-symbols-outlined {{
    font-family: 'Material Symbols Outlined' !important;
    font-weight: 300 !important;
    font-style: normal !important;
    font-size: 1.2rem !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    vertical-align: -0.15em !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-smoothing: antialiased !important;
}}
.icon-inline {{
    margin-right: 6px;
    font-size: 1.15rem !important;
}}
.icon-lg {{
    font-size: 2.4rem !important;
}}
.icon-danger {{ color: var(--pg-danger) !important; }}
.icon-warning {{ color: var(--pg-warning) !important; }}
.icon-success {{ color: var(--pg-success) !important; }}
.icon-accent {{ color: var(--pg-accent) !important; }}
.icon-muted {{ color: var(--pg-muted) !important; }}

.section-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid var(--pg-border);
    padding-bottom: 8px;
    margin: 20px 0 14px;
}}
.section-header h3, .section-header .section-title {{
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: var(--pg-text) !important;
    margin: 0 !important;
    letter-spacing: -0.01em !important;
}}


:root {{
    --pg-bg: {t['bg']};
    --pg-surface: {t['surface']};
    --pg-surface-subtle: {t['surface_subtle']};
    --pg-surface-card: {t['surface_card']};
    --pg-border: {t['border']};
    --pg-border-subtle: {t['border_subtle']};
    --pg-border-strong: {t['border_strong']};
    --pg-text: {t['text']};
    --pg-muted: {t['muted']};
    --pg-faint: {t['faint']};
    --pg-accent: {t['accent']};
    --pg-accent-hover: {t['accent_hover']};
    --pg-accent-subtle: {t['accent_subtle']};
    --pg-accent-border: {t['accent_border']};
    --pg-success: {t['success']};
    --pg-success-subtle: {t['success_subtle']};
    --pg-success-border: {t['success_border']};
    --pg-warning: {t['warning']};
    --pg-warning-subtle: {t['warning_subtle']};
    --pg-warning-border: {t['warning_border']};
    --pg-danger: {t['danger']};
    --pg-danger-subtle: {t['danger_subtle']};
    --pg-danger-border: {t['danger_border']};
    --pg-sidebar-bg: {t['sidebar_bg']};
    --pg-sidebar-border: {t['sidebar_border']};
    --pg-input-bg: {t['input_bg']};
    --pg-input-border: {t['input_border']};
    --pg-input-focus: {t['input_focus_border']};
    --pg-code-bg: {t['code_bg']};
    --pg-shadow-sm: {t['shadow_sm']};
    --pg-shadow-md: {t['shadow_md']};
}}

/* Universal font reset */
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--pg-text) !important;
    background-color: var(--pg-bg) !important;
}}

/* Tabular figures for all numbers and metrics */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"], .tabular-nums, code, pre {{
    font-variant-numeric: tabular-nums !important;
    font-feature-settings: "tnum" 1 !important;
}}

/* App Container & Background */
.stApp {{
    background-color: var(--pg-bg) !important;
    color: var(--pg-text) !important;
}}

[data-testid="stAppViewContainer"], .main {{
    background-color: transparent !important;
    color: var(--pg-text) !important;
}}

[data-testid="stHeader"] {{
    background-color: transparent !important;
}}

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], header[data-testid="stHeader"] .stDeployButton {{
    display: none !important;
}}

.main .block-container {{
    position: relative !important;
    z-index: 2 !important;
    padding-top: 1.0rem !important;
    padding-bottom: 1.2rem !important;
    max-width: 98% !important;
    width: 98% !important;
}}

/* 3D Digital Twin Background Canvas */
#pg-digital-twin-canvas {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 0 !important;
    pointer-events: none !important;
    display: block !important;
}}

.st-key-pg_3d_bg {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
    z-index: 0 !important;
}}

.st-key-pg_3d_bg iframe {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    pointer-events: none !important;
    background: transparent !important;
    z-index: 0 !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: var(--pg-sidebar-bg) !important;
    border-right: 1px solid var(--pg-sidebar-border) !important;
}}

[data-testid="stSidebar"] * {{
    color: var(--pg-text);
}}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: var(--pg-text) !important;
}}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {{
    color: var(--pg-text) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
}}

/* Sidebar Radios */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {{
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    padding: 6px 10px !important;
    margin-bottom: 2px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
    background-color: var(--pg-surface-subtle) !important;
    border-color: var(--pg-border-subtle) !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
    background-color: var(--pg-surface) !important;
    border-color: var(--pg-border) !important;
    box-shadow: var(--pg-shadow-sm) !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label span {{
    color: var(--pg-text) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}}

/* Inputs (Number, Text, etc.) */
input, textarea, [data-baseweb="input"] {{
    background-color: var(--pg-input-bg) !important;
    color: var(--pg-text) !important;
    border-color: var(--pg-input-border) !important;
    border-radius: 6px !important;
}}

[data-baseweb="input"] input {{
    background-color: transparent !important;
    color: var(--pg-text) !important;
}}

[data-baseweb="input"]:focus-within {{
    border-color: var(--pg-input-focus) !important;
    box-shadow: 0 0 0 1px var(--pg-input-focus) !important;
}}

/* Click-to-type number box styling */
[data-testid="stNumberInput"] button {{
    display: none !important;
}}
[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
[data-testid="stNumberInput"] input::-webkit-inner-spin-button {{
    -webkit-appearance: none !important;
    margin: 0 !important;
}}
[data-testid="stNumberInput"] input[type=number] {{
    -moz-appearance: textfield !important;
}}
[data-testid="stNumberInput"] input {{
    text-align: center !important;
    font-variant-numeric: tabular-nums !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    height: 36px !important;
    padding: 2px 4px !important;
    border-radius: 6px !important;
}}
[data-testid="stNumberInput"] {{
    width: 100% !important;
    margin-bottom: 0px !important;
}}

/* Selectbox & Dropdowns */
[data-baseweb="select"] > div {{
    background-color: var(--pg-input-bg) !important;
    border-color: var(--pg-input-border) !important;
    color: var(--pg-text) !important;
    border-radius: 6px !important;
}}

[data-baseweb="select"] * {{
    color: var(--pg-text) !important;
}}

[data-baseweb="popover"], [data-baseweb="menu"] {{
    background-color: var(--pg-surface) !important;
    border: 1px solid var(--pg-border) !important;
    border-radius: 8px !important;
    box-shadow: var(--pg-shadow-md) !important;
}}

[data-baseweb="menu"] li:hover {{
    background-color: var(--pg-surface-subtle) !important;
}}

/* Segmented Control & Button Group (Option 1: Soft Pill) */
[data-testid="stButtonGroup"],
.st-key-top_nav,
.st-key-seg_machine_type {{
    background: transparent !important;
}}

[data-testid="stButtonGroup"] > div,
[data-testid="stButtonGroup"] [role="radiogroup"],
[data-testid="stButtonGroup"] div:has(> button),
.st-key-top_nav > div,
.st-key-seg_machine_type > div {{
    background-color: {t['seg_track_bg']} !important;
    border: 1px solid {t['seg_track_border']} !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 3px !important;
    display: inline-flex !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    white-space: nowrap !important;
}}

.st-key-top_nav {{
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    overflow-x: auto !important;
    margin-bottom: 8px !important;
}}

.st-key-top_nav > div {{
    min-width: max-content !important;
}}

.st-key-top_nav button {{
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}}

[data-testid="stButtonGroup"] button,
button[data-variant="segmented_control"],
button[kind="segmented_control"],
[data-testid="stButtonGroup"] [role="radiogroup"] button {{
    background-color: transparent !important;
    background: transparent !important;
    color: {t['seg_unselected_text']} !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 5px 14px !important;
    box-shadow: none !important;
    transition: all 0.15s ease-in-out !important;
}}

[data-testid="stButtonGroup"] button p,
[data-testid="stButtonGroup"] button span,
[data-testid="stButtonGroup"] button div,
button[data-variant="segmented_control"] p,
button[data-variant="segmented_control"] span,
button[data-variant="segmented_control"] div,
button[kind="segmented_control"] p,
button[kind="segmented_control"] span {{
    color: {t['seg_unselected_text']} !important;
    font-weight: 500 !important;
}}

[data-testid="stButtonGroup"] button:hover,
button[data-variant="segmented_control"]:hover,
button[kind="segmented_control"]:hover,
[data-testid="stButtonGroup"] button[data-hovered="true"],
[data-testid="stButtonGroup"] [role="radiogroup"] button:hover {{
    background-color: {t['seg_hover_bg']} !important;
    background: {t['seg_hover_bg']} !important;
    color: {t['seg_hover_text']} !important;
    border-color: transparent !important;
}}

[data-testid="stButtonGroup"] button:hover p,
[data-testid="stButtonGroup"] button:hover span,
[data-testid="stButtonGroup"] button:hover div,
button[data-variant="segmented_control"]:hover p,
button[data-variant="segmented_control"]:hover span,
button[kind="segmented_control"]:hover p,
button[kind="segmented_control"]:hover span {{
    color: {t['seg_hover_text']} !important;
    font-weight: 500 !important;
}}

[data-testid="stButtonGroup"] button[data-selected="true"],
[data-testid="stButtonGroup"] button[data-selected],
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
button[data-variant="segmented_control"][data-selected],
button[data-variant="segmented_control"][data-selected="true"],
button[kind="segmented_controlActive"] {{
    background-color: {t['seg_selected_bg']} !important;
    background: {t['seg_selected_bg']} !important;
    color: {t['seg_selected_text']} !important;
    border: 1px solid {t['seg_selected_border']} !important;
    border-radius: 6px !important;
    box-shadow: {t['seg_selected_shadow']} !important;
    font-weight: 600 !important;
}}

[data-testid="stButtonGroup"] button[data-selected="true"] p,
[data-testid="stButtonGroup"] button[data-selected="true"] span,
[data-testid="stButtonGroup"] button[data-selected="true"] div,
[data-testid="stButtonGroup"] button[data-selected] p,
[data-testid="stButtonGroup"] button[data-selected] span,
[data-testid="stButtonGroup"] button[data-selected] div,
[data-testid="stButtonGroup"] button[aria-checked="true"] p,
[data-testid="stButtonGroup"] button[aria-checked="true"] span,
[data-testid="stButtonGroup"] button[aria-pressed="true"] p,
[data-testid="stButtonGroup"] button[aria-pressed="true"] span,
button[data-variant="segmented_control"][data-selected] p,
button[data-variant="segmented_control"][data-selected] span,
button[data-variant="segmented_control"][data-selected="true"] p,
button[data-variant="segmented_control"][data-selected="true"] span,
button[kind="segmented_controlActive"] p,
button[kind="segmented_controlActive"] span {{
    color: {t['seg_selected_text']} !important;
    font-weight: 600 !important;
}}

/* Scenario Presets Bar */
.scenario-bar .stButton > button {{
    background-color: var(--pg-surface) !important;
    color: var(--pg-text) !important;
    border: 1px solid var(--pg-border-strong) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 0.8rem !important;
    box-shadow: var(--pg-shadow-sm) !important;
    transition: all 0.15s ease-in-out !important;
}}

.scenario-bar .stButton > button:hover {{
    background-color: var(--pg-surface-subtle) !important;
    border-color: var(--pg-accent) !important;
    color: var(--pg-text) !important;
}}

.scenario-bar .stButton > button[kind="primary"],
.scenario-bar .stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: #FFFFFF !important;
    color: var(--pg-accent) !important;
    border: 1.5px solid var(--pg-accent) !important;
    box-shadow: 0 1px 3px rgba(43, 39, 35, 0.08) !important;
    font-weight: 600 !important;
}}

/* Buttons */
.stButton > button {{
    background-color: var(--pg-surface) !important;
    color: var(--pg-text) !important;
    border: 1px solid var(--pg-border-strong) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    box-shadow: var(--pg-shadow-sm) !important;
    transition: all 0.15s ease !important;
}}

.stButton > button:hover {{
    background-color: var(--pg-surface-subtle) !important;
    border-color: var(--pg-text) !important;
    color: var(--pg-text) !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: var(--pg-accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--pg-accent) !important;
    box-shadow: var(--pg-shadow-sm) !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background-color: var(--pg-accent-hover) !important;
    border-color: var(--pg-accent-hover) !important;
    color: #FFFFFF !important;
}}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] {{
    color: var(--pg-accent) !important;
}}

[data-testid="stSlider"] div[role="slider"] {{
    background-color: var(--pg-accent) !important;
    border-color: var(--pg-accent) !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background-color: var(--pg-surface) !important;
    border: 1px solid var(--pg-border) !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: var(--pg-shadow-sm) !important;
}}

[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {{
    color: var(--pg-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}}

[data-testid="stMetricValue"] {{
    color: var(--pg-text) !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
}}

/* Tabs */
[data-testid="stTabs"] button {{
    color: var(--pg-muted) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
}}

[data-testid="stTabs"] button[aria-selected="true"] {{
    color: var(--pg-accent) !important;
    border-bottom: 2px solid var(--pg-accent) !important;
    font-weight: 600 !important;
}}

/* Expanders */
[data-testid="stExpander"] {{
    background-color: var(--pg-surface) !important;
    border: 1px solid var(--pg-border) !important;
    border-radius: 8px !important;
}}

[data-testid="stExpander"] details {{
    background-color: var(--pg-surface) !important;
    border-radius: 8px !important;
}}

[data-testid="stExpander"] summary {{
    background-color: var(--pg-surface) !important;
    color: var(--pg-text) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}}

[data-testid="stExpander"] summary:hover {{
    color: var(--pg-accent) !important;
}}

[data-testid="stExpander"] div[role="region"] {{
    background-color: var(--pg-surface) !important;
    border-top: 1px solid var(--pg-border-subtle) !important;
    padding: 12px 14px !important;
}}

/* DataFrames & Tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid var(--pg-border) !important;
    border-radius: 8px !important;
    background-color: var(--pg-surface) !important;
}}

/* File Uploader */
[data-testid="stFileUploader"] section {{
    background-color: var(--pg-surface-subtle) !important;
    border: 1px dashed var(--pg-border-strong) !important;
    border-radius: 8px !important;
}}

[data-testid="stFileUploader"] section * {{
    color: var(--pg-text) !important;
}}

/* Segmented Control / Pills */
div[data-testid="stSegmentedControl"] button {{
    border: 1px solid var(--pg-border) !important;
    background-color: var(--pg-surface) !important;
    color: var(--pg-muted) !important;
    font-size: 0.85rem !important;
}}

div[data-testid="stSegmentedControl"] button[aria-checked="true"] {{
    background-color: var(--pg-accent) !important;
    color: #FFFFFF !important;
    border-color: var(--pg-accent) !important;
    font-weight: 600 !important;
}}

/* Page Header component */
.pg-page-header {{
    margin-bottom: 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--pg-border);
}}

.pg-page-title {{
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--pg-text);
    margin: 0;
    letter-spacing: -0.01em;
}}

.pg-page-desc {{
    font-size: 0.90rem;
    color: var(--pg-muted);
    margin: 4px 0 0 0;
}}

/* Card containers */
.pg-card {{
    background-color: var(--pg-surface);
    border: 1px solid var(--pg-border);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: var(--pg-shadow-sm);
}}

.pg-card-header {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--pg-text);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--pg-border-subtle);
}}

/* Chips & Badges */
.pg-chip {{
    display: inline-flex;
    align-items: center;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.02em;
    border: 1px solid;
}}

.pg-chip-neutral {{
    background-color: var(--pg-surface-subtle);
    border-color: var(--pg-border);
    color: var(--pg-muted);
}}

.pg-chip-success {{
    background-color: var(--pg-success-subtle);
    border-color: var(--pg-success-border);
    color: var(--pg-success);
}}

.pg-chip-warning {{
    background-color: var(--pg-warning-subtle);
    border-color: var(--pg-warning-border);
    color: var(--pg-warning);
}}

.pg-chip-danger {{
    background-color: var(--pg-danger-subtle);
    border-color: var(--pg-danger-border);
    color: var(--pg-danger);
}}

.pg-chip-accent {{
    background-color: var(--pg-accent-subtle);
    border-color: var(--pg-accent-border);
    color: var(--pg-accent);
}}

/* Result Card */
.pg-result-card {{
    background-color: var(--pg-surface);
    border: 1px solid var(--pg-border);
    border-radius: 8px;
    padding: 20px 22px;
    box-shadow: var(--pg-shadow-sm);
}}

.pg-result-card.verdict-failure {{
    border-left: 4px solid var(--pg-danger);
}}

.pg-result-card.verdict-elevated {{
    border-left: 4px solid var(--pg-warning);
}}

.pg-result-card.verdict-nominal {{
    border-left: 4px solid var(--pg-success);
}}

/* Progress bars */
.pg-prog-track {{
    height: 4px;
    width: 100%;
    background-color: var(--pg-border-subtle);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 6px;
}}

.pg-prog-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.2s ease;
}}

/* Custom scrollbars */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: var(--pg-bg);
}}
::-webkit-scrollbar-thumb {{
    background: var(--pg-border-strong);
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--pg-muted);
}}

/* Live Dashboard Sensor Card & Controls */
.sensor-panel-card {{
    background-color: var(--pg-surface);
    border: 1px solid var(--pg-border);
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: var(--pg-shadow-sm);
    margin-bottom: 12px;
}}

.machine-type-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--pg-border-subtle);
}}

.machine-type-title {{
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--pg-text);
}}

.sensor-header-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2px;
}}

.sensor-label {{
    font-size: 0.90rem;
    font-weight: 600;
    color: var(--pg-text);
}}

.sensor-unit {{
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--pg-muted);
    display: inline-flex;
    align-items: center;
    margin-left: 2px;
}}

.sub-slider-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    color: var(--pg-muted);
    margin-top: -6px;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}}

.sub-slider-end {{
    font-weight: 600;
    color: var(--pg-muted);
}}

.sub-slider-amber {{
    color: var(--pg-warning) !important;
    font-weight: 700 !important;
}}

.sub-slider-typ {{
    color: var(--pg-faint);
    font-size: 0.70rem;
}}

.sensor-warning {{
    font-size: 0.70rem;
    color: var(--pg-warning);
    background-color: var(--pg-warning-subtle);
    border: 1px solid var(--pg-warning-border);
    border-radius: 4px;
    padding: 2px 6px;
    margin-top: -2px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

.sensor-notice {{
    font-size: 0.70rem;
    color: var(--pg-muted);
    margin-top: -2px;
    margin-bottom: 6px;
    font-style: italic;
}}

.sensor-error {{
    font-size: 0.70rem;
    color: var(--pg-danger);
    background-color: var(--pg-danger-subtle);
    border: 1px solid var(--pg-danger-border);
    border-radius: 4px;
    padding: 2px 6px;
    margin-top: -2px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}}

/* 3 Compact Derived Physics Cards */
.stat-cards-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-top: 14px;
    margin-bottom: 8px;
}}

.stat-card {{
    background-color: var(--pg-surface-subtle);
    border: 1px solid var(--pg-border);
    border-radius: 6px;
    padding: 10px 14px;
}}

.stat-card-title {{
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--pg-muted);
    margin-bottom: 2px;
}}

.stat-card-val {{
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--pg-text);
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}}

/* Scenario preset buttons */
.scenario-bar {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}}

.scenario-label {{
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--pg-muted);
    margin-right: 4px;
}}

/* Right Panel Risk Gauge */
.risk-gauge-container {{
    background-color: var(--pg-surface);
    border: 1px solid var(--pg-border);
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: var(--pg-shadow-sm);
    margin-bottom: 10px;
}}

.risk-header-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}}

.risk-header-title {{
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--pg-muted);
}}

.risk-big-val {{
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 4px 0 2px;
    font-variant-numeric: tabular-nums;
}}

.risk-thresh-caption {{
    font-size: 0.76rem;
    color: var(--pg-muted);
    margin-bottom: 10px;
}}

.risk-bar-track {{
    position: relative;
    height: 8px;
    width: 100%;
    background-color: var(--pg-border-subtle);
    border-radius: 4px;
    margin-bottom: 6px;
}}

.risk-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.15s ease;
}}

.risk-bar-marker {{
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    background-color: var(--pg-text);
    z-index: 2;
}}

.risk-bar-scale {{
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--pg-faint);
}}
"""
    clean_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return "\n".join(clean_lines)


def apply_theme(mode: str = "light") -> Dict[str, str]:
    """
    Inject theme CSS and set matplotlib rcParams to match mode.
    Returns the theme token dictionary for charts and custom HTML.
    """
    t = get_theme(mode)
    font_links = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block">'
    )
    st.markdown(font_links, unsafe_allow_html=True)
    css_content = get_css(t)
    st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)

    # Configure matplotlib globals
    plt.rcParams.update({
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "axes.edgecolor": t["border"],
        "axes.labelcolor": t["text"],
        "text.color": t["text"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "grid.color": t["grid"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "axes.grid": True,
        "grid.alpha": 0.6,
        "grid.linestyle": ":",
    })
    return t


def get_plotly_layout(t: Dict[str, str], height: int = 360) -> Dict[str, Any]:
    """Helper layout config for Plotly charts."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(
            family="Inter, sans-serif",
            color=t["text"],
            size=12,
        ),
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(
            color=t["text"],
            gridcolor=t["grid"],
            linecolor=t["border"],
            zerolinecolor=t["border"],
            tickfont=dict(color=t["muted"]),
        ),
        yaxis=dict(
            color=t["text"],
            gridcolor=t["grid"],
            linecolor=t["border"],
            zerolinecolor=t["border"],
            tickfont=dict(color=t["muted"]),
        ),
        legend=dict(
            font=dict(color=t["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
