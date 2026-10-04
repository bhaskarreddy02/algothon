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
}


def get_theme(mode: str = "light") -> Dict[str, str]:
    return DARK if mode.lower() == "dark" else LIGHT


# -----------------------------------------------------------------------------
# GLOBAL CSS INJECTION
# -----------------------------------------------------------------------------

def get_css(t: Dict[str, str]) -> str:
    """Generate comprehensive CSS variables and Streamlit overrides."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..24,300..400,0,0&display=swap');

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
.stApp, [data-testid="stAppViewContainer"], .main {{
    background-color: var(--pg-bg) !important;
    color: var(--pg-text) !important;
}}

[data-testid="stHeader"] {{
    background-color: transparent !important;
}}

#MainMenu, footer, .stDeployButton {{
    display: none !important;
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
</style>
"""


def apply_theme(mode: str = "light") -> Dict[str, str]:
    """
    Inject theme CSS and set matplotlib rcParams to match mode.
    Returns the theme token dictionary for charts and custom HTML.
    """
    t = get_theme(mode)
    st.markdown(get_css(t), unsafe_allow_html=True)

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
