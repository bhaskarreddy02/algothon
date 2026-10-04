"""
PredictiveGuard Root Entrypoint for Streamlit Cloud & Hosting Platforms.
Delegates directly to app/streamlit_app.py.
"""

import sys
from pathlib import Path
import streamlit as st

# Must be the very first Streamlit command in the root execution script
st.set_page_config(
    page_title="PredictiveGuard | ALGOTHON26",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.streamlit_app import main

if __name__ == "__main__":
    main()
