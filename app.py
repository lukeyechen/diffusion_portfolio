from pathlib import Path
import streamlit as st
from config import APP_TITLE

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
)

SECTIONS = [
    "Portfolio",
    "Method Comparison",
    "Backtest",
    "Diffusion Diagnostics",
]

if "app_section" not in st.session_state:
    st.session_state["app_section"] = "Portfolio"

current = st.session_state["app_section"]
if current not in SECTIONS:
    current = "Portfolio"
    st.session_state["app_section"] = current

selected = st.sidebar.radio(
    "Navigation",
    SECTIONS,
    index=SECTIONS.index(current),
    key="manual_navigation_radio",
)

if selected != st.session_state["app_section"]:
    st.session_state["app_section"] = selected
    st.rerun()

VIEW_FILES = {
    "Portfolio": "Portfolio.py",
    "Method Comparison": "Method_Comparison.py",
    "Backtest": "Backtest.py",
    "Diffusion Diagnostics": "Diffusion_Diagnostics.py",
}

view_path = Path(__file__).parent / "views" / VIEW_FILES[selected]

# Execute only the selected view. This deliberately avoids Streamlit's
# multipage router (st.Page / st.navigation / pages directory), so there
# cannot be duplicate URL pathnames such as "Backtest".
code = compile(view_path.read_text(encoding="utf-8"), str(view_path), "exec")
exec(code, globals(), globals())
