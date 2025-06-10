import streamlit as st
import pickle
import os


def session_sidebar() -> None:
    """Displays sidebar buttons to save, load, or clear Streamlit session state."""
    st.sidebar.markdown("## 🔧 Session Management")

    if st.sidebar.button("💾 Save"):
        with open(os.getenv("SESSION_STATE_FILE"), "wb") as f:
            pickle.dump(dict(st.session_state), f)
        st.sidebar.success("Session saved!")

    if st.sidebar.button("📂 Load"):
        try:
            with open(os.getenv("SESSION_STATE_FILE"), "rb") as f:
                data = pickle.load(f)
                for k, v in data.items():
                    st.session_state[k] = v
            st.sidebar.success("Session loaded!")
        except FileNotFoundError:
            st.sidebar.warning("No saved session found.")

    if st.sidebar.button("🗑️ Clear"):
        if os.path.exists(os.getenv("SESSION_STATE_FILE")):
            os.remove(os.getenv("SESSION_STATE_FILE"))
            st.sidebar.success("Session cleared.")

def configure_layout() -> None:
    """Configures the initial Streamlit page layout."""
    st.set_page_config(
        page_title="AML - U-Net Semantic Segmentation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
