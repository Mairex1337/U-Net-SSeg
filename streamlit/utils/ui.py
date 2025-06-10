import streamlit as st
import pickle
import os

SESSION_STATE_FILE = "streamlit/session_state.pkl" #TODO: Refactor to config

def session_sidebar():
    st.sidebar.markdown("## 🔧 Session Management")

    if st.sidebar.button("💾 Save"):
        with open(SESSION_STATE_FILE, "wb") as f:
            pickle.dump(dict(st.session_state), f)
        st.sidebar.success("Session saved!")

    if st.sidebar.button("📂 Load"):
        try:
            with open(SESSION_STATE_FILE, "rb") as f:
                data = pickle.load(f)
                for k, v in data.items():
                    st.session_state[k] = v
            st.sidebar.success("Session loaded!")
        except FileNotFoundError:
            st.sidebar.warning("No saved session found.")

    if st.sidebar.button("🗑️ Clear"):
        if os.path.exists(SESSION_STATE_FILE):
            os.remove(SESSION_STATE_FILE)
            st.sidebar.success("Session cleared.")

def configure_layout():
    st.set_page_config(
        page_title="AML - U-Net Semantic Segmentation",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
