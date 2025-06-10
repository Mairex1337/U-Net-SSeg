import streamlit as st
from utils import configure_layout, session_sidebar


configure_layout()
session_sidebar()
st.title("📤 Upload Data for Predictions")

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

uploaded_file = st.file_uploader("Upload an image, ZIP file, or video", type=["jpg", "zip", "mp4"])

if uploaded_file and st.button("Save File"):
    st.session_state['uploaded_files'].append(uploaded_file)

    filename = uploaded_file.name
    st.success(f"File '{filename}' uploaded successfully!")
