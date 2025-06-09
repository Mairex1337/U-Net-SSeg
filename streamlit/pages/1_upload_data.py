import streamlit as st
# from api_client import upload_image_or_zip, upload_video
# from file_handlers import save_and_extract_zip
import os

st.title("📤 Upload Data for Predictions")

option = st.radio("Choose input type:", ["Image or ZIP", "Video"])
uploaded_file = st.file_uploader("Upload file", type=["jpg", "zip", "mp4"])

if uploaded_file and st.button("Send to backend"):
    st.info("Processing...")

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
