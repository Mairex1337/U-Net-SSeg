import streamlit as st
# from api_client import upload_image_or_zip, upload_video
# from file_handlers import save_and_extract_zip
import os

st.title("📤 Upload Data for Predictions")

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

option = st.radio("Choose input type:", ["Image or ZIP", "Video"])
if option == "Video":
    uploaded_file = st.file_uploader("Upload video file", type=["mp4"])
elif option == "Image or ZIP":
    uploaded_file = st.file_uploader("Upload file", type=["jpg", "zip"])

if uploaded_file and st.button("Send to backend"):
    st.session_state['uploaded_files'].append(uploaded_file)

    filename = uploaded_file.name
    st.success(f"File '{filename}' uploaded successfully!")
