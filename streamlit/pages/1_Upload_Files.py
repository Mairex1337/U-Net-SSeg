import streamlit as st
from utils import configure_layout, session_sidebar


configure_layout()
session_sidebar()
st.title("📤 Upload Data for Predictions")

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

uploaded_file = st.file_uploader("Upload an image, ZIP file, or video", type=["jpg", "zip", "mp4"])

if uploaded_file and st.button("Save File"):
    content = uploaded_file.read()
    st.session_state['uploaded_files'].append({
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "content": content,
    })

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

if st.session_state['uploaded_files']:
    st.subheader("Uploaded Files")
    for file in st.session_state['uploaded_files']:
        st.write(f"- {file['name']} ({file['type']})")
