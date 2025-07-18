from io import BytesIO

import streamlit as st
from utils import (configure_layout, session_sidebar, upload_image_or_zip,
                   upload_video)


configure_layout()
session_sidebar()
st.title("🚀 Run Predictions")

if 'uploaded_files' not in st.session_state:
    st.warning("Please upload files first on the 'Upload Files' page.")
else:
    uploaded_files = st.session_state['uploaded_files']
    filenames = [file['name'] for file in uploaded_files]

    selected_filename = st.selectbox("Select a file", filenames)

    if st.button("Run Predictions"):
        selected_file = next(file for file in uploaded_files if file['name'] == selected_filename)
        st.info(f"Running predictions on {selected_file['name']}...")
        file_content = BytesIO(selected_file['content'])

        if selected_file['name'].endswith(('.jpg', '.jpeg', '.png', '.zip')):
            result = upload_image_or_zip(file_content, selected_file['name'])
        elif selected_file['name'].endswith('.mp4'):
            result = upload_video(file_content, selected_file['name'])
        else:
            st.error("Unsupported file type.")
            result = None

        if 'results' not in st.session_state:
            st.session_state['results'] = {}

        if result:
            st.success("Predictions completed successfully! Dowload here or see the results on the next page.")
            st.session_state['results'][selected_filename] = result
            st.download_button(
                label="Download Results",
                data=result,
                file_name=f"predictions_{selected_file['name'].split('.')[0]}.zip",
                mime="application/zip"
            )
