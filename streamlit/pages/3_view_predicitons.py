import base64
import io
import os
import zipfile

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from utils import configure_layout, session_sidebar, video_html


configure_layout()
session_sidebar()
st.title("View Prediction Results")

if 'results' not in st.session_state:
    st.warning("No results available. Please run predictions first.")
else:
    selected_filename = st.selectbox("Select a file to view results", list(st.session_state['results'].keys()))

    result_zip = st.session_state['results'][selected_filename]
    col1, col2 = st.columns(2)

    with zipfile.ZipFile(result_zip) as z:

        namelist = z.namelist()
        video_files = [name for name in namelist if name.endswith(".mp4")]

        if video_files and len(video_files) == 2:
            predicted_video = next((v for v in video_files if "predicted" in v.lower()), None)
            original_video = next((v for v in video_files if v != predicted_video), None)

            video_bytes_original = z.read(original_video)
            video_bytes_predicted = z.read(predicted_video)

            b64_original = base64.b64encode(video_bytes_original).decode("utf-8")
            b64_predicted = base64.b64encode(video_bytes_predicted).decode("utf-8")

            v_html = video_html(b64_original, b64_predicted)

            components.html(v_html, height=600)
        else:
            original_images = {name: z.read(name) for name in z.namelist() if name.startswith("images/") and name.endswith((".png", ".jpg", ".jpeg"))}
            segmented_images = {name: z.read(name) for name in z.namelist() if name.startswith("pred_color/") and name.endswith((".png", ".jpg", ".jpeg"))}

            matched_files = []
            for name in original_images:
                basename = os.path.basename(name)
                if f"pred_color/{basename}" in segmented_images:
                    matched_files.append((original_images[name], segmented_images[f"pred_color/{basename}"], basename))

            for original_data, segmented_data, filename in matched_files:
                original_image = Image.open(io.BytesIO(original_data))
                segmented_image = Image.open(io.BytesIO(segmented_data))

                st.subheader(f"Results for {filename}")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="Original", use_container_width=True)
                with col2:
                    st.image(segmented_image, caption="Segmented", use_container_width=True)
                st.markdown("---")
