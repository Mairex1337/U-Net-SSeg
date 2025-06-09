import streamlit as st
import zipfile
from PIL import Image
import io
import os

st.title("View Prediction Results")


if 'results' not in st.session_state:
    st.warning("No results available. Please run predictions first.")
else:
    selected_filename = st.selectbox("Select a file to view results", list(st.session_state['results'].keys()))

    result_zip = st.session_state['results'][selected_filename]
    col1, col2 = st.columns(2)


    with zipfile.ZipFile(result_zip) as z:
        # Extract images from each relevant subfolder

        namelist = z.namelist()
        video_files = [name for name in namelist if name.endswith(".mp4")]

        if video_files:
            matched_files = []
            for video_file in video_files:
                with z.open(video_file) as video_data:
                    video_bytes = video_data.read()
                    matched_files.append((video_bytes, os.path.basename(video_file)))

        else:

            original_images = {name: z.read(name) for name in z.namelist() if name.startswith("images/") and name.endswith((".png", ".jpg", ".jpeg"))}
            segmented_images = {name: z.read(name) for name in z.namelist() if name.startswith("pred_color/") and name.endswith((".png", ".jpg", ".jpeg"))}

            # Match by filename (e.g., images/img001.png <--> pred_color/img001.png)
            matched_files = []
            for name in original_images:
                basename = os.path.basename(name)
                if f"pred_color/{basename}" in segmented_images:
                    matched_files.append((original_images[name], segmented_images[f"pred_color/{basename}"], basename))

    for original_data, segmented_data, filename in matched_files:
        original_image = Image.open(io.BytesIO(original_data))
        segmented_image = Image.open(io.BytesIO(segmented_data))

        st.subheader(f"Results for {filename}")
        col1.image(original_image, caption=f"Original: {filename}", use_column_width=True)
        col2.image(segmented_image, caption=f"Segmented: {filename}", use_column_width=True)
        st.write("---")
