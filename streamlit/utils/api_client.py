import requests
from io import BytesIO
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000") # works with local API but not Docker yet. Maybe work with env variables to make it work with both
# see if we can do a progress bar by counting entries in temp_input_dir and comparing them with entries in temp_output_dir/color_masks

def upload_image_or_zip(file_bytes: bytes, filename: str) -> BytesIO:

    response = requests.post(
        f"{API_BASE_URL}/predict-image/returns-zip/",
        files=[("files", (filename, file_bytes))],
    )
    response.raise_for_status()
    return BytesIO(response.content)

def upload_video(file_bytes: bytes, filename: str) -> BytesIO:

    response = requests.post(
        f"{API_BASE_URL}/predict-video/returns-zip/",
        files={"file": (filename, file_bytes, "video/mp4")}
    )
    response.raise_for_status()
    return BytesIO(response.content)
