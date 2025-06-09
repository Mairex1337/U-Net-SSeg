import requests
from io import BytesIO
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

def upload_image_or_zip(file_bytes: bytes, filename: str) -> BytesIO:
    """
    Send a single image or ZIP file to the FastAPI endpoint expecting 'files' (list of UploadFile).
    """
    response = requests.post(
        f"{API_BASE_URL}/predict-image/returns-zip/",
        files=[("files", (filename, file_bytes))],
    )
    response.raise_for_status()
    return BytesIO(response.content)

def upload_video(file_bytes: bytes, filename: str) -> BytesIO:
    """
    Send an MP4 video file to the FastAPI endpoint expecting a single UploadFile named 'file'.
    """
    response = requests.post(
        f"{API_BASE_URL}/predict-video/returns-zip/",
        files={"file": (filename, file_bytes, "video/mp4")}
    )
    response.raise_for_status()
    return BytesIO(response.content)
