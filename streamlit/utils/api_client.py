import requests
from io import BytesIO

API_BASE_URL = "http://localhost:8000"  #TODO: make available for docker as well or your deployed URL

def upload_image_or_zip(file: BytesIO, filename: str) -> BytesIO:
    """
    Send image or ZIP to the FastAPI image endpoint and get back ZIP result.
    """
    response = requests.post(
        f"{API_BASE_URL}/predict/image/",
        files={"file": (filename, file, "application/octet-stream")},
    )
    response.raise_for_status()
    return BytesIO(response.content)

def upload_video(file: BytesIO, filename: str) -> BytesIO:
    """
    Send video to the FastAPI video endpoint and get back ZIP result.
    """
    response = requests.post(
        f"{API_BASE_URL}/predict/video/",
        files={"file": (filename, file, "video/mp4")},
    )
    response.raise_for_status()
    return BytesIO(response.content)
