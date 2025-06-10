import requests
from io import BytesIO
import os


def upload_image_or_zip(file_bytes: bytes, filename: str) -> BytesIO:
    """Uploads an image or ZIP file to the prediction API and returns the result.

    Args:
        file_bytes (bytes): Byte content of the uploaded image or ZIP.
        filename (str): The name of the uploaded file.

    Returns:
        BytesIO: The returned ZIP file as a BytesIO stream.

    Raises:
        requests.exceptions.HTTPError: If the API call fails.
    """

    response = requests.post(
        f"{os.getenv('API_BASE_URL')}/predict-image/returns-zip/",
        files=[("files", (filename, file_bytes))],
    )
    response.raise_for_status()
    return BytesIO(response.content)

def upload_video(file_bytes: bytes, filename: str) -> BytesIO:
    """Uploads a video to the prediction API and returns the result ZIP.

    Args:
        file_bytes (bytes): Byte content of the video.
        filename (str): The name of the uploaded video file.

    Returns:
        BytesIO: The returned ZIP file as a BytesIO stream.

    Raises:
        requests.exceptions.HTTPError: If the API call fails.
    """
    response = requests.post(
        f"{os.getenv('API_BASE_URL')}/predict-video/returns-zip/",
        files={"file": (filename, file_bytes, "video/mp4")}
    )
    response.raise_for_status()
    return BytesIO(response.content)
