import os
import json
import base64
import tempfile
import zipfile

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

@router.post("/json-to-images/")
async def convert_json_to_images(file: UploadFile) -> FileResponse:
    """
    Converts a JSON file containing base64-encoded images, masks, and colormaps into image files.

    The resulting images are bundled in a ZIP file and returned.

    Expects JSON keys:
    - "images": list of base64-encoded input images.
    - "pred_mask": list of base64-encoded predicted mask images (grayscale).
    - "pred_color": list of base64-encoded predicted color mask images (colormap).

    Args:
        file (UploadFile): Uploaded JSON file with image data.

    Returns:
        FileResponse: Downloadable ZIP file containing all extracted images.
    """
    try:
        data = json.load(file.file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")

    required_keys = ["images", "pred_mask", "pred_color"]
    if not all(key in data for key in required_keys):
        raise HTTPException(status_code=400, detail=f"JSON must contain: {', '.join(required_keys)}")

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "images.zip")
    names = data.get("image_names", [f"image_{i}.jpg" for i in range(len(data["images"]))])


    try:
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i, name in enumerate(names):
                for kind, prefix in zip(["images", "pred_mask", "pred_color"], ["input", "mask", "color"]):
                    content = base64.b64decode(data[kind][i])
                    filename = f"{prefix}_{name}"
                    file_path = os.path.join(temp_dir, filename)

                    with open(file_path, "wb") as img_file:
                        img_file.write(content)

                    zipf.write(file_path, arcname=filename)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"""Error processing images: {str(e)}, 
            please upload a JSON file with encoded images for all values"""
            )

    return FileResponse(zip_path, filename="converted_images.zip", media_type="application/zip")
