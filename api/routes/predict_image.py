import os
import tempfile
import zipfile
from io import BytesIO

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

router = APIRouter()
# TODO: add two enpoints that predict on images and videos. return the predicitons
# then build streamlit app that can display both videos in parallel and lets the user upload predict and maybe save somewhere
# more funcitonality???

@router.post(
    "/predict-image/",
    summary="Run segmentation on uploaded image(s) in jpg or zip format",
    tags=["Prediction"],
    #response_class=FileResponse
)
async def predict_image(files: list[UploadFile] = File(...)):
    # Create temp dirs
    input_folder_temp = 'temp_input/'
    output_folder_temp = 'temp_output/'
    os.makedirs(input_folder_temp, exist_ok=True)
    os.makedirs(output_folder_temp, exist_ok=True)

    for file in files:
        filename = file.filename.lower()

        if filename.endswith(".zip"):
            # Save and extract ZIP
            zip_path = os.path.join(input_folder_temp, filename)
            with open(zip_path, "wb") as f:
                f.write(await file.read())

            import zipfile
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(input_folder_temp)
            except zipfile.BadZipFile:
                return {"error": f"{filename} is not a valid ZIP file."}

        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Save individual image
            img_path = os.path.join(input_folder_temp, os.path.basename(filename))
            with open(img_path, "wb") as f:
                f.write(await file.read())

        else:
            return {"error": f"Unsupported file type: {filename}. Only JPG and ZIP allowed."}

    # Collect all JPGs from input_folder_temp (including extracted ZIPs)
    image_paths = [
        os.path.join(input_folder_temp, f)
        for f in os.listdir(input_folder_temp)
        if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
    ]

    if not image_paths:
        return {"error": "No valid JPG images found."}

    # --- Later: Do inference and return results and clean up temp dirs ---

    return {
        "message": f"{len(image_paths)} images processed successfully.",
        "input_dir": input_folder_temp,
        "output_dir": output_folder_temp
    }
