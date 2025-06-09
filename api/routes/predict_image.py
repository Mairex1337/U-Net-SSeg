import os
import tempfile
import zipfile
from io import BytesIO
import shutil

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from src.utils import resolve_path
from api.utils.inference import make_prediction
from src.utils import load_model
from fastapi.responses import FileResponse

router = APIRouter()
# TODO: add two enpoints that predict on images and videos. return the predicitons
# then build streamlit app that can display both videos in parallel and lets the user upload predict and maybe save somewhere
# more funcitonality???
# refactor, no logic in endpoint

def move_jpgs_to_root(root_dir):
    for current_dir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                full_path = os.path.join(current_dir, f)
                new_path = os.path.join(root_dir, f)
                if full_path != new_path:
                    os.rename(full_path, new_path)


@router.post(
    "/predict-image/",
    summary="Run segmentation on uploaded image(s) in jpg or zip format",
    tags=["Prediction"],
    response_class=FileResponse
)
async def predict_image(files: list[UploadFile]) -> FileResponse:
    temp_input_dir = 'temp_input/'
    temp_output_dir = 'temp_output/'
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".zip"}

    for file in files:
        filename = file.filename
        name, ext = os.path.splitext(filename)
        name = name.lower()
        ext = ext.lower()

        if ext not in supported_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{ext}' in '{filename}'. Only .jpg and .zip are allowed."
            )

        file_path = os.path.join(temp_input_dir, filename)

        try:
            with open(file_path, "wb") as f:
                f.write(await file.read())
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file '{filename}': {str(e)}"
            )

        if ext == ".zip":
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_input_dir)
                    move_jpgs_to_root(temp_input_dir)
                    os.remove(file_path)
                    shutil.rmtree(os.path.join(temp_input_dir, name))

            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid ZIP archive: '{filename}'"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error extracting ZIP '{filename}': {str(e)}"
                )

    if len(os.listdir(temp_input_dir)) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid JPG images found after processing uploads."
        )

    run_id = "1"
    model_name = "unet"
    model = load_model(run_id, model_name)

    make_prediction(model, temp_input_dir, temp_output_dir)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=temp_output_dir)
                zipf.write(full_path, arcname)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=output.zip"}
    )
