from typing import Literal

from api.utils import (handle_input_inference, handle_output_inference,
                       make_prediction)
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
from src.utils import load_model

router = APIRouter()

@router.post("/")
async def predict_segmentation(
    file: UploadFile,
) -> FileResponse:
    """
    Predicts a segmentation map from an uploaded image file using a selected model.

    Args:
        file (UploadFile): The uploaded JSON file containing base64-encoded images and metadata.
        run_id (str): The run identifier for the training session (used to locate the checkpoint).
        model_name (Literal["baseline", "unet"]): The model architecture to use for inference.

    Returns:
        FileResponse: A JSON file (`output.json`) containing the original images and
        the predicted segmentation masks encoded as base64.
    """
    temp_input_dir, temp_output_dir = handle_input_inference(file)
    
    run_id = "3"
    model_name = "baseline"
    print(run_id)
    model = load_model(run_id, model_name)

    make_prediction(model, temp_input_dir, temp_output_dir)

    json_path = handle_output_inference(temp_input_dir, temp_output_dir)

    return FileResponse(path=json_path, media_type='application/json', filename='output.json')
