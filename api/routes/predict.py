from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
from api.utils import (handle_input_inference,
                                         handle_output_inference, load_model,
                                         make_prediction)

router = APIRouter()

@router.post("/")
async def predict_segmentation(file: UploadFile) -> FileResponse:
    """
    This function adds a post to the api where the user can use the model to predict a segmentation map of an image

    Args:
        file (UploadFile): file that has been uploaded by user

    Returns:
        FileResponse: File that will be returned to the user,
        which will be a zip file, which contains the original images and the predictions made by the model.
    """
    temp_input_dir, temp_output_dir = handle_input_inference(file)

    model = load_model()

    make_prediction(model, temp_input_dir, temp_output_dir)

    json_path = handle_output_inference(temp_input_dir, temp_output_dir)

    return FileResponse(path=json_path, media_type='application/json', filename='output.json')
