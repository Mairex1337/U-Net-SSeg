from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse

from api.utils import (handle_input_inference, handle_output_inference,
                       make_prediction)
from src.utils import load_model

router = APIRouter()

@router.post(
    "/",
    summary="Run segmentation on uploaded image(s) in base64 JSON",
    tags=["Prediction"],
    response_class=FileResponse
)
async def predict_segmentation(
    file: UploadFile,
) -> FileResponse:
    """
    Accepts a JSON file containing base64-encoded images and returns a JSON file
    with predicted segmentation masks and colormaps (also base64-encoded).

    

    **Input Format:**
    A `.json` file with the following structure:
    ```json
    {
        "image_names": [
            "name.jpg",
            ...
        ],
        "images": [
            "/9j/4AAQSkZJRgABAQAAAQABAAD/...",
            ...
        ]
    }
    ```

    **Example Request:**
    ```bash
        curl -X POST \\
        http://127.0.0.1:8000/predict/ \\
        -H "accept: application/json" \\
        -H "Content-Type: multipart/form-data" \\
        -F "file=@api_images.json;type=application/json"
    ```

    **Response Format:**
    A `.json` file with keys:
    ```json
        {
            "images": ["<base64_original_image1>", "<base64_original_image2>, ..."],]
            "pred_mask": ["<base64_prediction_mask1>", "<base64_prediction_mask2>, ..."],
            "pred_color": ["<base64_colored_prediction1>", "<base64_colored_prediction2>, ..."]
        }
    ```
    - `"images"` contains the original (transformed) input images.
    
    - `"pred_mask"` contains the raw predicted mask as a grayscale PNG.

    - `"pred_color"` contains the same mask with a colormap applied.
    
    - All images are base64-encoded PNG or JPEG.

    Check README.md for instructions on json conversion
    """
    temp_input_dir, temp_output_dir = handle_input_inference(file)
    
    run_id = "1"
    model_name = "unet"
    model = load_model(run_id, model_name)

    make_prediction(model, temp_input_dir, temp_output_dir)

    json_path = handle_output_inference(temp_input_dir, temp_output_dir)

    return FileResponse(path=json_path, media_type='application/json', filename='output.json')
