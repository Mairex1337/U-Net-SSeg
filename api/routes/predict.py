from api.utils import (cleanup_temp_dirs, handle_input_inference,
                       handle_output_inference, load_default_model,
                       make_prediction)
from fastapi import APIRouter, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse
from src.utils import resolve_path

router = APIRouter()

@router.post(
    "/returns-json/",
    summary="Run segmentation on uploaded image(s) in base64 JSON",
    tags=["Prediction"],
    response_class=FileResponse
)
async def predict_segmentation(
    file: UploadFile,
    background_tasks: BackgroundTasks
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

    model = load_default_model()

    make_prediction(model, temp_input_dir, temp_output_dir)

    json_path = handle_output_inference(temp_input_dir, temp_output_dir)

    background_tasks.add_task(cleanup_temp_dirs, resolve_path(""))

    return FileResponse(
        path=json_path,
        media_type='application/json',
        filename='output.json',
        background=background_tasks
        )
