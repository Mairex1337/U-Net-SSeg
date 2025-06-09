from api.utils import (cleanup_temp_dirs, create_temp_dirs,
                       create_zip_response, extract_files, load_default_model)
from api.utils.inference import make_prediction
from fastapi import APIRouter, BackgroundTasks, UploadFile
from fastapi.responses import StreamingResponse
from src.utils import resolve_path

router = APIRouter()
# TODO: add two enpoints that predict on images and videos. return the predicitons
# then build streamlit app that can display both videos in parallel and lets the user upload predict and maybe save somewhere
# progress bar?
# more funcitonality???


@router.post(
    "/returns-zip/",
    summary="Run segmentation on uploaded image(s) in jpg or zip format",
    tags=["Prediction"],
    response_class=StreamingResponse
)
async def predict_image(files: list[UploadFile], background_tasks: BackgroundTasks) -> StreamingResponse:

    temp_input_dir, temp_output_dir = create_temp_dirs()

    await extract_files(files, temp_input_dir)

    model = load_default_model()

    make_prediction(model, temp_input_dir, temp_output_dir)

    background_tasks.add_task(cleanup_temp_dirs, resolve_path(""))

    return StreamingResponse(
        create_zip_response(temp_output_dir),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=output.zip"},
        background=background_tasks
    )
