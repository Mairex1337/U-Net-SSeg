import os

from api.utils import (assemble_video_from_masks, cleanup_temp_dirs,
                       create_temp_dirs, create_zip_response, extract_frames,
                       load_default_model, make_prediction,
                       move_uploaded_video, save_uploaded_video)
from fastapi import APIRouter, BackgroundTasks, UploadFile
from fastapi.responses import StreamingResponse
from src.utils import resolve_path

router = APIRouter()

@router.post(
    "/returns-zip/",
    summary="Run segmentation on uploaded video in mp4, mov, or avi format",
    tags=["Prediction"],
    response_class=StreamingResponse
)
async def predict_video(
    file: UploadFile,
    background_tasks: BackgroundTasks
    ) -> StreamingResponse:
    """
    Processes an uploaded video by extracting frames, running segmentation, and returning results as a ZIP file.

    Args:
        file (UploadFile): The video file (.mp4, .mov, .avi) uploaded by the user.
        background_tasks (BackgroundTasks): FastAPI background task manager to handle post-response cleanup.

    Returns:
        StreamingResponse: A streaming response containing a ZIP file with the segmented video and predictions.
    """
    temp_input_dir, temp_output_dir = create_temp_dirs()

    temp_video_path = os.path.join(temp_input_dir, file.filename)
    save_uploaded_video(file, temp_video_path)
    frame_paths, fps, width, height = extract_frames(temp_video_path, temp_input_dir)
    move_uploaded_video(temp_video_path, temp_output_dir)

    model = load_default_model()
    make_prediction(model, temp_input_dir, temp_output_dir)

    assemble_video_from_masks(temp_output_dir, len(frame_paths), width, height, fps)

    background_tasks.add_task(cleanup_temp_dirs, resolve_path(""))

    return StreamingResponse(
        create_zip_response(temp_output_dir),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=output.zip"},
        background=background_tasks
    )
