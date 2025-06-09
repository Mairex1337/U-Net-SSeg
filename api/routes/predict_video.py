import os
import shutil
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from io import BytesIO

from src.utils import load_model
from api.utils.inference import make_prediction

router = APIRouter()

@router.post(
    "/predict-video/",
    summary="Run segmentation on uploaded video",
    tags=["Prediction"]
)
async def predict_video(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only video files with .mp4, .mov or .avi extensions are supported."
        )

    # Use static folders for debugging
    temp_input_dir = 'temp_input/'
    temp_output_dir = 'temp_output/'
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    temp_video_path = os.path.join(temp_input_dir, file.filename)

    # Save uploaded video
    try:
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}"
        )

    # Extract frames
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_input_dir, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()

    if count == 0:
        raise HTTPException(status_code=400, detail="No frames could be extracted from the video.")

    # Delete the uploaded video file after extracting frames
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    run_id = "1"
    model_name = "unet"
    model = load_model(run_id, model_name)
    make_prediction(model, temp_input_dir, temp_output_dir)

    output_video_path = os.path.join(temp_output_dir, "predicted_video.mp4")
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    for i in range(count):
        mask_path = os.path.join(temp_output_dir, "pred_color", f"{i:05d}.png")
        if not os.path.exists(mask_path):
            raise HTTPException(status_code=500, detail=f"Missing mask for frame {i}")
        mask_img = cv2.imread(mask_path)
        resized_mask = cv2.resize(mask_img, (width, height))
        out.write(resized_mask)
    out.release()

    # Read video into memory
    with open(output_video_path, "rb") as f:
        video_bytes = f.read()

    return StreamingResponse(
        BytesIO(video_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=predicted_video.mp4"}
    )
    # delete temp directories
