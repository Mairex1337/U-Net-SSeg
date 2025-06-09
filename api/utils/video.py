import os
import shutil

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, status

router = APIRouter()

def save_uploaded_video(file: UploadFile, path: str) -> None:
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only video files with .mp4, .mov or .avi extensions are supported."
        )
    try:
        with open(path, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}"
        )


def move_uploaded_video(temp_video_path: str, temp_output_dir: str) -> None:
    if os.path.exists(temp_video_path):
        shutil.move(temp_video_path, os.path.join(temp_output_dir, os.path.basename(temp_video_path)))


def extract_frames(video_path: str, input_dir: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(input_dir, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()

    if count == 0:
        raise HTTPException(status_code=400, detail="No frames could be extracted from the video.")

    return frame_paths, fps, width, height

def assemble_video_from_masks(output_dir: str, frame_count: int, width: int, height: int, fps: float) -> bytes:
    output_video_path = os.path.join(output_dir, "predicted_video.mp4")
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    for i in range(frame_count):
        mask_path = os.path.join(output_dir, "pred_color", f"{i:05d}.png")
        if not os.path.exists(mask_path):
            raise HTTPException(status_code=500, detail=f"Missing mask for frame {i}")
        mask_img = cv2.imread(mask_path)
        resized_mask = cv2.resize(mask_img, (width, height))
        out.write(resized_mask)
    out.release()
