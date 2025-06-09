import os
import shutil
import zipfile

from fastapi import HTTPException, status


def create_temp_dirs() -> tuple[str, str]:
    """
    Creates temporary input and output directories if they do not already exist.

    Returns:
        tuple[str, str]: Paths to the created input and output directories.
    """
    temp_input_dir = 'temp_input/'
    temp_output_dir = 'temp_output/'
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    return temp_input_dir, temp_output_dir


def cleanup_temp_dirs(base_dir: str) -> None:
    """
    Deletes known temporary folders and output file from previous runs.

    Args:
        base_dir (str): The current working directory to clean up in.
    """
    paths = [
        os.path.join(base_dir, "output.json"),
        os.path.join(base_dir, "temp_input"),
        os.path.join(base_dir, "temp_output"),
    ]
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

def move_jpgs_to_root(root_dir: str) -> None:
    """
    Moves all .jpg and .jpeg files from subdirectories of `root_dir` to the root level.

    Args:
        root_dir (str): Path to the root directory in which to consolidate JPEG files.
    """

    for current_dir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                full_path = os.path.join(current_dir, f)
                new_path = os.path.join(root_dir, f)
                if full_path != new_path:
                    os.rename(full_path, new_path)


async def extract_files(files: list, temp_input_dir: str):
    """
    Saves and extracts uploaded image or ZIP files into a temporary input directory.

    Supports .jpg, .jpeg, and .zip files. ZIP files are extracted and their JPEG contents
    are moved to the root of the input directory.

    Args:
        files (list): List of uploaded image or ZIP files.
        temp_input_dir (str): Path to the temporary input directory.

    Raises:
        HTTPException: If unsupported file types are uploaded, or extraction fails,
                       or no valid JPEG images are found.
    """
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
