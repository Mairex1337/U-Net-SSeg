import base64
import json
import os
import shutil
import zipfile
from io import BytesIO
from api.utils.file_management import create_temp_dirs
from fastapi import HTTPException, UploadFile


def json_to_img(file :UploadFile = None) -> dict[str, str] | str:
    """
    Parses a JSON file containing base64-encoded images and decodes them into binary data.

    The JSON file must contain equally long lists under the keys 'images', 'pred_mask',
    'pred_color', and 'image_names'.

    Args:
        file (UploadFile): The uploaded JSON file containing image data.

    Returns:
        dict[str, str]: A dictionary where the 'images' key contains binary image data.

    Raises:
        HTTPException: If the file is not valid JSON or if the structure is incorrect.
    """
    try:
        img_dict = json.load(file.file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
        status_code=400,
        detail="Uploaded file is not of type JSON"
    )

    lengths = [len(v) for v in img_dict.values()]
    if not all(length == lengths[0] for length in lengths):
        raise HTTPException(
            status_code=400,
            detail="Values of all JSON categories must be of the same length"
            )

    try:
        for i in range(len(img_dict['images'])):
            img_dict['images'][i] = base64.b64decode(img_dict['images'][i])
    except (ValueError, KeyError):
        raise HTTPException(
        status_code=400,
        detail="JSON file does not have the correct structure"
    )

    return img_dict


def img_to_json(img_dir: str):
    """
    Converts image files in a directory structure to a JSON string with base64-encoded contents.

    Assumes that `img_dir` contains three subdirectories (e.g. images, pred_mask, pred_color),
    and encodes the content of each image file found inside these subdirectories.

    Args:
        img_dir (str): Path to the directory containing image subdirectories.

    Returns:
        str: A JSON string containing base64-encoded image data organized by category.
    """
    expected_keys = ['images', 'pred_mask', 'pred_color']
    img_dict = {key: [] for key in expected_keys}

    for key in expected_keys:
        folder = os.path.join(img_dir, key)
        if not os.path.isdir(folder):
            raise ValueError(f"Expected folder '{folder}' not found.")
        filenames = sorted(os.listdir(folder))
        for filename in filenames:
            path = os.path.join(folder, filename)
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
                img_dict[key].append(encoded)

    return json.dumps(img_dict, indent=4)


def handle_input_inference(file: UploadFile) -> tuple[str, str]:
    """
    Handles input given by the user and transforms it to a type which the model can use for prediction

    Args:
        file (UploadFile): file that has been uploaded by user

    Returns:
        tuple[str, str]:  path to temporary directory for input, path to temporary directory for output
    """

    input_folder_temp, output_folder_temp = create_temp_dirs()

    img_dict = json_to_img(file = file)

    try:
        for i in range(len(img_dict['images'])):
            with open(os.path.join(input_folder_temp, img_dict['image_names'][i]), "wb") as image_file:
                image_file.write(img_dict['images'][i])
    except (ValueError, KeyError):
        raise HTTPException(
        status_code=400,
        detail="JSON file does not have the correct structure"
    )

    return input_folder_temp, output_folder_temp


def handle_output_inference(temp_input_dir: str, temp_output_dir: str) -> str:
    """
    Handles output, by cleaning up root directory and zipping output, so it can be returned in API call

    Args:
        temp_input_dir (str): path to temporary directory for input
        temp_output_dir (str): path to temporary directory for output

    Returns:
        str: returns path to zip file which can be returns in API call
    """
    cwd = os.getcwd()
    json_output = img_to_json(temp_output_dir)

    shutil.rmtree(os.path.join(cwd, temp_input_dir))
    shutil.rmtree(os.path.join(cwd, temp_output_dir))

    json_file_name = "output.json"

    with open(json_file_name, "w") as outfile:
        outfile.write(json_output) #TODO Check if we want this

    json_path = os.path.join(cwd, json_file_name)
    return json_path


def create_zip_response(temp_output_dir):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=temp_output_dir)
                zipf.write(full_path, arcname)

    zip_buffer.seek(0)
    return zip_buffer
