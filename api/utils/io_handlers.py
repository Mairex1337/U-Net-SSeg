import base64
import json
import os
import shutil

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
    assert all(length == lengths[0] for length in lengths), "JSON categories should have same length"

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
    img_dict = {
        'images': [],
        'pred_mask': [],
        'pred_color': []
    }

    dir_list = os.listdir(img_dir)
    for i, sub_dir in enumerate(dir_list):
        img_list = os.listdir(os.path.join(img_dir, sub_dir))
        for j in img_list:
            with open(os.path.join(img_dir, sub_dir, j), "rb") as image:
                encoded_string = base64.b64encode(image.read()).decode()
            list(img_dict.values())[i].append(encoded_string)

    json_object = json.dumps(img_dict, indent=4)
    return json_object


def handle_input_inference(file: UploadFile) -> tuple[str, str]:
    """
    Handles input given by the user and transforms it to a type which the model can use for prediction

    Args:
        file (UploadFile): file that has been uploaded by user

    Returns:
        tuple[str, str]:  path to temporary directory for input, path to temporary directory for output
    """

    input_folder_temp = 'temp_input/'
    output_folder_temp = 'temp_output/'
    os.makedirs(input_folder_temp, exist_ok=True)
    os.makedirs(output_folder_temp, exist_ok=True)

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
