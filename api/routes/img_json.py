from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import json
import base64
import tempfile
import zipfile
from PIL import Image
import shutil

router = APIRouter()

@router.post("/images_to_json/")
async def convert_images_to_json(file: UploadFile) -> FileResponse:
    """
    Converts a JSON file containing base64-encoded images, masks, and colormaps into image files.

    The resulting images are bundled in a ZIP file and returned.

    Expects JSON keys:
    - "images": list of base64-encoded input images.
    - "pred_mask": list of base64-encoded predicted mask images (grayscale).
    - "pred_color": list of base64-encoded predicted color mask images (colormap).

    Args:
        file (UploadFile): Uploaded JSON file with image data.

    Returns:
        FileResponse: Downloadable ZIP file containing all extracted images.
    """
    cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    
    if file.filename.endswith('.jpg'):
        im = Image.open(file.file)
        im.save(os.path.join(cwd, temp_dir, file.filename),'JPEG')
        input_path = temp_dir        
    elif file.filename.endswith('.zip'):
        zip_path_input = os.path.join(temp_dir, file.filename)

        with open(zip_path_input, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        shutil.unpack_archive(zip_path_input, temp_dir, 'zip')
        input_path =  zip_path_input.replace('.zip', '')
    
    dict_imgs = {
        'image_names': [],
        'images': []
    }
    
    images = os.listdir(input_path)
    for img in images:
        with open(os.path.join(input_path, img), "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        dict_imgs['image_names'].append(img)
        dict_imgs['images'].append(encoded_string)
    
    json_object = json.dumps(dict_imgs)
    filename_json = "img_to_json.json"
    json_path = os.path.join(temp_dir, filename_json)
    
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

    return FileResponse(json_path, filename=filename_json, media_type="application/json")