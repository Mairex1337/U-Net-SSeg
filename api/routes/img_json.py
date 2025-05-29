import os
import json
import base64
import tempfile
import shutil

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

router = APIRouter()

@router.post("/images_to_json/")
async def convert_images_to_json(file: UploadFile) -> FileResponse:
    """
    Converts a single image or multiple images into a JSON file which can be used for prediction.

    Args:
        file (UploadFile): Either a single JPG file or a zip file containing multiple JPG files.

    Returns:
        FileResponse: Downloadable JSON file, which can be used for prediction.
    """
    temp_dir = tempfile.mkdtemp()
    
    if not file.filename.endswith(('.jpg', '.zip')):
        raise HTTPException(
            status_code=400, 
            detail="Uploaded file is not of type .jpg or .zip"
        )
        
    if file.filename.endswith('.jpg'):
        im = Image.open(file.file)
        im.save(os.path.join(temp_dir, file.filename),'JPEG')
        img_folder = temp_dir        
    elif file.filename.endswith('.zip'):
        zip_path_input = os.path.join(temp_dir, file.filename)

        with open(zip_path_input, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        shutil.unpack_archive(zip_path_input, temp_dir, 'zip')
        img_folder =  zip_path_input.replace('.zip', '')    
    
    dict_imgs = {
        'image_names': [],
        'images': []
    }
    
    images = os.listdir(img_folder)
    for img in images:
        with open(os.path.join(img_folder, img), "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()
        dict_imgs['image_names'].append(img)
        dict_imgs['images'].append(encoded_string)
        if not file.filename.endswith(('.jpg')):
            raise HTTPException(
                status_code=400, 
                detail="files in .zip file are not of type jpg"
            )
    
    json_object = json.dumps(dict_imgs)
    filename_json = "img_to_json.json"
    json_path = os.path.join(temp_dir, filename_json)
    
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

    return FileResponse(json_path, filename=filename_json, media_type="application/json")