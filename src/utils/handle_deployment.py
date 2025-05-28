import os
import shutil
import base64
import json

from fastapi import UploadFile
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2

from src.utils import read_config, get_device, get_model, convert_grayscale_to_colored_mask
from scripts.inference.inference_dataloader import get_inference_dataloader


def json_to_img(file :UploadFile = None):
    img_dict = json.load(file.file)
    print(img_dict)
    for i in range(len(img_dict['images'])):
        img_dict['images'][i] = base64.b64decode(img_dict['images'][i])
        # img_dir['images'][i] = img_bytes
        
    return img_dict

def img_to_json(img_dir: str):
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
                encoded_string = base64.b64encode(image.read()).decode('utf-8')
            list(img_dict.values())[i].append(encoded_string)
            
    json_object = json.dumps(img_dict, indent=4)
    print(json_object)
    return json_object

def handle_input_inference(file: UploadFile) -> tuple[str, str]:
    """
    Handles input given by the user and transforms it to a type which the model can use for prediction

    Args:
        file (UploadFile): file that has been uploaded by user

    Returns:
        tuple[str, str, str]: path to the input for the prediction, path to temporary directory for input, 
        path to temporary directory for output
    """
    # cwd = os.getcwd()
    
    input_folder_temp = 'temp_input/'
    output_folder_temp = 'temp_output/'
    os.makedirs(input_folder_temp, exist_ok=True)
    os.makedirs(output_folder_temp, exist_ok=True)

    img_dict = json_to_img(file = file)
    
    for i in range(len(img_dict['images'])):
        print(img_dict['image_names'][i], img_dict['images'][i])
        with open(os.path.join(input_folder_temp, img_dict['image_names'][i]), "wb") as image_file:
            image_file.write(img_dict['images'][i])
    # if file.filename.endswith('.jpg'):
    #     im = Image.open(file.file)
    #     im.save(os.path.join(cwd, input_folder_temp, file.filename),'JPEG')
    #     input_path = input_folder_temp        
    # elif file.filename.endswith('.zip'):
    #     zip_path_input = os.path.join(input_folder_temp, file.filename)
        
    #     with open(zip_path_input, 'wb') as buffer:
    #         shutil.copyfileobj(file.file, buffer)
        
    #     shutil.unpack_archive(zip_path_input, input_folder_temp, 'zip')
    #     input_path =  zip_path_input.replace('.zip', '')
        
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
    # zip_name_output = 'output'
    json_output = img_to_json(temp_output_dir)
    
    # shutil.make_archive(base_name=zip_name_output, format='zip', root_dir=cwd, base_dir=temp_output_dir)
    shutil.rmtree(os.path.join(cwd, temp_input_dir))
    shutil.rmtree(os.path.join(cwd, temp_output_dir))
    
    json_file_name = "output.json"
    
    with open(json_file_name, "w") as outfile:
        outfile.write(json_output)
    
    json_path = os.path.join(cwd, json_file_name) 
    return json_path


def load_model() -> torch.nn.Module:
    """
    Loads the model which will be used for inference.
    
    Returns:
        torch.nn.Module: initialized model
    """    
    cfg = read_config()
    device = get_device()
    model = get_model(cfg, 'baseline').to(device)
    
    model_path = cfg['inference']['model_path']
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def make_prediction(model: torch.nn.Module,  img_dir: str, output_dir: str) -> None:
    """
    Inference for model, makes an prediction based on the image given by the user.

    Args:
        model (torch.nn.Module): model that is being used for inference
        img_dir (str): path to directory for input images
        out_dir (str): path to directory for output of the model
    """
    cfg = read_config()
    
    dir_images = os.path.join(output_dir, 'images')
    dir_pred = os.path.join(output_dir, 'predictions')
    dir_pred_color = os.path.join(output_dir, 'predictions_color')
    
    os.makedirs(dir_images, exist_ok=True)
    os.makedirs(dir_pred, exist_ok=True)
    os.makedirs(dir_pred_color, exist_ok=True)
    
    device = get_device()
    dataloader = get_inference_dataloader(cfg, img_dir)
    model.eval()

    norms=cfg['transforms']['normalize']

    mean = torch.tensor(norms['mean']).view(3, 1, 1).to(device)
    std = torch.tensor(norms['std']).view(3, 1, 1).to(device)
    
    with torch.no_grad():
        for batch_idx, (images) in enumerate(dataloader):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for idx, (prediction, img) in enumerate(zip(preds, images)):
                pred_np = prediction.cpu().numpy().astype("uint8")
                img_idx = batch_idx * dataloader.batch_size + idx

                img = (img * std + mean).cpu().clamp(0, 1)
                Image.fromarray(pred_np).save(os.path.join(dir_pred, f"{img_idx:05}.png"))
                TF.to_pil_image(img).save(os.path.join(dir_images, f"{img_idx:05}.png"))
                color_img = convert_grayscale_to_colored_mask(os.path.join(dir_pred, f"{img_idx:05}.png"))
                cv2.imwrite(os.path.join(dir_pred_color, f"{img_idx:05}.png"), color_img)
    
        