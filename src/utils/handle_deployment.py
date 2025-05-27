import os
import shutil

from fastapi import UploadFile
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.utils import read_config, get_device, get_model
from scripts.inference.inference_dataloader import get_inference_dataloader

def handle_input_inference(file: UploadFile) -> tuple[str, str, str]:
    """
    Handles input given by the user and transforms it to a type which the model can use for prediction

    Args:
        file (UploadFile): file that has been uploaded by user

    Returns:
        tuple[str, str, str]: path to the input for the prediction, path to temporary directory for input, 
        path to temporary directory for output
    """
    cwd = os.getcwd()
    
    input_folder_temp = 'temp_input/'
    output_folder_temp = 'temp_output/'
    os.mkdir(input_folder_temp)
    os.mkdir(output_folder_temp)
    
    if file.filename.endswith('.jpg'):
        im = Image.open(file.file)
        im.save(os.path.join(cwd, input_folder_temp, file.filename),'JPEG')
        input_path = input_folder_temp        
    elif file.filename.endswith('.zip'):
        zip_path_input = os.path.join(input_folder_temp, file.filename)
        
        with open(zip_path_input, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        shutil.unpack_archive(zip_path_input, input_folder_temp, 'zip')
        input_path =  zip_path_input.replace('.zip', '')
        
    return input_path, input_folder_temp, output_folder_temp


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
    zip_name_output = 'output'
    
    shutil.make_archive(base_name=zip_name_output, format='zip', root_dir=cwd, base_dir=temp_output_dir)
    shutil.rmtree(os.path.join(cwd, temp_input_dir))
    shutil.rmtree(os.path.join(cwd, temp_output_dir))
    
    zip_path = os.path.join(cwd, zip_name_output + '.zip') 
    return zip_path


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
    os.mkdir(os.path.join(output_dir, 'predictions'))
    os.mkdir(os.path.join(output_dir, 'images'))
    
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
                Image.fromarray(pred_np).save(os.path.join(output_dir, "predictions", f"{img_idx:05}.png"))
                TF.to_pil_image(img).save(os.path.join(output_dir, "images", f"{img_idx:05}.png"))