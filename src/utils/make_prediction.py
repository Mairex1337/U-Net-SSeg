import os

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from scripts.inference.inference_dataloader import get_inference_dataloader
from src.utils import read_config, get_device


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