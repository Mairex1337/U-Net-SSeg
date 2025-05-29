import os

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from api.data.dataloader import get_inference_dataloader
from src.utils import (convert_grayscale_to_colored_mask, get_device,
                       read_config)


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
    dir_pred = os.path.join(output_dir, 'pred_mask')
    dir_pred_color = os.path.join(output_dir, 'pred_color')

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
