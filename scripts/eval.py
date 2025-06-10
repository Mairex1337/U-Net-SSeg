import argparse
import logging
import os

import torch
import torchvision.transforms.functional as TF
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from src.data import get_dataloader
from src.utils import (SegmentationMetrics, get_device,
                       get_logger, get_run_dir, read_config, load_model)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    run_dir: str,
    logger: logging.Logger,
    norms: dict[list[float]]
) -> None:
    """
    Evaluate a semantic segmentation model using multiple metrics.

    This function evaluates a trained model on a given dataset using several performance metrics:
    - Intersection over Union (IoU)
    - Dice Score (F1-score)
    - Precision
    - Recall
    - Pixel Accuracy (micro accuracy)
    - Mean Accuracy (macro accuracy)

    Args:
        model (torch.nn.Module): The trained segmentation model to evaluate.
        dataloader (DataLoader): DataLoader providing evaluation data.
        device (str): The device used.
        num_classes (int): Number of classes in the segmentation task.
        run_dir (str): Directory of the training run of the model to be evaluated.
        logger (logging.Logger): Logger.
        norms (dict[list[float]]): Dict containing rgb mean, and std values.

    Returns:
        None
    """
    model.eval()

    metrics = SegmentationMetrics(num_classes=num_classes, device=device)
    loop = tqdm.tqdm(
        total=len(dataloader.dataset),
        unit=" samples",
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
    )
    os.makedirs(os.path.join(run_dir, "outputs", "predictions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "outputs", "images"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "outputs", "masks"), exist_ok=True)

    mean = torch.tensor(norms['mean']).view(3, 1, 1).to(device)
    std = torch.tensor(norms['std']).view(3, 1, 1).to(device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

            for idx, (pred, img, mask) in enumerate(zip(preds, images, masks)):
                pred_np = pred.cpu().numpy().astype("uint8")
                mask_np = mask.cpu().numpy().astype("uint8")
                img_idx = batch_idx * dataloader.batch_size + idx

                img = (img * std + mean).cpu().clamp(0, 1)
                Image.fromarray(pred_np).save(os.path.join(run_dir, "outputs", "predictions", f"{img_idx:05}.png"))
                Image.fromarray(mask_np).save(os.path.join(run_dir, "outputs", "masks", f"{img_idx:05}.png"))
                TF.to_pil_image(img).save(os.path.join(run_dir, "outputs", "images", f"{img_idx:05}.png"))

            loop.update(len(images))


    results = metrics.compute()
    metrics.log_metrics(results, logger)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'])
    parser.add_argument('--run-id', required=True, type=str, help='Run identifier')
    args = parser.parse_args()

    cfg = read_config()
    run_dir = get_run_dir(args.run_id, args.model)
    device = get_device()
    logger = get_logger(run_dir, "eval.log")

    model = load_model(args.run_id, args.model)
    hyperparams = cfg["hyperparams"][args.model]
    dataloader = get_dataloader(cfg=cfg, split="test", batch_size=hyperparams["batch_size"] // 2)

    evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            num_classes=hyperparams['num_classes'],
            run_dir=run_dir,
            logger=logger,
            norms=cfg['transforms']['normalize']
    )
