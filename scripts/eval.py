import argparse
import os

import torch
import tqdm
from torch.utils.data import DataLoader

from src.data import get_dataloader
from src.utils import (SegmentationMetrics, get_best_checkpoint, get_device,
                       get_logger, get_model, get_run_dir, read_config)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int
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

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            metrics.update(preds, masks)

            loop.update(len(images))
    results = metrics.compute()
    metrics.log_metrics(results)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'])
    parser.add_argument('--run-id', required=True, type=str, help='Run identifier')
    args = parser.parse_args()

    cfg = read_config()
    device = get_device()
    model = get_model(cfg, args.model).to(device)
    run_dir = get_run_dir(args.run_id, args.model)
    logger = get_logger(run_dir, "eval.log")

    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    checkpoint_path = get_best_checkpoint(checkpoints_dir)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    hyperparams = cfg["hyperparams"][args.model]
    dataloader = get_dataloader(cfg=cfg, split="test", batch_size=cfg["batch_size"])

    evaluate_model(model, dataloader, device, hyperparams['num_classes'])
