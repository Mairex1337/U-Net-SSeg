import argparse
import os
from typing import Dict, Union

import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex, MulticlassF1Score

from src.data import get_dataloader
from src.utils import (Timer, get_best_checkpoint, get_device, get_logger,
                       get_model, get_run_dir, read_config, log_metrics, SegmentationMetrics)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Evaluate a semantic segmentation model using multiple metrics.

    This function evaluates a trained model on a given dataset using several performance metrics:
    - Intersection over Union (IoU)
    - Dice Score (F1-score)
    - Precision
    - Recall
    - Pixel Accuracy (micro accuracy)
    - Mean Accuracy (macro accuracy)

    Metrics are computed using class-index inputs (i.e., shape [B, H, W]), which align with the
    format of most semantic segmentation model outputs. All metrics are accumulated across the
    entire dataset and returned in a single dictionary.

    Args:
        model (torch.nn.Module): The trained segmentation model to evaluate.
        dataloader (DataLoader): DataLoader providing evaluation data.
        device (torch.device): The device (CPU or GPU) for evaluation.
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        Dict[str, Union[float, torch.Tensor]]: A dictionary containing:
            - 'IoU_per_class' (Tensor): Per-class Intersection over Union.
            - 'Dice_per_class' (Tensor): Per-class Dice (F1) score.
            - 'Precision_per_class' (Tensor): Per-class precision.
            - 'Recall_per_class' (Tensor): Per-class recall.
            - 'Pixel_Accuracy' (float): Overall pixel accuracy.
            - 'Mean_Accuracy' (float): Mean class-wise accuracy.
            - 'mIoU' (float): Mean Intersection over Union.
            - 'mDice' (float): Mean Dice (F1) score.
    """
    model.eval()

    metrics = SegmentationMetrics(num_classes=num_classes, device=device)

    loop = tqdm.tqdm(
        total=len(dataloader.dataset),
        unit=" samples",
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
    )

    with Timer() as t:
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                metrics.update(preds, masks)

                loop.update(len(images))

    return metrics.compute()


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

    dataloader = get_dataloader(cfg=cfg, split="test")

    results = evaluate_model(model, dataloader, device, cfg['hyperparams'][args.model]['num_classes'])
    log_metrics(results, logger)
