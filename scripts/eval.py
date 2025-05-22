import argparse
import os
from typing import Dict, Union

import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex, MulticlassF1Score

from src.data import get_dataloader
from src.utils import (Timer, get_best_checkpoint, get_device, get_logger,
                       get_model, get_run_dir, read_config, resolve_path)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Evaluate a semantic segmentation model using per-class and mean IoU and Dice scores.

    This function uses `torchmetrics.JaccardIndex` for computing per-class Intersection over Union (IoU),
    and `torchmetrics.MulticlassF1Score` to compute the per-class Dice score.

    Note:
        Dice score is mathematically equivalent to the F1-score in the multiclass setting:
        Dice = 2 * TP / (2 * TP + FP + FN) == F1.
        `MulticlassF1Score` accepts class-index inputs (e.g., shape [B, H, W]), which aligns with the
        output format of typical semantic segmentation tasks. On the other hand, `torchmetrics.DiceScore`
        expects one-hot encoded masks or multi-label inputs, making it incompatible with class index masks.

    Args:
        model (torch.nn.Module): The trained segmentation model to evaluate.
        dataloader (DataLoader): DataLoader providing evaluation data.
        device (torch.device): The device (CPU or GPU) for evaluation.
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        Dict[str, Union[float, torch.Tensor]]: A dictionary with:
            - 'IoU_per_class': Per-class IoU (Intersection over Union).
            - 'Dice_per_class': Per-class Dice scores (F1-score equivalent).
            - 'mIoU': Mean IoU across all classes.
            - 'mDice': Mean Dice score across all classes.
    """
    model.eval()

    iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, average=None).to(device)
    dice_metric = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

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

                iou_metric.update(preds, masks)
                dice_metric.update(preds, masks)

                loop.update(len(images))

    iou_per_class = iou_metric.compute().cpu()
    dice_per_class = dice_metric.compute().cpu()

    mean_iou = iou_per_class.mean()
    mean_dice = dice_per_class.mean()

    return {
        "IoU_per_class": iou_per_class,
        "Dice_per_class": dice_per_class,
        "mIoU": mean_iou.item(),
        "mDice": mean_dice.item()
    }


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

    dataloader = get_dataloader(cfg=cfg, train=False)

    results = evaluate_model(model, dataloader, device, cfg['hyperparams'][args.model]['num_classes'])

    logger.info("Evaluation Results:")
    logger.info(f"Mean IoU: {results['mIoU']:.4f}")
    logger.info(f"Mean Dice: {results['mDice']:.4f}")
    logger.info("\nPer-Class Evaluation Metrics:")
    logger.info(f"{'Class':<18}{'IoU':>10}{'Dice':>10}")
    logger.info("-" * 38)
    for idx, (iou, dice) in enumerate(zip(results['IoU_per_class'], results['Dice_per_class'])):
        logger.info(f"{idx:<4}{cfg['class_distribution']['id_to_class'][idx]:<14}{iou.item():>10.4f}{dice.item():>10.4f}")
