import logging

import torch
import torchmetrics.classification as metric

from src.utils.path import read_config


class SegmentationMetrics:
    """
    A wrapper class to compute and manage multiple evaluation metrics for semantic segmentation.

    This class supports per-class and aggregated metrics including:
    - IoU (Intersection over Union)
    - Dice (F1-score)
    - Precision
    - Recall
    - Pixel Accuracy (micro)
    - Mean Accuracy (macro)
    """
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device

        self.metrics = {
            "iou" : metric.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255, average=None).to(device),
            "dice" : metric.MulticlassF1Score(num_classes=num_classes, ignore_index=255, average=None).to(device),
            "precision" : metric.MulticlassPrecision(num_classes=num_classes, ignore_index=255, average=None).to(device),
            "recall" : metric.MulticlassRecall(num_classes=num_classes, ignore_index=255, average=None).to(device),
            "pixel_accuracy" : metric.MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average='micro').to(device),
            "mean_accuracy" : metric.MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average='macro').to(device)
        }

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update all metrics with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predicted class indices of shape (B, H, W)
            targets (torch.Tensor): Ground truth class indices of shape (B, H, W)
        """
        preds = preds.flatten()
        targets = targets.flatten()

        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self) -> dict:
        """
        Compute all aggregated and per-class metrics.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        return {
            "IoU_per_class": self.metrics["iou"].compute(),
            "Dice_per_class": self.metrics["dice"].compute(),
            "Precision_per_class": self.metrics["precision"].compute(),
            "Recall_per_class": self.metrics["recall"].compute(),
            "Pixel_Accuracy": self.metrics["pixel_accuracy"].compute().item(),
            "Mean_Accuracy": self.metrics["mean_accuracy"].compute().item(),
            "mIoU": self.metrics["iou"].compute().mean().item(),
            "mDice": self.metrics["dice"].compute().mean().item()
        }

    def reset(self):
        """
        Reset all internal metric states for reuse on a new dataset or evaluation run.
        """
        for metric in self.metrics:
            metric.reset()

 
    def log_metrics(self, results: dict, logger: logging.Logger) -> None:
        """
        Logs evaluation metrics for semantic segmentation.

        This function prints overall metrics (e.g., pixel accuracy, mean accuracy, mean IoU, mean Dice)
        and detailed per-class metrics including IoU, Dice (F1), precision, and recall.

        Args:
            results (Dict): A dictionary containing evaluation results.

            logger (logging.Logger):
                The logger instance used to write the output to a file or console.

        Returns:
            None
        """
        cfg = read_config()

        logger.info("Evaluation Results:")
        logger.info(f"{'Pixel Accuracy:':<25}{results['Pixel_Accuracy']:.4f}")
        logger.info(f"{'Mean Accuracy:':<25}{results['Mean_Accuracy']:.4f}")
        logger.info(f"{'Mean IoU:':<25}{results['mIoU']:.4f}")
        logger.info(f"{'Mean Dice (F1):':<25}{results['mDice']:.4f}")
        logger.info("\nPer-Class Evaluation Metrics:")
        logger.info(f"{'Class':<18}{'IoU':>10}{'Dice':>10}{'Prec.':>10}{'Recall':>10}")
        logger.info("-" * 58)

        for idx, (iou, dice, prec, recall) in enumerate(zip(
            results['IoU_per_class'],
            results['Dice_per_class'],
            results['Precision_per_class'],
            results['Recall_per_class']
        )):
            class_name = cfg['class_distribution']['id_to_class'].get(idx, f"Class {idx}")
            logger.info(f"{class_name:<18}{iou.item():>10.4f}{dice.item():>10.4f}{prec.item():>10.4f}{recall.item():>10.4f}")
