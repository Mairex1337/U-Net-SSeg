from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAccuracy,
    MulticlassF1Score,
    JaccardIndex
)
import torch

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

        self.iou = JaccardIndex(task="multiclass", num_classes=num_classes, average=None).to(device)
        self.dice = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average=None).to(device)
        self.pixel_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        self.mean_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)

        self.metrics = [
            self.iou,
            self.dice,
            self.precision,
            self.recall,
            self.pixel_accuracy,
            self.mean_accuracy,
        ]

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update all metrics with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predicted class indices of shape (B, H, W)
            targets (torch.Tensor): Ground truth class indices of shape (B, H, W)
        """
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        for metric in self.metrics:
            metric.update(preds_flat, targets_flat)

    def compute(self) -> dict:
        """
        Compute all aggregated and per-class metrics.

        Returns:
            dict: A dictionary containing:
                - 'IoU_per_class' (Tensor): Per-class IoU values.
                - 'Dice_per_class' (Tensor): Per-class Dice/F1 scores.
                - 'Precision_per_class' (Tensor): Per-class precision.
                - 'Recall_per_class' (Tensor): Per-class recall.
                - 'Pixel_Accuracy' (float): Overall pixel-level accuracy.
                - 'Mean_Accuracy' (float): Mean per-class accuracy.
                - 'mIoU' (float): Mean IoU over all classes.
                - 'mDice' (float): Mean Dice/F1 score over all classes.
        """
        return {
            "IoU_per_class": self.iou.compute().cpu(),
            "Dice_per_class": self.dice.compute().cpu(),
            "Precision_per_class": self.precision.compute().cpu(),
            "Recall_per_class": self.recall.compute().cpu(),
            "Pixel_Accuracy": self.pixel_accuracy.compute().item(),
            "Mean_Accuracy": self.mean_accuracy.compute().item(),
            "mIoU": self.iou.compute().mean().item(),
            "mDice": self.dice.compute().mean().item()
        }

    def reset(self):
        """
        Reset all internal metric states for reuse on a new dataset or evaluation run.
        """
        for metric in self.metrics:
            metric.reset()
