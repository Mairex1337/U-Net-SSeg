from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(
        loss_name: Literal['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice'],
        cfg: dict,
        device: str,
) -> nn.Module:
    """
    Returns the specified loss criterion based on its name and configuration.

    Args:
        loss_name (str): The name of the loss function to retrieve.
                         Expected values: "weighted_cle", "dice", "mixed_cle_dice", "OHEMLoss".
        cfg (dict): A dictionary containing configuration parameters for the loss.
        device (str): The device ("cpu" or "cuda") where tensors should be placed.

    Returns:
        nn.Module: An instance of the requested loss criterion.

    Raises:
        AssertionError: If an unknown loss_name is provided.
    """
    assert loss_name in ['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice']
    params = cfg['hyperparams']['unet']
    num_classes = params['num_classes']
    ignore_index = params['ignore_index']
    if loss_name == 'weighted_cle':
        return get_weighted_criterion(cfg, device)
    
    if loss_name == 'OHEMLoss':
        return OHEMLoss(num_classes, ignore_index, params['topk_percent'])

    if loss_name == 'mixed_cle_dice':
        return MixedDiceCle(
            num_classes,
            params['cle_weight'],
            params['dice_weight'],
            ignore_index,
        )
    
    if loss_name == 'dice':
        return DiceLoss(num_classes, ignore_index)


def get_weighted_criterion(cfg: dict, device: str) -> torch.nn.CrossEntropyLoss:
    """
    Returns a weighted CrossEntropyLoss for semantic segmentation.

    Computes class weights from pixel frequencies provided,
    applying sqrt scaling and normalization range [0, 1]. Ignores the class index 255.

    Args:
        cfg (dict): Configuration dictionary
        device (str): Contains the device used for training

    Returns:
        torch.nn.CrossEntropyLoss: Weighted loss function with ignore_index set to 255.
    """
    frequencies = cfg["class_distribution"]["class_frequencies"]
    total_pixels = cfg["class_distribution"]["total_pixels"]
    id_to_class = cfg["class_distribution"]["id_to_class"]
    weights_by_id = torch.tensor([
        total_pixels / frequencies[id_to_class[i]]
        for i in range(len(id_to_class))
    ], device=device)
    weights_sqrt = torch.sqrt(weights_by_id)
    normalized_weights = weights_sqrt / weights_sqrt.sum()
    return torch.nn.CrossEntropyLoss(weight=normalized_weights, ignore_index=255)


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class semantic segmentation.

    Computes the Dice loss per class and averages over classes that are present
    in the ground truth. Non-present classes are ignored.

    Args:
        num_classes (int): Total number of classes in the segmentation task.
        ignore_index (int): Label index to ignore during loss computation. 
            Pixels with this label are excluded from the Dice calculation.
        eps (float): Smoothing term to avoid division by zero.
    """
    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes Dice loss between predictions and ground truth.

        Args:
            preds (torch.Tensor): logits, shape (B, C, H, W)
            targets (torch.Tensor): class indices, shape (B, H, W)
        
        Returns:
            float: The computed loss
        """
        preds = F.softmax(preds, dim=1)

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        targets = targets * valid_mask  # ignored pixels → 0

        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()  # shape (B, 1, H, W)

        preds = preds * valid_mask
        targets_onehot = targets_onehot * valid_mask

        intersection = (preds * targets_onehot).sum(dim=(0, 2, 3))
        union = preds.sum(dim=(0, 2, 3)) + targets_onehot.sum(dim=(0, 2, 3))
        dice_per_class = (2 * intersection + self.eps) / (union + self.eps)

        present = targets_onehot.sum(dim=(0, 2, 3)) > self.eps

        dice = dice_per_class[present].mean() if present.any() else dice_per_class.mean()

        return 1 - dice


class MixedDiceCle(nn.Module):
    """
    Mixed Dice/Cle Loss for multi-class semantic segmentation.

    Args:
        num_classes (int): Total number of classes in the segmentation task.
        cle_weight (float): weight scalar for cle part (must be in [0.0, 1.0])
        dice_weight (float): weight scalar for dice part (must be in [0.0, 1.0])
        ignore_index (int): Label index to ignore during loss computation. 
        eps (float): Smoothing term to avoid division by zero.
    """
    def __init__(
            self,
            num_classes: int,
            cle_weight: float = 0.5,
            dice_weight: float = 0.5,
            ignore_index: int = 255,
            eps: float = 1e-6
        ) -> None:
        super().__init__()
        assert 0.0 <= cle_weight <= 1.0 and 0.0 <= dice_weight <= 1.0
        assert abs(cle_weight + dice_weight - 1.0) < 1e-6
        self.cle = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index, eps)
        self.cle_weight = cle_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            logits (torch.Tensor): B, C, H, W raw outputs
            targets (torch.Tensor): B, H, W ground truth class indices

        Returns:
            loss (float): mixed loss
        """
        cle = self.cle(logits, targets)
        dice = self.dice(logits, targets)
        total = (self.cle_weight * cle) + (self.dice_weight * dice)
        return total


class OHEMLoss(nn.Module):
    """
    Computes CEL only for the top-k percent hardest pixels.

    Args:
        num_classes (int): number of classes
        ignor_index (int): index of pixels to ignore
        top_k_percent (float): float in (0,1], e.g. 0.25 means top 25% hardest pixels
    """
    def __init__(
            self,
            num_classes: int,
            ignore_index: int = 255,
            top_k_percent: float = 0.25
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.topk = top_k_percent

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            logits (torch.Tensor): B, C, H, W raw outputs
            targets (torch.Tensor): B, H, W ground truth class indices

        Returns:
            loss: scalar loss over top-k hardest pixels
        """
        C = self.num_classes
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets = targets.view(-1)  # (B*H*W)

        valid_mask = (targets != self.ignore_index)
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        pixel_loss = F.cross_entropy(logits, targets, reduction='none')  # (N,)

        k = int(self.topk * pixel_loss.numel())
        topk_loss, _ = torch.topk(pixel_loss, k, sorted=False)

        return topk_loss.mean()


class EarlyStopping:
    """
    Early stops training if the monitored metric(s) don't improve after a given patience.

    Args:
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
        mode (str): 'min' for val_loss, 'max' for metrics like IoU.
    """
    def __init__(self, patience=10, min_delta=1e-4, mode='max') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def get_metric_score(self, results: dict) -> float:
        """
        Determine total score from mean accuracy, mIoU, and mDice.

        Args:
            results (dict): Results dictionary containing metrics.

        Returns:
            float: Sum of the mean metrics
        """
        s = 0.0
        for k, v in results.items():
            if k in ["Mean_Accuracy", "mIoU", "mDice"]:
                s += v
        return s

    def __call__(self, current_score: float) -> bool:
        """
        Determine whether to early stop training.

        Args:
            current_score (float): Metric score to be used for early stopping.

        Returns:
            bool: Boolean indicating whether or not to early stop training.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (
            (current_score - self.best_score) > self.min_delta
            if self.mode == 'max'
            else (self.best_score - current_score) > self.min_delta
        )

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop