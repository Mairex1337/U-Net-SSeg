import torch


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


class EarlyStopping:
    """
    Early stops training if the monitored metric doesn't improve after a given patience.

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

    def __call__(self, current_score: float) -> bool:
        """
        Determine whether to early stop training.

        Args:
            current_score (float): Score on the metric for current epoch.

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