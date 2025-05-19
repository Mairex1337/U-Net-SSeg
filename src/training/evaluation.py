import torch


def get_weighted_criterion(cfg: dict, device: str) -> torch.nn.CrossEntropyLoss:
    """
    Returns a weighted CrossEntropyLoss for semantic segmentation.

    Computes class weights from pixel frequencies provided,
    applying inverse frequency normalization. Ignores the class index 255.

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
        total_pixels / (len(id_to_class) * frequencies[id_to_class[i]])
        for i in range(len(id_to_class))
    ], device=device)
    return torch.nn.CrossEntropyLoss(weight=weights_by_id, ignore_index=255)
