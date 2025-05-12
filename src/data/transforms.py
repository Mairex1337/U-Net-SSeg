from typing import Any, Dict

from torchvision import transforms


def get_transforms(cfg: Dict[str, Any], mask: bool) -> transforms.Compose:
    """
    Builds a transform pipeline from a config dict.

    Args:
        cfg (Dict[str, Any]): Key value pairs of transforms for mask/img.
        is_mask (bool): Signifies whether transforms are for mask or img.

    Returns:
        transforms.Compose: Composed transform pipeline.

    Raises:
        ValueError: If required keys ('resize' or 'normalize') are missing.
    """
    operations = []
    if "resize" in cfg:
        operations.append(transforms.Resize(cfg["resize"]))
    else:
        raise ValueError("No resize transform found in cfg.")
    if "flip" in cfg:
        operations.append(transforms.RandomHorizontalFlip(cfg["flip"]))
    if mask:
        operations.append(transforms.PILToTensor())
    else:
        operations.append(transforms.ToTensor())
        if "normalize" not in cfg:
            raise ValueError("No normlize transform found in cfg.")
        operations.append(transforms.Normalize(
            mean=cfg["normalize"]["mean"],
            std=cfg["normalize"]["std"]
        ))

    return transforms.Compose(operations)