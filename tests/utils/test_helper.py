from typing import Any, Dict
from unittest.mock import patch

from src.data.dataloader import get_dataloader


class IdentityTransform:
    def __call__(self, img, mask) -> tuple:
        """Returns the input image and mask without any transformation."""
        return img, mask


def get_batch(cfg: Dict[str, Any], split: str = "train", batch_size: int = 1, debug: bool = True, stats: bool = False) -> tuple:
    """Returns the first batch from a dataloader.

    Args:
        cfg: Dataset configuration.
        split: Dataset split (train/val/test).
        batch_size: Batch size.
        debug: Whether to include additional debug outputs.
        stats: Whether to include statistics in the output.

    Returns:
        Tuple of tensors representing one batch.
    """
    return next(iter(get_dataloader(cfg, split=split, batch_size=batch_size, debug=debug, stats=stats)))


def patch_all_except(*exclude):
    """Decorator to patch all transform classes with IdentityTransform except those specified.

    Args:
        *exclude: Names of transform classes to leave unpatched.

    Returns:
        Callable decorator to apply patches to a test function.
    """
    def decorator(func):
        patches = {
            "RandomScale": lambda a, b, c: IdentityTransform(),
            "PadIfSmaller": lambda size: IdentityTransform(),
            "RandomCrop": lambda size: IdentityTransform(),
            "RandomHorizontalFlip": lambda p: IdentityTransform(),
        }
        for key in exclude:
            patches.pop(key, None)
        for path, val in reversed(patches.items()):
            func = patch(f"src.data.transforms.{path}", val)(func)
        return func
    return decorator
