import numpy as np
from PIL import Image

from src.utils.path import read_config


def transform_classes(mask: Image.Image) -> Image.Image:
    """
    Transform class labels in a segmentation mask using oldid_newid mapping.

    Args:
        mask (Image): Input mask with original label IDs.

    Returns:
        Image: Transformed mask with new label IDs.
    """
    cfg = read_config()
    old_new = cfg['oldid_newid']

    mask_np = np.array(mask)

    for old_id, new_id in old_new.items():
        mask_np[mask_np == int(old_id)] = int(new_id)

    return Image.fromarray(mask_np)