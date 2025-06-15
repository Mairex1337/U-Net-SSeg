import cv2
import numpy as np

from src.utils.path import read_config


def convert_grayscale_to_colored_mask(
    image_path: str
) -> np.ndarray:
    """
    Converts a grayscale mask image into a colorized mask using a colormap.

    Args:
        image_path (str): Path to the grayscale image file.
        save_path (Optional[str]): If provided, the colorized mask is saved to this path.

    Returns:
        np.ndarray: The colorized RGB mask as a NumPy array (in BGR format for OpenCV).
    Raises:
        FileNotFoundError: If the image file does not exist or cannot be read.
    """
    cfg = read_config()
    old_to_new = cfg['oldid_newid']
    colormap = cfg['class_distribution']['color_map']

    new_to_old = {}
    for old_id, new_id in old_to_new.items():
        if new_id == 255:
            continue
        new_to_old.setdefault(new_id, old_id)

    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for new_id, old_id in new_to_old.items():
        color = colormap[int(old_id)]
        color_mask[mask == int(new_id)] = color

    return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
