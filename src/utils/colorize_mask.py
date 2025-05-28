import cv2
import numpy as np
from src.utils import read_config


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
    colormap = cfg['class_distribution']['color_map']

    gray_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_mask is None:
        raise FileNotFoundError(f"Image not found or could not be read: {image_path}")

    color_mask = np.zeros((*gray_mask.shape, 3), dtype=np.uint8)

    for label, color in colormap.items():
        color_mask[gray_mask == label] = color

    return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
