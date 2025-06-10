import numpy as np
from PIL import Image

from src.utils import read_config

def transform_classes(mask: Image) -> Image:
    """
    Transform class labels in a masks according to configuration.

    Args:
        mask (Image): Mask for segmentation task.

    Returns:
        Image: Mask with transformed class labels, where specified classes 
        are combined or excluded based on configuration.
    """
    cfg = read_config()
    mask = np.array(mask)

    classes_to_combine = cfg['classes_to_transform']['classes_to_combine']
    classes_to_exclude = cfg['classes_to_transform']['classes_to_exclude']
    
    for combine_items in classes_to_combine.items():
        for cls in combine_items[1]:
            mask[mask == cls] = combine_items[0]
    
    exclude_class_id = 255
    for cls in classes_to_exclude:
        mask[mask == cls] = exclude_class_id
        
    mask = Image.fromarray(mask)

    return mask