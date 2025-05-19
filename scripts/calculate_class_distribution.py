from collections import defaultdict

import numpy as np
import yaml

from src.data import get_dataloader
from src.utils import read_config, resolve_path, write_config


def calculate_class_distribution() -> None:
    """
    Calculates the class distribution for the given training set.
    Saves distribution to cfg.yaml.
    """

    cfg = read_config()

    colormap_id = cfg['class_distribution']["id_to_class"]

    train_loader = get_dataloader(cfg=cfg, train=True, batch_size=8, debug=False, stats=True)
    pixel_counts = defaultdict(int)

    for _ , masks in train_loader:
        unique, counts = np.unique(masks, return_counts=True)
        for class_mask , count in zip(unique, counts):
            if class_mask in colormap_id.keys():
                pixel_counts[colormap_id[class_mask]] += count

    total_pixels = int(sum(pixel_counts.values()))
    class_distribution = {class_name: int(total_pixels_class) for class_name, total_pixels_class in pixel_counts.items()}

    path = resolve_path("cfg.yaml")

    cfg['class_distribution']['total_pixels'] = total_pixels
    cfg['class_distribution']['class_frequencies'] = class_distribution

    write_config(cfg)

if __name__ == '__main__':
    calculate_class_distribution()
