from collections import defaultdict

import numpy as np
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from src.data.dataloader import get_dataloader
from src.data.dataset import SegmentationDataset
from src.data.transforms import Compose, Resize, ToTensor
from src.utils.read_config import read_config
from src.utils.resolve_path import resolve_path


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

    path = resolve_path("cfg.yaml", 2)

    cfg['class_distribution']['total_pixels'] = total_pixels
    cfg['class_distribution']['class_frequencies'] = class_distribution

    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

if __name__ == '__main__':
    calculate_class_distribution()
