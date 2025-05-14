import sys
sys.path.append(r"C:\Users\daand\RUG\applied ml\project\U-Net-SSeg")

import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

from src.data.dataset import SegmentationDataset
from src.utils.read_config import read_config
from src.data.transforms import ToTensor, Resize, Compose
from src.utils.resolve_path import resolve_path

def calculate_class_distribution() -> None:
    """
    Calculates the class distribution for the given training set. 
    Saves distribution to cfg.yaml.
    """
    
    colormap_id = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }
    
    cfg = read_config()
    print(tuple(cfg["transforms"]["resize"]))
    
    transform = Compose([
        Resize(cfg["transforms"]["resize"]),
        ToTensor()
    ])

    train_dataset = SegmentationDataset(
        cfg['data']['train_images'],
        cfg['data']['train_masks'], 
        transforms=transform,
    )
    
    train_loader = DataLoader(train_dataset)
    pixel_counts = defaultdict(int)
    
    for _ , masks in train_loader:
        unique, counts = np.unique(masks, return_counts=True)
        for class_mask , count in zip(unique, counts):
            if class_mask in colormap_id.keys():
                pixel_counts[colormap_id[class_mask]] += count

    class_distribution = {class_name: int(total_pixels_class) for class_name, total_pixels_class in pixel_counts.items()}
    
    path = resolve_path("cfg.yaml", 2)
    
    cfg['class_distribution'] = class_distribution
    
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
if __name__ == '__main__': 
    calculate_class_distribution()