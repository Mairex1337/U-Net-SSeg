import os
from copy import deepcopy
import sys
sys.path.append(r"C:\Users\daand\RUG\applied ml\project\U-Net-SSeg") 

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

from src.data.dataset import SegmentationDataset
from src.utils.read_config import read_config

def calculate_class_distribution():
    """
    """
    
    #ids are from the label.py in the bdd100k
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
    
    transform = transforms.Compose([
        transforms.Resize((256,448)),
        transforms.ToTensor()
    ])
    
    cfg = read_config()

    train_dataset = SegmentationDataset(
        cfg['data']['train_images'],
        cfg['data']['train_masks'], 
        img_transforms=transform,
        mask_transforms=transform
    )
    train_loader = DataLoader(train_dataset)
    pixel_counts = defaultdict(int)
    
    for _ , masks in train_loader:
        unique, counts = np.unique(masks, return_counts=True)
        for c, count in zip(unique, counts):
            c = int(c*255)
            if c in colormap_id.keys():
                pixel_counts[colormap_id[c]] += count
    
    average_pixel_counts = {c: int(total / len(train_loader)) for c, total in pixel_counts.items()}
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    cfg['class_distribution'] = average_pixel_counts
    
    with open(os.path.join(project_root, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
if __name__ == '__main__': 
    calculate_class_distribution()