from copy import deepcopy

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset
from src.utils.read_config import read_config
from src.utils.resolve_path import resolve_path


def calculate_mean_std() -> None:
    """
    
    """
    transform = transforms.Compose([
        transforms.Resize((456,256)),
        transforms.ToTensor()
    ])
    cfg = read_config()

    train_dataset = SegmentationDataset(
        resolve_path(cfg['data']['train_images'], 2),
        resolve_path(cfg['data']['train_masks'], 2), 
        img_transforms=transform,
        mask_transforms=transform
    )
    train_loader = DataLoader(train_dataset)

    channel_sums = torch.zeros(3)
    channel_sqr_sums = torch.zeros(3)
    total_pixels = 0

    for image, _ in train_loader:
        image = image.squeeze()
        _, h, w = image.shape
        total_pixels += h * w
        channel_sums += image.sum(dim=[1, 2])
        channel_sqr_sums += (image ** 2).sum(dim=[1, 2])
        
    mean = channel_sums / total_pixels
    var = (channel_sqr_sums / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    mean = mean.tolist()
    std = std.tolist()

    path = resolve_path(cfg.yaml, 2)

    with open(path, 'r') as f:
        cfg = yaml.unsafe_load(f)

    cfg['normalization'] = {'mean': deepcopy(mean), 'std': deepcopy(std)}
    
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

if __name__ == '__main__': 
    calculate_mean_std()