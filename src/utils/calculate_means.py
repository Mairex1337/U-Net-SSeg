import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset
from src.utils.read_config import read_config
from src.utils.resolve_path import resolve_path
from src.data.transforms import ToTensor, Resize, Compose


def calculate_mean_std() -> None:
    """
    Compute per-channel mean and standard deviation from the training dataset,
    then write the values to the 'normalize' section of cfg.yaml.
    """
    cfg = read_config()
    
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
    
    path = resolve_path("cfg.yaml", 2)

    cfg['transforms']['normalize'] = {
        'mean':mean.tolist(),
        'std': std.tolist(),
    }
    
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

if __name__ == '__main__': 
    calculate_mean_std()