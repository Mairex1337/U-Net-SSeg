import torch
import yaml

from src.data.dataloader import get_dataloader
from src.utils import read_config, resolve_path, write_config


def calculate_mean_std() -> None:
    """
    Compute per-channel mean and standard deviation from the training dataset,
    then write the values to the 'normalize' section of cfg.yaml.
    """
    cfg = read_config()

    train_loader = get_dataloader(cfg=cfg, split="train", batch_size=8, stats=True)

    channel_sums = torch.zeros(3)
    channel_sqr_sums = torch.zeros(3)
    total_pixels = 0

    for image, _ in train_loader:
        b, c, h, w = image.shape
        total_pixels += b * h * w
        channel_sums += image.sum(dim=[0, 2, 3])
        channel_sqr_sums += (image ** 2).sum(dim=[0, 2, 3])

    mean = channel_sums / total_pixels
    var = (channel_sqr_sums / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    path = resolve_path("cfg.yaml")

    cfg['transforms']['normalize'] = {
        'mean':mean.tolist(),
        'std': std.tolist(),
    }

    write_config(cfg)

if __name__ == '__main__':
    calculate_mean_std()
