import os

from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import SegmentationDataset
from src.data.transforms import (get_stats_transforms, get_train_transforms,
                                 get_val_transforms)


def get_dataloader(
    cfg: dict,
    train: bool,
    world_size: int = 1,
    rank: int = 0,
    batch_size: int = 8,
    debug: bool = False,
    stats: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader object.

    Args:
        cfg: Dictionary of parsed cfg.yaml file.
        train (bool): Signifies if this is the train or val dataloader.
        world_size (int): Number of processes when training distributed.
        rank (int): Device rank when training distributed.
        batch_size (int): Number of samples per batch. Default is 8.
        debug (bool): Indicates debug mode. If true, DataLoader also returns
            original images and masks within the batch.
        stats (bool): If True, return data without Normalize/Augment
            (for mean/std calc).

    Returns:
        DataLoader: A PyTorch DataLoader that samples from dataset.
    """
    img_dir = cfg['data']['train_images'] if train else cfg['data']['val_images']
    mask_dir = cfg['data']['train_masks'] if train else cfg['data']['val_masks']
    if stats:
        transforms = get_stats_transforms(cfg['transforms'])
    else:
        transforms = get_train_transforms(cfg['transforms']) if train else get_val_transforms(cfg['transforms'])

    ds = SegmentationDataset(
        img_dir,
        mask_dir,
        transforms=transforms,
        debug=debug
    )

    if world_size > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=train
        )
    else:
        sampler = None

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train if sampler is None else False,
        sampler = sampler,
        pin_memory=(sampler is not None),
        num_workers=os.cpu_count() // 2,
    )

    return dataloader
