import os
from typing import Literal

from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset import SegmentationDataset
from src.data.transforms import (get_stats_transforms, get_train_transforms,
                                 get_val_transforms)
from src.utils import resolve_path


def get_dataloader(
    cfg: dict,
    split: Literal["train", "val", "test"],
    world_size: int = 1,
    rank: int = 0,
    batch_size: int = 8,
    debug: bool = False,
    stats: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader object for the specified split.

    Args:
        cfg: Dictionary of parsed cfg.yaml file.
        split (Literal): Signifies the split for the dataloader.
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
    assert split in ["train", "val", "test"], "`split` must be in ['train', 'val', 'test']"

    img_key = f"{split}_images"
    mask_key = f"{split}_masks"
    img_dir = cfg["data"][img_key]
    mask_dir = cfg["data"][mask_key]

    if world_size > 1:
        tmp_dir = os.environ["TMPDIR"]
        img_dir = os.path.join(tmp_dir, img_dir)
        mask_dir = os.path.join(tmp_dir, mask_dir)
    else:
        img_dir, mask_dir = resolve_path(img_dir), resolve_path(mask_dir)
    if stats:
        transforms = get_stats_transforms(cfg['transforms'])
    else:
        transform_map = {
            "train": get_train_transforms,
            "val": get_val_transforms,
            "test": get_val_transforms  # or define `get_test_transforms`
        }
        transforms = transform_map[split](cfg["transforms"])

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
            shuffle=(split == "train")
        )
    else:
        sampler = None

    num_workers = min(16, os.cpu_count() // world_size)
    if os.getenv("DISABLE_WORKERS") == "1":
        num_workers = 0
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train") if sampler is None else False,
        sampler = sampler,
        pin_memory=(sampler is not None),
        num_workers=num_workers,
        persistent_workers=True if sampler is not None else False
    )

    return dataloader
