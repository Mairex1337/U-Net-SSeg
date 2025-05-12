import os

from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset
from src.data.transforms import get_train_transforms, get_val_transforms


def get_dataloader(
    cfg: dict,
    train: bool,
    batch_size: int = 8,
    debug: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader object.

    Args:
        cfg: Dictionary of parsed cfg.yaml file.
        train (bool): Signifies if this is the train or val dataloader.
        batch_size (int): Number of samples per batch. Default is 8.
        debug (bool): Indicates debug mode. If true, DataLoader also returns
            original images and masks within the batch.

    Returns:
        DataLoader: A PyTorch DataLoader that samples from dataset.
    """
    img_dir = cfg['data']['train_images'] if train else cfg['data']['val_images']
    mask_dir = cfg['data']['train_masks'] if train else cfg['data']['val_masks']
    transforms = get_train_transforms(cfg['transforms']) if train else get_val_transforms(cfg['transforms'])

    ds = SegmentationDataset(
        img_dir,
        mask_dir,
        transforms=transforms,
        debug=debug
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
    )

    return dataloader
