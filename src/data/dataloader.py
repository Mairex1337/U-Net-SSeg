import os

from torch.utils.data import DataLoader

from src.data.dataset import SegmentationDataset


def get_dataloader(
    img_dir: str,
    mask_dir: str,
    img_transforms,
    mask_transforms,
    batch_size: int = 8,
):
    """
    """
    #TODO: need fix for test set where we have no masks
    ds = SegmentationDataset(
        img_dir,
        mask_dir,
        img_transforms=img_transforms,
        mask_transforms=mask_transforms
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
    )


    return dataloader
