import os

import torch.utils.data as data
from PIL import Image
from torch import Tensor

from src.utils.resolve_path import resolve_path


class SegmentationDataset(data.Dataset):
    def __init__(
            self,
            img_dir: str,
            mask_dir: str,
            img_transforms: list = None, # These typehints are still not correct
            mask_transforms = None
    ) -> None:
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.img_dir = resolve_path(img_dir)
        self.mask_dir = resolve_path(mask_dir)
        self.img_idx = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
        ])
        self.mask_idx = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.mask_dir)
        ])
        assert self.img_idx == self.mask_idx, "Image and Mask names do not match."

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path = os.path.join(self.img_dir, f'{self.img_idx[idx]}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{self.img_idx[idx]}.png')
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.img_transforms:
            img = self.img_transforms(img)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        return img, mask

    def __len__(self) -> int:
        return len(self.img_idx)
